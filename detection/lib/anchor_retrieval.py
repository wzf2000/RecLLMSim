"""
Anchor turn 检索器（用于个性化满意度评估 prompt 的 few-shot in-context 锚点）

动机
----
Memory v2 的 turn eval prompt 是抽象 rubric（文本描述的评分边界）。从诊断
（reports/diagnose_confusion.md）观察到：
  - GPT-4o-mini 把 51% 的真实 5 分预测为 4 分
  - Qwen3-8B 把 54% 的真实 5 分预测为 4 分
说明两个模型都读懂了 rubric 但无法把 "当前 turn" 和 rubric 里的抽象描述
精确对上。本检索器从历史 turns 中挑出与当前 assistant 回复最相似的
k 条带标签的参考轮，直接作为 few-shot 锚点插入 prompt。

设计
----
- 字符 3-gram TF-IDF + 余弦相似度（零额外依赖，对中文友好）
- 每个 PersonalizedSample 构建一次 retriever（over 所有 history turns）
- 检索时同时考虑当前 assistant reply + 最近一条 user query
- k 个锚点中强制按分数多样性采样（尽量覆盖不同分数档），避免全部是同一分数
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .personalized_data import SessionData


@dataclass
class AnchorTurn:
    """用于 prompt 展示的锚点轮次。"""
    task: str
    user_msg: str
    assistant_reply: str
    score: int
    reason: str
    evidence_role: str = ""


class AnchorRetriever:
    """
    基于 TF-IDF 字符 n-gram 的历史 turn 检索器。

    用法：
        retriever = AnchorRetriever(history_sessions)
        anchors = retriever.retrieve(
            query_user_msg=..., query_assistant_reply=..., k=3,
        )
    """

    def __init__(
        self,
        history_sessions: list[SessionData],
        char_ngram_range: tuple[int, int] = (2, 4),
        min_df: int = 1,
    ) -> None:
        self._turns: list[AnchorTurn] = self._extract_turns(history_sessions)
        if not self._turns:
            self._vectorizer = None
            self._matrix = None
            return

        corpus = [self._turn_to_text(t) for t in self._turns]
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    # ──────────────────────────────────────────────────────────────────────────
    # 构建阶段
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_turns(sessions: list[SessionData]) -> list[AnchorTurn]:
        out: list[AnchorTurn] = []
        for session in sessions:
            assistant_idx = 0
            last_user = ""
            for utt in session.history:
                if utt["role"] == "user":
                    last_user = utt["content"]
                elif utt["role"] == "assistant":
                    if assistant_idx >= len(session.satisfaction_scores):
                        break
                    score = session.satisfaction_scores[assistant_idx]
                    reason = session.dissatisfaction_reasons[assistant_idx]
                    out.append(AnchorTurn(
                        task=session.task,
                        user_msg=last_user,
                        assistant_reply=utt["content"],
                        score=score,
                        reason=reason,
                    ))
                    assistant_idx += 1
        return out

    @staticmethod
    def _turn_to_text(turn: AnchorTurn) -> str:
        """拼接用于索引的文本。

        策略：user_msg 权重高于 assistant_reply，这样检索是【按问题相似性】而非
        【按回复风格相似性】——避免 top-k 总是命中"长篇高分"回复而引入偏置。
        """
        return f"{turn.user_msg} {turn.user_msg} {turn.assistant_reply}"

    @staticmethod
    def _query_to_text(user_msg: str, assistant_reply: str) -> str:
        return f"{user_msg} {user_msg} {assistant_reply}"

    # ──────────────────────────────────────────────────────────────────────────
    # 检索阶段
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_user_msg: str,
        query_assistant_reply: str,
        k: int = 3,
        diversify_by_score: bool = False,
    ) -> list[AnchorTurn]:
        """
        返回与当前 turn 最相似的 k 条历史 turn。

        diversify_by_score=True 时，按分数分组，每组取最相似的一条再合并排序，
        保证锚点覆盖不同分数档（便于模型对比判断）。
        """
        if self._vectorizer is None or not self._turns or k <= 0:
            return []

        query_text = self._query_to_text(query_user_msg, query_assistant_reply)
        qv = self._vectorizer.transform([query_text])
        sims = cosine_similarity(qv, self._matrix)[0]

        if not diversify_by_score:
            top_idx = sims.argsort()[::-1][:k]
            return [self._turns[int(i)] for i in top_idx]

        # ── 按分数分组取最相似，再合并取 topk ────────────────────────────────
        by_score: dict[int, list[tuple[float, int]]] = {}
        for i, s in enumerate(sims):
            sc = self._turns[i].score
            by_score.setdefault(sc, []).append((float(s), i))
        for sc in by_score:
            by_score[sc].sort(reverse=True)

        # 策略：先为每个出现的分数取 1 条最相似，再按相似度排序取 topk
        picks: list[tuple[float, int]] = [
            grp[0] for grp in by_score.values() if grp
        ]
        # 如果 k > 出现的分数种类数，从剩下的里继续补
        if len(picks) < k:
            used = {idx for _, idx in picks}
            remainders: list[tuple[float, int]] = []
            for grp in by_score.values():
                for s, idx in grp[1:]:
                    if idx not in used:
                        remainders.append((s, idx))
            remainders.sort(reverse=True)
            need = k - len(picks)
            picks.extend(remainders[:need])

        picks.sort(key=lambda x: (-self._turns[x[1]].score, -x[0]))
        picks = picks[:k]
        return [self._turns[idx] for _, idx in picks]

    def retrieve_boundary_paired(
        self,
        query_user_msg: str,
        query_assistant_reply: str,
        k: int = 4,
    ) -> list[AnchorTurn]:
        """
        返回成对的 3/4 边界证据。

        与普通 top-k 不同，该方法尽量同时取相似的不满意侧（<=3）和满意侧（>=4）
        历史轮次，用于让 judge 对当前回复做 paired comparison，而不是只参考单侧
        高分或低分案例。
        """
        if self._vectorizer is None or not self._turns or k <= 0:
            return []

        query_text = self._query_to_text(query_user_msg, query_assistant_reply)
        qv = self._vectorizer.transform([query_text])
        sims = cosine_similarity(qv, self._matrix)[0]

        dsat_budget = max(1, k // 2)
        sat_budget = max(1, k - dsat_budget)
        dsat = self._top_by_score_side(sims, lambda score: score <= 3, dsat_budget)
        sat = self._top_by_score_side(sims, lambda score: score >= 4, sat_budget)

        used = {idx for _, idx in dsat + sat}
        picks = dsat + sat
        if len(picks) < k:
            remainder = [
                (float(s), i)
                for i, s in enumerate(sims)
                if i not in used
            ]
            remainder.sort(reverse=True)
            picks.extend(remainder[: k - len(picks)])

        out: list[AnchorTurn] = []
        for _, idx in picks[:k]:
            turn = self._turns[idx]
            role = "DSAT-side evidence (score<=3)" if turn.score <= 3 else "SAT-side evidence (score>=4)"
            out.append(replace(turn, evidence_role=role))
        return out

    def _top_by_score_side(
        self,
        sims,
        score_predicate,
        k: int,
    ) -> list[tuple[float, int]]:
        candidates = [
            (float(s), i)
            for i, s in enumerate(sims)
            if score_predicate(self._turns[i].score)
        ]
        candidates.sort(reverse=True)
        return candidates[:k]

    @property
    def n_history_turns(self) -> int:
        return len(self._turns)
