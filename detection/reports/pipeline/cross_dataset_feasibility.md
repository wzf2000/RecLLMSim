# Memory-Agent 个性化满意度预测流程 — 跨数据集可行性分析

> 目的：评估当前基于 RecLLMSim 人工标注数据构建的【memory-agent training-free】流水线能否复用到其他公开数据集。重点是**流程能否接上**，而非字段表面差异（字段差异可通过 prompt 模板改写解决的不在讨论范围内）。

## 1. 当前流程的关键依赖

核心代码入口：`detection/lib/personalized_data.py`、`detection/lib/memory.py`、`detection/trace/collect_personalized.py`、`detection/eval/calibrate.py`。

把流水线拆成**五个硬依赖**（缺一不可）和**两个软依赖**（缺了可退化）：

| 依赖 | 说明 | 用在哪里 | 性质 |
|---|---|---|---|
| **D1. 持续的用户身份（user id）** | 同一用户在多个 session / 对话中出现，可聚合 | `load_all_sessions` 按 `user` 分组；memory cache 文件名 `{user}__{task}__{model}.json` | 硬 |
| **D2. 同一用户的多任务 session** | Cross-Task Split 需要至少 2 类任务：一类作 history（标签可见），一类作 target（待预测） | `build_personalized_samples` 里 `history_sessions = 其他任务的所有 session` | 硬 |
| **D3. 每用户多次打分样本** | memory v2 要求看到该用户的分数分布（≥ 几十次打分）才能抽取对比式评分边界 | `build_memory_prompt` 里按分数分组的证据区；CDF calibration 的 `score_distribution` | 硬 |
| **D4. 逐轮满意度标签（整数 1-5 或可离散化）** | 既是模型目标，也是评估的 gold | 所有 eval/predictor 路径 | 硬 |
| **D5. 对话结构（role=user/assistant 序列）** | rubric 里按 turn 审 assistant 回复 | `history` 字段 | 硬 |
| S1. 用户 profile（画像） | 辅助 memory 建模，缺失时可置空 | v2 `UserMemoryContent` | 软 |
| S2. 不满原因标签 | 仅 reason-prediction 子任务需要 | `dissatisfaction_reasons` | 软 |

以及一条**隐含假设**：

- **H. 用户评分风格跨任务迁移** — CDF 校准与 v2 对比边界都预设用户的评分基准（严 / 宽 / 中）不随任务剧变。本项目的 `reports/analysis/calibration_results.md` 在 4 类生活任务上验证了它基本成立。

---

## 2. USS（`detection/data/uss/`）

数据：5 个子集 JDDC / SGD / MWOZ / ReDial / CCPE，单位是 **dialogue**（`dialogue_id`），已在 `lib/uss_data.py` 里跑通 `predictor/lora_ordinal.py` 的**监督训练路线**，但**尚未接入 memory-agent 流水线**。

规模（当前 processed 数据）：

| 子集 | dialogues | turns | 平均轮数 | 语言 | 领域 |
|---|---:|---:|---:|---|---|
| JDDC  | 3300 | 30241 | 9.2  | zh | 电商客服 |
| SGD   | 1000 | 11833 | 11.8 | en | 任务对话 |
| MWOZ  | 1000 | 10553 | 10.6 | en | 多域 TOD |
| ReDial| 1000 | 6792  | 6.8  | en | 电影推荐 |
| CCPE  |  500 | 5180  | 10.4 | en | 偏好引导 |

**依赖对照**：

- D1（用户身份）：❌ **不满足**。`dialogue_id` 不等于 user id——原始语料每个 dialogue 来源不同（众包 worker 或客服对话对，无持久 user 标识）。这是核心阻塞点。
- D2（跨任务 session）：❌ 同一用户不存在，跨任务更谈不上。
- D3（每用户多次打分）：❌ 每 dialogue 独立评分，无 per-user 分布。
- D4（逐轮分数）：✅ 满足（1-5）。
- D5（对话结构）：✅ 满足。

**结论**：**memory-agent 个性化流程不能整块迁移**，但可以**降级**到三条更轻的路线：

- **R1. 对话内的 warm-up 校准**（最直接可用）
  将一条 dialogue 切成"前 n 轮做历史标签可见 + 后续轮作 target"。前 n 轮的 gold 分布当作"该 dialogue 的评分风格"，用它的 CDF / mean-shift 校准后续预测。
  - 等价把"用户"降级为"dialogue"，`score_distribution` 由同一 dialogue 已评过的 turn 构成。
  - 要求 n 轮足够多（建议 ≥5）才有分布意义；USS 平均 9-12 轮基本够用。
  - 这其实是对 CDF 校准的轻量自适应，代码改动很小：`calibrate.py` 读 memory_cache 的分支换成"读 input JSONL 中同一 `dialogue_id` 前 n 轮的 gold_score"。

- **R2. 子集级 rubric memory（population-level）**
  不再 per-user 建 memory，而是 per-subset 建一个"人群风格 rubric"（从训练 split 里随机采 K 条 dialogue、丢给 LLM 归纳出 `four_vs_five_distinction / three_vs_four_distinction` 等字段）。
  - 退化为"领域 rubric"，失去个性化，但保留 memory v2 的【对比式评分边界 prompt】，对**分布坍缩**和**系统性偏差**可能仍有帮助。
  - 可作为 USS 上 memory-agent 的**基线**对比。

- **R3. Dialogue-level 满意度**（利用 `*_overall.jsonl`）
  每 dialogue 只有 1 个 overall_score，本质是**分类任务**，与当前"逐轮 + memory" 路线目标不一致。只能用于评估 R2 的 rubric 是否对最终印象判断有帮助。

**决策建议**：USS 上应走 **R1 + R2**。CDF 校准（R1）做 post-hoc 后处理；memory-agent 走 R2 跑一条 subset-level rubric baseline，和已有 `predictor/lora_ordinal.py` 的监督线对比，检验 training-free 路线对 **zero-shot / 跨领域迁移** 有没有优势。

**改造量**：

- `lib/uss_data.py` 里要新增 `build_uss_personalized_samples()`，按"dialogue 切 warm-up"或"subset 分组"构造 `PersonalizedSample` 兼容对象；
- `collect_personalized.py` 的 `run_agent_on_sample` 无需大改，但 memory cache 命名要从 `{user}__{task}` 换成 `{dialogue_id}__{dataset}` 或 `{dataset}__population`；
- eval 链路可直接复用。

---

## 3. URS（`detection/data/urs/`）

数据：三个 JSON 文件。`chinese_merged.json`（515 session，107 user，zh）+ `english_processed.json`（305 session，74 user，en）。每条 item 结构：

```
conversation_history: [{role, content} × N]        # 完整对话
user_id, user_satisfaction, intent_label,
task_difficulty, title, llm
```

> **⚠️ 命名空间注意**：`all_conversations.json` 直接拼接了中英两个文件，但 **zh 与 en 的 `user_id` 是互相独立的命名空间**（zh 中的 `user_id=3` 和 en 中的 `user_id=3` 是不同用户）。实测：重合的 17 个 uid 在两语言下对话主题风格完全脱节（例：uid=3 zh 全是政经，en 全是闲聊；uid=8 zh 全是瑞士/日内瓦，en 全是 tech/娱乐），证实是 id 冲突而非跨语言同人。**接入 pipeline 时必须给 uid 加语言前缀**（如 `zh_3` / `en_3`），或仅处理其中一种语言；直接用 `all_conversations.json` 会把不同人的打分混成"同一用户"，污染 memory v2 的 `score_distribution`。

**规模与分布（正确命名空间下）**：

| 指标 | 数值 |
|---|---|
| Session 总数 | 820（zh 515 + en 305） |
| Unique users | **181**（zh 107 + en 74，无跨语言同人） |
| 每用户 session 数 min/avg/max | 1 / 4.53 / 13 |
| 每用户 intent 类别数 min/avg/max | 1 / 2.8 / 8 |
| 轮数 / session min/avg/max | 2 / 3.8 / 64 |
| 满意度 5 档分布（1→5） | 29 / 63 / 188 / 369 / 171 |
| LLM 来源 | ChatGPT 651，New Bing 38，ERNIE 34，Bard 21，Gemini 14，其他（Claude/Baichuan/Qwen/Llama/...） |

满意度标签天然是 **5 档 ordinal**，中英文一一映射：`很不满意 / dissatisfied → 1 … 非常满意 / very satisfied → 5`。intent_label 有 7 个 canonical 类（professional / retrieval / text / advice / creative / leisure / other），中英可直接规范化。

**依赖对照（基于 181 user 的正确命名空间）**：

- D1（用户身份）：✅ **满足**。加前缀后 181 user，平均每人 4.53 session。
- D2（跨任务 session）：✅ **满足**。canonical 化后 **145/181 (80.1%) 用户有 ≥2 类 intent**，cross-task split 直接可用（共 **426 个 valid (user, target_intent) pair**）。分语言看：zh 87/107 (81%) 268 pair，en 58/74 (78%) 158 pair。
- D3（每用户多次打分）：✅ **满足（略弱于 RecLLMSim）**。**114/181 (63%) 用户有 ≥5 个 session**，适合 memory v2 的 `score_distribution` 估计；另有 27 个 1-2 session 的用户可以 `min_history_sessions` 过滤掉。
- D4（1-5 分标签）：⚠️ **粒度不同**——**label 是 session-level，非 turn-level**。一整段对话共享 1 个满意度分数。
- D5（对话结构）：✅ 满足。
- S1（profile）：❌ 无用户画像字段。与现有 fallback 一致（置空）。
- S2（不满原因）：❌ 无 reason 标签。现有 pipeline 已对 reason 做过降级（仅 reason prediction 子任务受影响）。

**真正要处理的结构差异**是 D4 的 session-level 粒度。三条候选路线：

- **U1. Session-level 评分直接复用（推荐）**
  把整个 session 当成一条"item"，`pred_score` 直接在 session 粒度算；memory v2 的 `score_distribution` 按 session 统计（而非 turn）。**这条路线和现有 pipeline 在结构上完全对齐**——我们 RecLLMSim 数据虽然是逐轮标，但 v2 memory 核心就是"用户评分风格的全局分布"，天然是 session/user 汇总量。
  - 改造核心：`SessionData` 里新增一个 `session_score`，`satisfaction_scores` 退化为 `[session_score]`（单元素列表），`assistant_turns = 1`；memory 与 eval 只在 session 级评估。
  - CDF 校准直接适用（现有代码已按 block 聚合，block 粒度不变）。

- **U2. Turn-level 广播标签（辅助路线）**
  把 session-level 的一个分数广播给每个 assistant 回复作为 gold，复现 RecLLMSim 的 turn-level pipeline。**不推荐**——人为引入重复 label，MAE / Pearson 这些逐 turn 评估不反映真实语义（任何一轮预测对即可得高分）。仅当想复用 turn-level memory（逐 turn 的 `dissatisfaction_reasons_by_turn`）时作为兜底。

- **U3. Joint session + turn 评估（上限路线）**
  保留 session-level gold 作为**监督信号与评估主指标**，同时让 LLM 在 turn-level 输出 per-turn 预测（无 gold），用 aggregator（mean / max / min / last-turn）聚到 session 级和 gold 对比。副产物是：per-turn pseudo-label 可用于**未标注数据的 self-distillation**，但超出当前 scope。

**核心价值**：URS 是**唯一一个在依赖对照上几乎与 RecLLMSim 完全一致的公开数据集**——它有真实用户、跨任务、多 session、1-5 ordinal 分数。差异只有 session-level 粒度和缺少 profile/reason，都能通过 U1 路线"几乎无感"地接入现有 pipeline。

**对核心假设 H（评分风格跨任务迁移）的进一步验证价值**：RecLLMSim 是生活类规划任务（旅行/礼物/菜谱/学习），URS 是更通用对话（专业问题/检索/文本/建议/创意/闲聊）——**同一个 H 假设在不同任务分布下的稳健性**是很自然的消融。

**额外红利：天然的语言消融**
由于 zh 与 en 用户池完全不相交，可以干净地做 `train(zh) → test(en)` / `train(en) → test(zh)` 的跨语言迁移实验（不存在 user leakage），这是 RecLLMSim 无法提供的评估轴。

**潜在风险点**：

1. **每用户 session 数波动大**：1-13，其中 27 个只有 1-2 session 的用户在 cross-task 划分时只能落到 train 或被过滤。有 ≥5 session 的用户占 63%，比 RecLLMSim（基本全员 ≥4 个任务 × 多 session）稀疏。
2. **语言 + LLM 双重不均衡**：62% 中文 / 38% 英文；LLM 来源 79% ChatGPT。如果在 en 子集单独跑，用户量会缩到 74；memory v2 的 prompt 模板需要处理多语言与多模型（比如 `chat_model` 字段直接拼进 prompt 会让不同模型的打分风格互相串扰）。
3. **Intent 类别粒度比 RecLLMSim 粗**：7 类中 `retrieval`（32%）和 `professional`（20%）占大头，`other` 只有 11 条容易稀疏——`min_history_sessions` 过滤阈值需要调小（比如从 1 到 0 即允许"只看同任务历史"）或对 `other` 合并到邻近类别。
4. **缺少 profile**：memory v2 的 `build_memory_prompt` 对 profile 空的情况已有 fallback，但 prompt 里的画像段要整段删；可复用 OASST 处理模式。
5. **uid 命名空间陷阱**：实现时要强制 `zh_{uid}` / `en_{uid}` 前缀化；若不做则会悄无声息地把不同用户的分数合并到同一个 `score_distribution`，污染 memory 抽取。属于 **需要在代码里显式防御** 的点。

**改造量预估**：最核心的是数据加载器 + `SessionData` 粒度切换，评估链路几乎不需要动。

**结论**：**强烈建议作为 #1 优先级**。它是目前所有候选中对 memory-agent 流水线结构最无缝的数据集，可直接验证**"用户评分风格 + CDF 校准"在通用助手对话场景**的可迁移性。

---

## 4. OpenAssistant (OASST / OASST2)

原始数据组成：一棵棵"消息树"（message tree），包含用户提示、多个候选回复、以及若干 labeler 对每条消息打的 labels。每条 label 有 10+ 维度（quality / helpfulness / humor / creativity / toxicity / ...），多数是连续 [0,1] 或 ordinal。

**依赖对照（两种可能的映射方式分别看）**：

### 3.1 映射 A：把"prompter"（写 prompt 的用户）当作"user"

- D1：⚠️ **弱满足**。数据公布时 prompter 被保留为匿名 `user_id`，但**绝大多数 prompter 只贡献 1-2 条消息**（长尾分布），达不到 v2 memory 需要的数十条打分样本。
- D2：❌ 同一 prompter 的多个 prompt 没有"任务分类"标签；虽然有 message 级的 `labels` 字段，但不构成离散任务类型。
- D3：❌ 同上，不够。
- 结论：**不走得通**（prompter 端数据太稀疏）。

### 3.2 映射 B：把"labeler"当作"user"，消息/回复当作"item"

这是更自然的映射——本项目的"用户评分风格"对应 OASST 的"标注者评分风格"。

- D1：✅ labeler_id 公开且持久（除非被匿名化为哈希，OASST 公开数据里有 labeler id）。
- D2：⚠️ **需要构造任务分类**。OASST 没有原生的 task 字段，但每条 message 有 `lang`、`role`（prompter/assistant）、`parent_id` 结构，可以自定义伪任务类型（如 coding / writing / Q&A / translation），通过提示主题聚类或 metadata 字段（如 `labels` 中的 category 子字段）生成。
- D3：✅ 活跃 labeler 通常标数百条消息，分布足够。
- D4：⚠️ **标签非单一 1-5 整数**——是多维 [0,1] 连续值（quality 是主指标）。简单离散化 quality 为 5 档可接入现有 pipeline（这只是字段层面改造），但 **"为什么给 4 而不是 5" 的对比式边界 prompt 可能需要重写**，因为 labeler 看到的维度比 RecLLMSim 的整数满意度复杂得多。
- D5：✅ 天然满足（树结构 linearize）。
- H（跨任务迁移）：⚠️ **风险点**。labeler 在不同话题下的严宽可能不一致（比如对代码严、对闲聊宽）——需要先做一次诊断：训练集任务类型上 per-labeler 均值与测试集上的相关性是否显著 >0。

**结论**：**OASST 可以走，但任务定位发生变化**——不再是"预测用户对 AI 回复的满意度"，而是"预测标注者对某条消息的质量评分"。前者是 RecLLM 语境下的 end-user satisfaction，后者是**标注者偏好建模 / labeler bias modeling**。两者的学术价值和故事主线不同，需要先定 scope。

**改造量（走映射 B 时）**：

- 重写 `personalized_data.py` 的数据加载层（替换 `TASK_LIST` 为自定义 topic 分类器 + 自定义任务切分函数）；
- 新增一个 OASST 预处理脚本：聚合 labeler × topic × message，输出与 `SessionData` 等效结构——注意 "session" 概念在 OASST 并不天然成立，一位 labeler 对 N 条消息的打分序列可能要按某种时间 / tree 聚合；
- 评分 prompt 要重写（目标是 quality 评分而非满意度）。

---

## 5. WildBench

AllenAI 发布的 LLM benchmark，约 1024 条 challenging real-world prompts（从 WildChat 筛选 + 人工净化），用来对比 LLM 的回复质量。每条样本包含：任务描述 + 参考答案 + 多个模型的回复 + 成对偏好（win/tie/lose） + AI judge 分数。

**依赖对照**：

- D1（用户身份）：❌ 无持久用户——prompts 是从 WildChat 摘来的"典型困难样本"，评分者是模型或独立人工，无个性化可言。
- D2（跨任务 session）：❌ 每条 prompt 独立。
- D3（每用户多次打分）：❌。
- D4（满意度标签）：❌ **不是逐轮 1-5 满意度**，而是 pairwise preference + judge score。
- D5（对话结构）：❌ 多为单轮 prompt + response（少量多轮）。

**结论**：**完全不适合走 memory-agent 个性化流程**。WildBench 的目标是 **model comparison benchmark**，定位天然排除了 user-personalization。

**可用之处**（侧写）：可以把 WildBench prompts 作为**外部 OOD 评估集**，跑训练好的满意度预测器看它们在 WildBench prompts 上的预测分布和 AI-judge 分数的相关性——作为泛化能力检查，但这是另一条评估线，不是 memory-agent 流水线能承载的。

---

## 6. WildChat

AllenAI 发布的 1M+ 真实 ChatGPT 对话日志（来自同意授权的公共接口）。每条对话包含：
- 多轮 user-assistant 交互；
- 元数据：`hashed_ip`、country、model 版本、timestamp；
- **不含任何用户主动打分**；
- 有模型侧 redacted / moderation flags，但不等于满意度。

**依赖对照**：

- D1：✅ `hashed_ip` 可做近似 user id（同 IP 多次访问会重复出现）；WildChat 数据量大，能筛出活跃用户。
- D2：⚠️ 需自行给对话分类为不同任务（coding / writing / QA / advice / ...），本身有成本但可做。
- D3：⚠️ 分布式好，部分 heavy user 有数十至数百对话；但缺乏打分样本。
- D4（满意度标签）：❌ **核心阻塞**——WildChat 没有显式用户满意度标签，只有 implicit 信号（conversation 长度、用户是否在下一条 prompt 重说 / 抱怨、"regenerate" 在这份数据里不可见因为是 API log）。
- D5：✅ 满足。

**结论**：WildChat 是一个数据规模极大的好资产，但**关键信号缺失**使得它不能直接接到当前评估流程——**既没有 gold 分数来训，也没法算 MAE / Pearson 等评估指标**。

**可行路径**（代价不小，属于扩展方向而非直接复用）：

- **W1. LLM-judge 代标**：用 GPT-4/Claude 对每轮 assistant 回复打 1-5 分（rubric 可沿用 RecLLMSim 或重新设计）。风险：训练和评估同时用 LLM-judge 会形成 label noise + 循环偏置。
- **W2. 行为代理弱标签**：从 `hashed_ip` + timestamp 维度挖掘连续信号（下轮是否重述 / 对话是否在该轮后终止 / 用户是否在次轮肯定回复）。代价高、噪声大，但真实世界。
- **W3. 小规模人工标注**：从 WildChat 采样 N 条对话交付众包逐轮打分。可与 RecLLMSim 拼接形成 cross-domain benchmark，但离线成本与合规都要过。

**若只做 memory building（不评估）**：training-free 的 memory agent 可以在 WildChat 上"活起来"（给每个 heavy user 构 memory、生成 prompt），但没有 label 就无法量化收益。

**定位**：WildChat 是**未来**的扩展数据集，应放在 roadmap 而非立即复用。

---

## 7. 汇总判断

| 数据集 | D1 用户 | D2 多任务 | D3 分布 | D4 1-5 | D5 对话 | 直接复用？ | 降级路线 |
|---|---|---|---|---|---|---|---|
| **URS（本地）**   | ✅（181 user，zh/en 独立命名空间） | ✅（145/181 ≥2 intents） | ✅（114/181 ≥5 sessions） | ⚠️ session-level | ✅ | **接近直接复用** | **U1：session 粒度化 + uid 语言前缀**（小改造，~1 天） |
| **USS（本地）**   | ❌ | ❌ | ❌ | ✅ | ✅ | 否 | **R1 对话内 warm-up + R2 subset rubric**（中等改造，2-3 天） |
| **OpenAssistant** | ⚠️（labeler） | ⚠️（自造） | ✅（labeler） | ⚠️ | ✅ | 否（需重定位） | **映射 B：labeler-bias modeling**（任务重定位，中大改造） |
| **WildBench**     | ❌ | ❌ | ❌ | ❌ | ⚠️ | 否 | 只可做 OOD 评估参考 |
| **WildChat**      | ✅ | ⚠️ | ⚠️ | ❌ | ✅ | 否 | **须先补 label**（LLM-judge / 弱信号 / 众包，代价最高） |

**优先级建议（更新后）**：

1. **首选 URS（U1）**——唯一一个在 5 条硬依赖上几乎完全匹配的外部数据集，仅 D4 粒度差异可通过 session 级评估吸收。能直接验证 memory v2 + CDF calibration 在**跨领域任务分布（通用助手场景）+ 跨语言 + 多 LLM 来源**下的迁移能力，且可直接对比 RecLLMSim 上的 H 假设是否持续成立。
2. **其次做 USS（R1+R2）**——虽然无 user id，但已落地、可用作 training-free 对照线与现有 LoRA 监督 baseline 的消融。
3. **OpenAssistant**——研究方向是否愿意从 "end-user satisfaction" 扩到 "labeler-bias modeling" 是先决条件；若扩，数据量最大、故事线独立。
4. **WildBench** 只作 OOD 泛化检查。
5. **WildChat** 放长期 roadmap（缺 gold label）。

## 8. 落地改造路径

### 8.1 URS（首选，U1 路线）

最小可行改动清单：

1. **数据读取**：**不要**直接用 `all_conversations.json`——它对 17 个 id 冲突未做去冲突处理。正确做法是分别读 `chinese_merged.json` 和 `english_processed.json`，对 `user_id` 加语言前缀（`zh_{uid}`, `en_{uid}`）后合并。
2. `detection/lib/urs_data.py` — 新建：
   - 定义 canonical intent 映射（`解决专业问题 ↔ Solve Professional Problem` 等 7 类）。
   - 定义 satisfaction 映射（`很不满意 → 1 … 非常满意 → 5`，en 同构）。
   - `load_urs_sessions(languages=('zh','en')) -> dict[user, dict[intent, list[SessionData]]]`：读入时即加语言前缀；每条 session 的 `satisfaction_scores` 只装一个元素（session-level score），`dissatisfaction_reasons` 填 `"满意" / "其它"`。
   - `build_urs_personalized_samples(split, train_ratio, seed, min_history_sessions, languages=...)`：复用 `personalized_data.py` 的 cross-task split 模板，`TASK_LIST` 替换为 7 个 canonical intent。`languages` 参数天然支持"zh-only / en-only / both"三种 run 配置与跨语言迁移实验。
3. `detection/trace/collect_personalized.py` — 几乎不改：
   - 把数据入口抽象成 `sample_builder`（callable），对 URS 路径指向 `build_urs_personalized_samples`。
   - memory cache 命名沿用 `{user}__{intent}__{model}`，由于 user 已带 `zh_` / `en_` 前缀，缓存 key 不会冲突。
4. `detection/lib/memory.py` — 无须动核心 schema；URS 分支在 prompt 模板里把"用户画像"段落略过，"任务描述"改为从 `title + intent_label` 拼接。多语言 prompt：如果 `zh_` 前缀用户，用中文 rubric；`en_` 前缀用户，用英文 rubric。
5. `detection/eval/calibrate.py` — **无需改动**；calibration 已按 block 聚合，block 粒度天然是 session-level。
6. 评估：`eval/personalized.py` 支持 session-level 指标（MAE / RMSE / Pearson / Spearman / QWK / per-user bias），当前逻辑已 block-aware，粒度切换只影响 turn-level 指标的含义（退化为 session-level）。

预计改造量：**~1 天**工作可跑通完整 training-free 基线 + CDF 后校准对比。

### 8.2 USS（R1+R2 路线）

1. `detection/lib/uss_data.py` — 新增：
   - `build_uss_dialogue_samples()`：把每条 dialogue 拆成 `(warmup_prefix, target_suffix)`，前 n=5 轮标签可见构成 `history_sessions` 等价对象。
   - `build_uss_subset_samples()`：按 subset 聚合为 population-level 样本，配合 subset-rubric memory。
2. `detection/trace/collect_personalized.py` — 改动：
   - memory cache 命名抽象为 `cache_key_fn`（callable），USS 分支替换为 `{dialogue_id}__{subset}` 或 `{subset}__population`。
3. `detection/lib/memory.py` — 无须动核心 schema，只需为 USS 子集重写 prompt 模板里的"任务描述" / "用户画像"片段（profile 置空时已有 fallback）。
4. `detection/eval/calibrate.py` — 支持从输入 JSONL 的**同 dialogue 前缀 gold** 算 CDF，而不是从 memory cache 读。新增一个 `--calibration_source dialogue_prefix|memory_cache` 参数。
5. 评估沿用 `eval/personalized.py` + `eval/diagnose_confusion.py`，无需改动。

预计改造量：2-3 天工作，能跑通完整 training-free 基线 + CDF 后校准对比。

## 9. 尚未验证的关键假设（做之前建议先小样本 pilot）

- **URS U1**：session-level 粒度下的 memory v2 是否仍保留 RecLLMSim 上观测到的"对比式评分边界"信号——建议先在 URS 上跑 30-50 个 block 的 pilot，对比 `no_memory` vs `memory_v2` 的 QWK 与 CDF 前后的 |bias|，确认 CDF 校准增益（在 RecLLMSim 上 −10% MAE / +20% Pearson）是否可复现到 session 粒度。
- **URS H 假设**：RecLLMSim 的 4 类生活任务 H 成立，URS 是更多样的通用助手任务（专业/检索/闲聊），评分风格是否仍跨 intent 迁移——**先做 pandas 快速统计**：每用户在 training intents 上的 mean score vs test intent 上的 mean score 的 Pearson（跨 145 个有效用户），>0.3 即可认为迁移有效。对 zh（87 user）和 en（58 user）分别算，看语言是否影响稳定性。
- **URS 语言异构**：加语言前缀后 zh/en 完全不相交，memory v2 在单语言子集（比如只跑 zh 87 user）vs 混合（145 user）下结果是否一致——若差异显著，说明 prompt 模板里混语言材料互相干扰，需要分语言训练/推理。
- **USS R1**：subset 内"前 n 轮 gold 能代表整段 dialogue 的评分风格"——若 dialogue 本身只有 ~9 轮且分数波动剧烈，warm-up 的估计方差会大。建议先在 JDDC 上取 100 条 pilot 看 warm-up 前 5 轮 vs 全程的 mean / std 相关性。
- **USS R2**：subset 级别的 rubric 是否比 zero-shot prompt 更准——需和"no_memory"直接对比。
- **OASST 映射 B**：labeler 在不同 topic 下打分是否稳定（H 是否成立）——无需写复杂 pipeline，先用小 pandas 统计看一眼。

这些 pilot 实验代价都很低，比直接上大规模 run 风险小得多。
