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

- **H. 用户评分风格跨任务迁移** — CDF 校准与 v2 对比边界都预设用户的评分基准（严 / 宽 / 中）不随任务剧变。本项目的 `reports/calibration_results.md` 在 4 类生活任务上验证了它基本成立。

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

## 3. OpenAssistant (OASST / OASST2)

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

## 4. WildBench

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

## 5. WildChat

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

## 6. 汇总判断

| 数据集 | D1 用户 | D2 多任务 | D3 分布 | D4 1-5 | D5 对话 | 直接复用？ | 降级路线 |
|---|---|---|---|---|---|---|---|
| **USS（本地）**   | ❌ | ❌ | ❌ | ✅ | ✅ | 否 | **R1 对话内 warm-up + R2 subset rubric**（中等改造） |
| **OpenAssistant** | ⚠️（labeler） | ⚠️（自造） | ✅（labeler） | ⚠️ | ✅ | 否（需重定位） | **映射 B：labeler-bias modeling**（任务重定位，中大改造） |
| **WildBench**     | ❌ | ❌ | ❌ | ❌ | ⚠️ | 否 | 只可做 OOD 评估参考 |
| **WildChat**      | ✅ | ⚠️ | ⚠️ | ❌ | ✅ | 否 | **须先补 label**（LLM-judge / 弱信号 / 众包，代价最高） |

**优先级建议**：

1. **先做 USS（R1+R2）**——数据已落地、改动最集中、能验证 memory v2 rubric + CDF calibration 在**跨语言 / 跨领域 / 无用户画像**条件下的鲁棒性。这同时给现有 USS 监督预测器（`predictor/lora_ordinal.py`）一个 training-free 对照。
2. **其次再评估 OpenAssistant**——可转型为"标注者偏好建模"研究，故事线完整且数据丰富，但需要先与研究目标对齐（是否愿意把 scope 从 end-user-satisfaction 扩到 labeler-bias）。
3. **WildBench 只作 OOD 泛化检查**（如果主线预测器需要 external validation）。
4. **WildChat 放长期 roadmap**——若项目后续要讲"真实世界部署"故事，再启动 LLM-judge 代标或众包路线。

## 7. 落地改造路径（仅针对推荐的 USS 路线）

如果决定先做 USS，最小可行改动清单：

1. `detection/lib/uss_data.py` — 新增：
   - `build_uss_dialogue_samples()`：把每条 dialogue 拆成 `(warmup_prefix, target_suffix)`，前 n=5 轮标签可见构成 `history_sessions` 等价对象。
   - `build_uss_subset_samples()`：按 subset 聚合为 population-level 样本，配合 subset-rubric memory。
2. `detection/trace/collect_personalized.py` — 改动：
   - memory cache 命名抽象为 `cache_key_fn`（callable），默认沿用 `{user}__{task}`，USS 分支替换为 `{dialogue_id}__{subset}` 或 `{subset}__population`。
3. `detection/lib/memory.py` — 无须动核心 schema，只需为 USS 子集重写 prompt 模板里的"任务描述" / "用户画像"片段（profile 置空时已有 fallback）。
4. `detection/eval/calibrate.py` — 支持从输入 JSONL 的**同 dialogue 前缀 gold** 算 CDF，而不是从 memory cache 读。新增一个 `--calibration_source dialogue_prefix|memory_cache` 参数。
5. 评估沿用 `eval/personalized.py` + `eval/diagnose_confusion.py`，无需改动。

预计改造量：2-3 天工作，能跑通完整 training-free 基线 + CDF 后校准对比。

## 8. 尚未验证的关键假设（做之前建议先小样本 pilot）

- **USS R1**：subset 内"前 n 轮 gold 能代表整段 dialogue 的评分风格"——若 dialogue 本身只有 ~9 轮且分数波动剧烈，warm-up 的估计方差会大。建议先在 JDDC 上取 100 条 pilot 看 warm-up 前 5 轮 vs 全程的 mean / std 相关性。
- **USS R2**：subset 级别的 rubric 是否比 zero-shot prompt 更准——需和"no_memory"直接对比。
- **OASST 映射 B**：labeler 在不同 topic 下打分是否稳定（H 是否成立）——无需写复杂 pipeline，先用小 pandas 统计看一眼。

这些 pilot 实验代价都很低，比直接上大规模 run 风险小得多。
