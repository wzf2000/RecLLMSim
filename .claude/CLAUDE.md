# RecLLMSim Project Guidelines

## Shell Scripts

- **不得在 `.sh` 脚本中硬编码任何本地 Python 环境路径**（如 `/path/to/conda/envs/<env>/bin/python`）。
  脚本中统一使用 `python`，由用户在运行时自行激活所需 conda 环境。

## 实验记录

- **所有实验结果、设计分析、诊断报告均须记录在 `detection/reports/` 目录下的 Markdown 文件中。**
  每次 session 结束前，检查本次新增的实验结果或重要分析是否已落地为报告文件；若有遗漏，主动补充。
- 文件命名建议：`{主题}_{内容类型}.md`，例如 `memory_v2_design.md`、`personalized_satisfaction_results.md`。
- 报告内容应足够完整，使其他模型或研究者读后能理解已做了什么、怎么做的、得到了什么结论，无需查看对话历史。
