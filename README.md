# JShi_memory

JShi_memory 是匠石（JShi）使用的记忆后端项目。Python 包名为 `rems`。当前代码提供记忆写入、回忆、对象肖像查询和 HTTP API；记忆事件、对象关联、检索索引和回忆轨迹由本项目管理。

本目录是 `JShi_memory_old_days` 独立副本，基于 `feat/recall-scaling` 的 `ec776ca`。
默认恢复完整长事件：正常封存范围为 2000 ≤ 字符数 < 3000，封存时生成摘要、对象事实和情绪。
近期小事件及 90/365 天提炼默认关闭；保留后续检索、来源、JSON 输出、人物经验及简体规范化改进。
分支历史、取舍和使用方法见[长事件恢复说明](docs/long-events-restoration.md)。

默认同时启用独立 Mem0 OSS 近期库：两库并行写入、并行检索，最后按共同预算合并去重。
近期原文保留 365 天，10 GB 预算，8 GB 触发清理到 7 GB；长期以 L1 索引为主，保留 L0 回查。
接入边界、配置及真实内容测试程序见[双库运行说明](docs/dual-memory.md)。

## 文档入口

- [长事件恢复说明](docs/long-events-restoration.md)：本副本的默认行为、Git 审查、保留范围和 JShi 接口。
- [双库运行说明](docs/dual-memory.md)：Mem0、并行读写、清理策略和真实内容评测。
- [项目总览](docs/project-overview.md)：运行组成、边界和主要数据流。
- [文档索引](docs/README.md)：当前文档规则、模块地图和历史文档入口。
- [记忆写入](docs/memory.md)：输入模型、代谢封存、事件和存储。
- [回忆检索](docs/recall.md)：过滤、语义/词面检索、近因与长期通道、排序和预算。
- [匠石交互](docs/jshi-integration.md)：MemoryBackendPort、字段映射和 HTTP 接口。
- [开发与验证](docs/development.md)：环境、配置和测试入口。
- [LoCoMo 匠石评测](docs/locomo-evaluation.md)：真实核心、英文记忆、答案条目覆盖及费用控制。
- [草稿箱](docs/drafts/README.md)：模块方案、实施步骤和调试记录的工作区。
- [历史文档](docs/history/README.md)：本次迁移前的文档原件，供追溯，不作为当前行为依据。

## 快速认识

写入入口是 `MemoryBackendPort.ingest_batch`，主要实现位于 `src/rems/pipeline.py`；回忆入口是 `MemoryBackendPort.recall`，检索实现位于 `src/rems/recall/pipeline.py`。FastAPI 路由位于 `src/rems/api/`。

代码和当前契约是行为依据。接口声明、实现、配置和文档之间若有差异，应先记录差异，再以正在运行的实现确认实际行为；不要仅凭历史设计文档推断已上线功能。
