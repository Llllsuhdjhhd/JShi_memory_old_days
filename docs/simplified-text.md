# 简体中文工作文本

输入、模型生成的自然语言字段及持久化工作正文统一简体。人物、事件和来源ID、协议枚举、引用、字典键、网址、代码和工具参数保留。原始输入与工作字段转换前的内容存入 simplified_originals，模型输出因转换而改变时另存 simplified-model-originals.jsonl。原始证据仍可追溯。

utils/simplified_text.py 实现边界转换；storage/database.py 注册写入规范，避免导入或旁路写入漏过；recall/pipeline.py 只在查询入口转换一次，候选关键词匹配的 _norm 仅小写，不再调用 OpenCC。索引事件使用与数据库相同的简体正文。

已有JShi数据使用 jshi.app.simplified_migration 预览、备份并迁移。受影响事件待现有服务正常重启后，用唯一Qdrant客户端补齐当前集合索引；清单未成功处理完不会删除。更新不需要付费模型。

本规则替代 port.py 中“输入不归一化”的旧约定：不改写原始证据，内部工作副本仅统一字形，不改变来源、时间及对象归属。检索排序和预算不因简繁统一而改变。
