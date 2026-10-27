# 回忆检索

回忆实现位于 `src/rems/recall/pipeline.py`，主端口转发位于 `src/rems/pipeline.py`。下文区分了硬过滤、候选检索和排序，避免把“可及性”误写成向量相似度。

## 输入及硬过滤

回忆接收主体 ID、查询文本，以及可选对象 ID、对话对象 ID、时间范围、字符预算、结果数量、是否展开原文和检索通道。

分阶段模式中，近期轻量小事件没有摘要，默认按完整原文取回。
已经被上层吸收的节点退出普通候选与索引，上层提炼事件按原有预算选择表示。
上层事件的语义分取它自己的向量，以及名下第一层原文向量中的最高分。原文向量的材料类型是 `event_detail`，不单独成为候选。
词面分对摘要和每一条原文分别计算，取最高分。
相似度达到门槛的第一层原文才附在同一条结果里；放不下则整条不附，预算截取也不从附加原文中间切断。
被吸收的节点仍不能单独成为候选。提炼事件的时间过滤按原始来源区间重叠判断，不按提炼时刻判断。
详见[分阶段提炼](multistage-memory-refinement.md)。

1. 空查询直接返回空结果。
2. 事件候选限定到 `subject_id`，并排除墓碑事件。
3. 如果传入时间范围，事件按 `occurred_at` 过滤；缺少发生时间时使用 `create_time`。边界包含在范围内。
4. 如果传入对象，候选必须与所请求对象关联。对象和时间在向量搜索前过滤，不是向量分数中的加权项。

注意：近期分池按 `create_time` 排序；时间范围过滤优先使用 `occurred_at`。因此“近期”与“发生在某段时间”使用的时间字段并不完全相同。

## 召回信号与候选准入

- 默认 `recall_event_index_level="L0"`：事件原文生成向量，长文本按实际编码器 token 上限分块并保留相邻重叠；同一事件多个块命中取最高语义分，不重复交付事件。该模式不生成压缩对象事实向量，压缩摘要和对象事实仍供预算选级与词面检索使用。`L1` 保留旧事件/对象事实向量路径作为对照。L0 使用独立 `_l0` 集合；切换后需执行 `backfill_index` 重建已有事件，新写入事件自动索引。原文/版本/对象载荷未变时跳过重复嵌入，过期分块会删除。
- 向量索引来源和取回表示独立：改用 L0 索引不会自动把原文交给核心，也不改变原来的压缩层预算选择。当前对照固定词面规则、排序及核心组装，以检查替换向量来源的作用。
- 词面通道对可用摘要/对象事实文本计算匹配分：完整查询命中为 1，其余结合最长公共片段及查询双字片段的覆盖率。单独出现人名或日期不会让整句查询得满分。
- 纯英文可启用 `recall_english_word_lexical=true`：按单词边界计算去功能词后的查询覆盖，处理所有格和简单复数；避免长英文摘要靠常见双字母片段取得虚高分。默认关闭以保留兼容行为，含中文的查询仍走原规则。
- 语义分数和词面分数分别与各自阈值比较；任一启用通道过线即可准入（未完成事件候选仅使用词面条件）。
- 一个候选的原始相关度是 `min(1, max(semantic, lexical) + 0.25 × min(semantic, lexical))`。词面与语义同时匹配时得到小幅加分；可及性不乘入向量分数。
- 指定对象时，历史对话对象候选使用 0.7 权重，其他涉及对象使用 0.3 权重；未指定对象时对象权重为 1.0。权重作用于排序分。

## 候选级联、可及性与通道

事件数不超过 `recall_cascade_min_events`（默认 200）时走全量扫描：全部符合主体、对象和时间过滤的事件都参加语义与词面检索，排序严格按相关度；可及性只在同分时打破平局。

超过该阈值且 `recall_candidate_mode="cascade"`（默认）时，召回先由四路生成有界候选，再只加载并精排这些事件：

1. 语义路：向量 `top_k=recall_candidate_vector_k`（默认 200）。
2. 词面路：SQLite FTS5 trigram `top_k=recall_candidate_lexical_k`（默认 200）；2 字查询走受限 `LIKE`。
3. 近期路：最新 `recall_candidate_recent_k`（默认 80）条。
4. 对象路：`role_list`、`refinement_objects`、历史对话对象关系，每个对象最多 `recall_candidate_route_k`（默认 100）条；旧 `object_timeline` 路由也并入。

四路按 event_id 去重后取 `recall_candidate_limit`（默认 500）。随后仍然用精确词面分、语义分、对象权重和可及性排序；候选路只负责“找”，不改变最终打分口径。语义路只覆盖自己的 top-K；词面/近期候选不在其中时，会按 `recall_candidate_semantic_backfill`（默认 100）补一次受限语义分，避免 `max(semantic, lexical)` 系统性压低 FTS 命中。

候选生成少于 `recall_candidate_min_events`（默认 8）时回退全量扫描；`recall_candidate_mode="full"` 可整体关闭级联。级联路径不再使用原来的“近期保底 + 长期概率采样”选池；可及性仍参与候选排序，但不再决定哪些老事件能被语义检索到。

FTS5 是派生索引：`events` 仍是唯一事实源，触发器在 insert/update/delete 时同步 `content_raw` 与摘要值；首次在已有库启动时 backfill 一次。词面路按 trigram 的 OR 查询匹配（长句不要求整句连续），短于 3 字走受限 `LIKE`；FTS 不可用时自动退回全量扫描。

### 全量模式下的长期概率采样

`recall_candidate_mode="full"` 且事件数超过 `recall_max_capacity`（默认 1000）时，仍使用“近期保底 + 长期概率采样”：

1. 对完成主体、对象、时间过滤的事件，按 `create_time` 从新到旧排序。
2. 最新的 80 条进入近期语义池，近期语义搜索的 `top_k` 默认是 40。
3. 更早的事件逐条按可及性概率抽样后，才进入长期语义搜索。
4. 如果近期候选通过相关度准入，且其排序分达到最高候选的 90%，可将其前置。

长期事件的当前概率计算为：

```text
age_decay = 0.5 ** (age_days / recall_access_half_life_days)
effective_access = event_fatigue * forgetting_factor * age_decay
p = floor + (1 - floor) * effective_access / (1 + effective_access)
```

默认半衰期为 60 天，长期概率下限为 0.05。`event_fatigue` 是配置值；`forgetting_factor` 在事件封装时由情绪唤醒度初始化，并可在回忆命中后增长。该路径只控制语义搜索池；词面通道仍会遍历已通过硬过滤的事件。

## 排序与重复强化

先按“相关度 × 对象角色权重”统一排列事件和对象事实；同分时对象事实优先。大库中，只有排序分落在 `recall_relevance_close` 配置的接近范围内，才用可及性打破顺序。小库中，只有严格同分时才用可及性。

只有进入最终回忆片段的已封存事件才会被强化；未封存片段不强化。同一事件在一次回忆中最多强化一次。默认 `forgetting_factor` 乘数为 1.15，上限为 3.0。强化保存后影响后续长期语义池概率。

## 预算与返回

`RecallBudgetManager` 以及回忆选择逻辑按字符预算选择一个完整表示级别。材料放不下时不会截断后塞回；可请求展开原文，但仍需符合预算。`limit` 限制返回片段数量。上层事件附带的原文占同一条的字符预算，不增加条数。默认最多附 1 条，见 `recall_detail.max_count`。相似度门槛默认取 `recall_semantic_min`，可用 `recall_detail.min_similarity` 覆盖。日期锚点由 `recall_detail.date_anchor_locale` 决定，默认中文，同一条结果不混用另一种写法。预算若会切进附加原文，就去掉整段原文，并清空 `detail_event_ids`。

回忆片段保留事件 ID、文本/预算选中的完整内容、类型、对象 ID、历史对话对象归属、来源 ID、分数、摘要级别、检索信号和时间。`interlocutor` 只有在历史事件唯一明确的对话对象时才有值；多对象情况看 `interlocutor_object_ids` 和 `interlocutor_attributions`。

预算选择可选的诊断出口：`recall(..., candidate_sink=...)`（或实例属性 `recall_candidate_sink`）会按最终顺序记录每条被访问候选的排名、材料类型、表示层级、字符数和淘汰原因（`selected` / `limit` / `duplicate` / `no_room`）。上层事件的原文另记 `detail_attached` / `detail_below_threshold` / `detail_no_room` / `detail_limit`，并写下相似度。它只读，不参与排序、不占预算、不改变返回结果；默认关闭。`limit` 截断时只会记下第一条被挡住的候选，要拿完整排名就不要传 `limit`。一次回放记下的相似度可以离线比较不同门槛，不必为每个门槛重跑。

## 已知接口限制与待验证点

- 当前 `RecallPipeline.recall` 接受 `level` 和 `anchor_event_ids`，但实现中会忽略这两个参数；表示级别由预算选择，细节条数由 `recall_detail.max_count` 固定，不会随档位增减，也不会按锚点扩展检索。
- JShi 适配层传来的 `interlocutor_object_id` 优先；未显式给出时主端口按 `object_ids` 的第一项推定。调用方应保持“对话人优先”的顺序，或显式传入。
- 当前实现的概率曲线、情绪映射和强化系数都应通过可重复的离线分布/场景验证后再调整；对应试验记录在[回忆可及性草稿](drafts/recall-accessibility.md)。
