# 训练实验报告

截至 2026-10-04，effective batch512 三 seed 全部 FAIL，跨 seed 为 REJECT；不进入 final holdout，不追加样本或第四 seed，默认 batch256 不变。既有日志确认 4h→6h 阶段 sample-update ratio（replay sampling intensity）约为 baseline 的 2×，不是按更新次数定义的 UTD 翻倍；frozen-replay 梯度与 Adam 单步探针已完成，不能把本轮解释为纯梯度 batch 对照。详见 [batch512 拒绝与机制诊断](batch512-20261004.zh-CN.md)。

截至 2026-10-03，6h→8h 三 seed 为 1 PASS、2 INCONCLUSIVE。按预注册规则保留 6h，停止预算扩张，不进入 12h，不追加第四 seed 或补抽留出。历史报告中的下一步建议不代表当前执行计划。

## 发布清单

| 阶段 | 报告 | 当轮结果 |
|---|---|---|
| 4h→6h | [seed42](six-hours-seed42-20261002.zh-CN.md) | 验证 3/5，留出 5/5，整体扩展门槛未过 |
| 4h→6h | [seed43](six-hours-seed43-20261002.zh-CN.md) | PASS 5/5 |
| 4h→6h | [seed44](six-hours-seed44-20261002.zh-CN.md) | PASS 5/5 |
| 6h→8h | [seed44](eight-hours-seed44-20261002.zh-CN.md) | PASS 5/5 |
| 6h→8h | [seed43](eight-hours-seed43-20261003.zh-CN.md) | INCONCLUSIVE 4/5 |
| 6h→8h | [seed45 与最终汇总](eight-hours-seed45-20261003.zh-CN.md) | INCONCLUSIVE 4/5，停止扩张 |
| batch256→512 | [三 seed 与机制诊断](batch512-20261004.zh-CN.md) | 全部 FAIL，REJECT effective batch512 |

本目录只发布以上七份报告与本索引，不包含独立哈希清单、统计 JSON、协议文件、脚本、日志、checkpoint、权重或证据归档。报告内已有的产物哈希保留以便识别，不表示原始文件已发布。证据获取与历史口径见[证据说明](../experiment-evidence.zh-CN.md)。
