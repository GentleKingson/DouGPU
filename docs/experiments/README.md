# 实验报告

2026-10-09，[协议强化离线审计](protocol-hardening-20261009.zh-CN.md)完成：三段真实历史已还原，准入 **FAIL / PROTOCOL_NOT_READY，Training PAUSED**。两地复算一致，执行证据缺口继续阻断；0 GPU、生产算法未变。

2026-10-08，[NTP 消融](ntp-ablation-20261008.zh-CN.md)复核结案为 **CLOSED / INVALID_PROTOCOL**，**Training：PAUSED**。下一阶段仅为 **PROTOCOL HARDENING**，不自动重训。此前因遗漏原 Baseline 的 8-update 恢复边界而停在 500 更新；GPU 共 90.174 秒，完整中止状态已双域恢复核验，未运行棋力 A/B，不形成 NTP 效果结论。

2026-10-08，新 Baseline 的 2,000→20,000 更新独立强对手评估已完成，主比较为 **POSITIVE_SIGNAL**；20,000 latest 优于 best。完整输入、逐副结果与复核脚本已归档并读回校验。该单训练 seed 结果不恢复历史权重、不触发模型晋升或续训，见[本轮报告](baseline-strength-20261008.zh-CN.md)。

截至 2026-10-04，effective batch512 三 seed 全部 FAIL，跨 seed 为 REJECT；不进入 final holdout，不追加样本或第四 seed，默认 batch256 不变。既有日志确认 4h→6h 阶段 sample-update ratio（replay sampling intensity）约为 baseline 的 2×，不是按更新次数定义的 UTD 翻倍；frozen-replay 梯度与 Adam 单步探针已完成，不能把本轮解释为纯梯度 batch 对照。详见 [batch512 拒绝与机制诊断](batch512-20261004.zh-CN.md)。

截至 2026-10-03，6h→8h 三 seed 为 1 PASS、2 INCONCLUSIVE。按预注册规则保留 6h，停止预算扩张，不进入 12h，不追加第四 seed 或补抽留出。历史报告中的下一步建议不代表当前执行计划。

## 发布清单

| 阶段 | 报告 | 当轮结果 |
|---|---|---|
| NTP 辅助目标消融 | [协议偏差与结案记录](ntp-ablation-20261008.zh-CN.md) | CLOSED / INVALID_PROTOCOL，Training PAUSED，无棋力比较 |
| 角色学习差异 | [端点机制分析](role-mechanism-20261008.zh-CN.md) | COMPLETE / INCONCLUSIVE / STOP，未识别持续机制 |
| 独立 Checkpoint Selection | [20k latest 对 best](checkpoint-selection-20261008.zh-CN.md) | NO_DECISION / STOP，地主非劣证据不足 |
| 新 Baseline 2,000→20,000 更新 | [独立 WP/ADP 评估](baseline-strength-20261008.zh-CN.md) | COMPLETE / POSITIVE_SIGNAL，未晋升模型 |
| 4h→6h | [seed42](six-hours-seed42-20261002.zh-CN.md) | 验证 3/5，留出 5/5，整体扩展门槛未过 |
| 4h→6h | [seed43](six-hours-seed43-20261002.zh-CN.md) | PASS 5/5 |
| 4h→6h | [seed44](six-hours-seed44-20261002.zh-CN.md) | PASS 5/5 |
| 6h→8h | [seed44](eight-hours-seed44-20261002.zh-CN.md) | PASS 5/5 |
| 6h→8h | [seed43](eight-hours-seed43-20261003.zh-CN.md) | INCONCLUSIVE 4/5 |
| 6h→8h | [seed45 与最终汇总](eight-hours-seed45-20261003.zh-CN.md) | INCONCLUSIVE 4/5，停止扩张 |
| batch256→512 | [三 seed 与机制诊断](batch512-20261004.zh-CN.md) | 全部 FAIL，REJECT effective batch512 |

本目录发布以上报告与本索引；原始文件不随 Git 发布。新 Baseline 报告提供已验证的独立归档入口；旧实验报告中的哈希仅作历史身份记录。证据获取与历史口径见[证据说明](../experiment-evidence.zh-CN.md)。
