# 实验报告

2026-10-09，[新 Baseline B1 三 seed 训练结案](baseline-replication-b1-20261009.zh-CN.md)：**B1_COMPLETE / EXECUTION_QUALIFIED / TRAINING_PAUSED**。获批 LocalServer B1 最多 10,800 秒，三条 0→8→2k→20k 链及两地封存读回全部 PASS；含最终读回 47.45 分钟，60,000 次成功更新。B2/G 未进入，尚无棋力结论。

2026-10-09，[B1_READINESS_AUDIT 零 GPU 结案](b1-readiness-audit-20261009.zh-CN.md)：**PASS_WITH_BLOCKERS / TRAINING_PAUSED**。P1–P5 按序验收，22 项回归通过、两地封存读回一致；设备/驱动/镜像 GPU 能力 NOT_VERIFIED，可申请受限 preflight 审批，完整训练仍阻断，B1/B2 未进入。

2026-10-09，[新 Baseline 方向性复现 B0 协议草案](baseline-replication-protocol-20261009.zh-CN.md)已审阅修订：**PROTOCOL_DRAFT / TRAINING_PAUSED**。原始配方与固定 WP/ADP 字节已核对，预定三条 0→8→2k→20k 链；B1/B2 未进入，获准 GPU 预算为 0，三 seed 筛查不提供总体显著性结论。

2026-10-09，[冻结 Replay 跨端点研究 R0–R2](cross-replay-20261009.zh-CN.md)结案：**R0_PASS；R1 数值执行 PASS；R2=RESEARCH_SCREEN_INCONCLUSIVE / STOP**。仅探索性数值观察，G_NOT_ENTERED / TRAINING_PAUSED。

2026-10-09，[执行证据闭环](execution-evidence-20261009.zh-CN.md)完成：**INFRASTRUCTURE_READY / TRAINING_PAUSED**。恢复 SHA256、Actor seeds 和启动源码身份已记录，两地 CPU 离线复算一致；历史 Baseline 仍 FAIL，NTP 仍 CLOSED / INVALID_PROTOCOL。

2026-10-09，[协议强化离线审计](protocol-hardening-20261009.zh-CN.md)完成：三段真实历史已还原，准入 **FAIL / PROTOCOL_NOT_READY，Training PAUSED**。两地复算一致，执行证据缺口继续阻断；0 GPU、生产算法未变。

2026-10-08，[NTP 消融](ntp-ablation-20261008.zh-CN.md)复核结案为 **CLOSED / INVALID_PROTOCOL**，**Training：PAUSED**。下一阶段仅为 **PROTOCOL HARDENING**，不自动重训。此前因遗漏原 Baseline 的 8-update 恢复边界而停在 500 更新；GPU 共 90.174 秒，完整中止状态已双域恢复核验，未运行棋力 A/B，不形成 NTP 效果结论。

2026-10-08，新 Baseline 的 2,000→20,000 更新独立强对手评估已完成，主比较为 **POSITIVE_SIGNAL**；20,000 latest 优于 best。完整输入、逐副结果与复核脚本已归档并读回校验。该单训练 seed 结果不恢复历史权重、不触发模型晋升或续训，见[本轮报告](baseline-strength-20261008.zh-CN.md)。

截至 2026-10-04，effective batch512 三 seed 全部 FAIL，跨 seed 为 REJECT；不进入 final holdout，不追加样本或第四 seed，默认 batch256 不变。既有日志确认 4h→6h 阶段 sample-update ratio（replay sampling intensity）约为 baseline 的 2×，不是按更新次数定义的 UTD 翻倍；frozen-replay 梯度与 Adam 单步探针已完成，不能把本轮解释为纯梯度 batch 对照。详见 [batch512 拒绝与机制诊断](batch512-20261004.zh-CN.md)。

截至 2026-10-03，6h→8h 三 seed 为 1 PASS、2 INCONCLUSIVE。按预注册规则保留 6h，停止预算扩张，不进入 12h，不追加第四 seed 或补抽留出。历史报告中的下一步建议不代表当前执行计划。

## 发布清单

| 阶段 | 报告 | 当轮结果 |
|---|---|---|
| 冻结 Replay 跨端点 R0–R2 | [资格、数值比较与结案](cross-replay-20261009.zh-CN.md) | R0_PASS；R1 PASS；R2 INCONCLUSIVE / STOP，G_NOT_ENTERED |
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
