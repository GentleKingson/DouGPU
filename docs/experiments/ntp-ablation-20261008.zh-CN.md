# NTP 消融实验结案报告

日期：2026-10-08（Asia/Hong_Kong）。**NTP_ABLATION：CLOSED / INVALID_PROTOCOL；Training：PAUSED。** 下一阶段仅为 PROTOCOL HARDENING，不启动训练。

本轮因实验协议漏核历史会话边界，在 500 次成功更新时停止。NTP=0 的计算路径及 CPU/GPU 工程检查通过，但没有完成棋力 A/B，不能判断 NTP 的棋力收益或训练效率。生产默认 NTP=0.02 不变，500-update 状态仅保留为中止证据，不续训、不作为新实验起点。

## 原计划与协议偏差

计划复用既有 seed `20261009` 的 20,000-update NTP-on latest 作为 A，仅新增同 seed、从零初始化的 NTP-off 轨迹 B。正式配置只将 `ntp_weight` 从 0.02 改为 0，保持 DMC、Transformer、Belief=0.05、有效 batch256、优化器及采样配置不变。[归档配置](../../configs/experiments/ntp-off-20261008.json)来自实际 Baseline，不是当前默认配置；该文件是本轮历史记录，不是重试入口。

执行代理在 Gate 0 核验了端点配置和完整状态，却遗漏了训练日志中的会话边界：

| 对象 | 训练会话边界 | 中途恢复次数 |
|---|---|---:|
| A：历史 NTP-on | 0→8；8→2,000；2,000→20,000 | 2 |
| B：冻结计划 | 0→2,000；2,000→20,000 | 1 |
| B：实际执行 | 0→500，随后安全停止 | 0 |

A 的 8-update smoke 状态曾直接恢复为正式 Baseline。B 的两更新 smoke 则在独立目录运行，没有并入正式训练。`dougpu.train` 虽恢复 learner、Adam、replay 和主 RNG，但每次启动进程都会重新抽取 worker seeds、创建 ActorPool；actor 的在途牌局和内部 RNG 不恢复。额外一次重启可能改变后续自博弈数据，使比较混入 NTP 以外的执行差异。

这是实验准入核验遗漏，不是生产训练器错误，也不是 NTP 无效的证据。发现后立即发出 STOP，B 保存时 `Adam.step = successful updates = 500`，端点为 **INCOMPLETE / signal**。没有重开训练、选择替代轨迹或运行强对手评估。维护者复核认可停止并正式结案。

## 已完成的验证与预算

| 阶段 | 实际结果 | GPU 进程墙钟 |
|---|---|---:|
| Gate 0 | 两端完整状态与身份核验通过；会话历史漏核，准入不完整 | 0 |
| Gate 1 CPU | NTP-only 自检及 58 项现有测试通过 | 0 |
| Gate 1 GPU preflight | 通过 | 38.014 秒 |
| Gate 1 独立 smoke | 两周期、2 更新；NTP loss=0，Q/Belief 正常、梯度有限 | 10.004 秒 |
| Gate 2 正式 B | 500 更新安全停止，nonfinite=0 | 42.156 秒 |
| Gate 3 / Gate 4 | 未运行 | 0 |
| 合计 | 502 次新增更新，无棋力比较 | **90.174 秒（1.503 分钟）** |

GPU 进程墙钟包含预检、smoke 和训练进程，未达到获批的 3,600 秒上限；剩余预算不作为重试授权。从协议冻结到最终归档恢复校验约 **729.584 秒（12.160 分钟）**，包含 CPU 检查、传输、人工核验与归档，不是 GPU 忙碌时间。结束时 GPU 已空闲。

原 A 的 8-update smoke 属于正式轨迹，三段训练进程合计 **704.791 秒**；仅后两段的 694.550 秒不能代表完整 A 训练时间。A 全部 GPU 阶段为 750.309 秒，含恢复核验总墙钟为 1,223.419 秒。B 未到同一终点，且 A 为历史运行，**本轮不报告速度优势或单位 GPU 时间棋力收益**。

生产模型、训练器和评估源码未修改，未加依赖。运行采用冻结源码 `03f414021d51c2e50db70d0e83d93ad4d7abaad7`，已与当时 main 的运行文件逐项核对；复用 RTX 5070、JAX 0.7.2 和现存依赖。

## 证据、备份与恢复范围

原 Baseline 保持只读。独立 smoke 和 500-update 中止状态均已在 Mac、LocalServer 两个故障域，从冻结 ZIP 解压并加载；完整参数、Adam、champion、replay、主 RNG、配置、日志及清单核验一致。恢复 PASS 只说明状态完整，不表示实验有效或达到 2k。

| 产物 | SHA256 |
|---|---|
| 原冻结协议 | `c9728e3d6b1c39e64752abb3b636ff3bde69212541499392c983a88460aad1d6` |
| A：20k latest 策略 | `575b7d57a1aa03f69e23374b5402ac8ec6eb4739f32fa699cc646f20f92ed1a8` |
| B：500-update 完整 checkpoint | `ce0889c69a5b852ab474055d2f66db4ef387c085b264f59c7f46885e2232aea3` |
| B：中止状态冻结归档 | `0c8c86d482b8902b010d6a0fc77e77f63d5c877eb2fccd5a04d775516efce924` |
| 本轮完整证据归档 | `4fd87ffc88a5de98f8c6914b34f401089b054407072c7755ac7e872436bbe71f` |

本地证据目录为 `reports/ntp-ablation-20261008/`，入口包括 `completion.json`、`restart-audit.json`、`protocol.json`、`audit-restarts.py` 及两端恢复回执。完整包为 `reports/ntp-ablation-20261008.zip`，回执为 `reports/ntp-ablation-20261008-archive-verification.json`。包内含原 Baseline 输入归档、源码、配置、执行日志、自检和完整中止状态；`verify-package.py` 可在有兼容 NumPy 的环境中复核，无须 GPU。

`stage2000-full.zip` 沿用目标阶段名，实际保存的是 500-update 中止状态，不能按文件名认定完成 2k。原协议与证据包保留不改，后续结案文字通过独立补充件归档。`reports/` 被 Git 忽略，原始证据不会随 clone 获取；公开报告中的相对路径是证据定位信息，不是下载链接。

归档采用哈希与恢复核验，不是 WORM；GPU 基础镜像只记录身份，未整镜像导出。WP/ADP 权重沿用既有冻结输入，其官方发布身份未独立认证；本轮未使用它们进行正式评估。

## 后续仅强化协议准入

后续复用历史 Baseline 时，必须在任何 GPU 预检、smoke 或训练之前完成[会话历史一致性联合核验](../experiment-evidence.zh-CN.md#执行前会话历史一致性门禁)：还原实际会话及更新边界，联合核对 checkpoint metadata、恢复来源、每次配置与执行记录，再与新计划逐项比较。缺失、矛盾或未经声明的学习条件变化均应直接 INVALID_PROTOCOL / STOP。

现有 `audit-restarts.py` 只提取日志边界，并针对本轮输出固定的失败结论；退出码 0 不能作为准入 PASS。它不是已实现的完整联合门禁。将来另获实验授权后，可在一次性准入检查中复用其边界提取，补齐交叉核验，不必改生产训练器或增加通用框架。

若另行批准 NTP 重试，B 必须从零初始化，匹配 `0→8→2,000→20,000` 会话边界，不能恢复本轮的 500-update 状态。匹配重启边界只能排除本次发现的明显混杂，不能保证两种目标的自博弈轨迹逐位一致，也不能消除单训练 seed 的不确定性。

原冻结协议中的 20k latest 比较、2,000 副 WP/ADP 配对评估和统计门槛均未执行，保留在证据包中用于审计，不再作为现行执行计划。此前角色机制分析仍为 **INCONCLUSIVE / STOP**；本轮结案不改变其结论。
