# NTP 辅助目标单变量消融

日期：2026-10-08（Asia/Hong_Kong）。复核后正式结案：**NTP_ABLATION：CLOSED / INVALID_PROTOCOL；Training：PAUSED**。下一阶段仅为 **PROTOCOL HARDENING**，不启动训练。此前授权的 Gate 1–3 已在本轮停止，剩余预算不作为重试授权；未完成 A/B、未运行强对手评估，不能回答 NTP 是否改善棋力。此前角色机制分析保持 **INCONCLUSIVE / STOP**。

## 结案与后续准入

维护者复核认可停止决策。本轮确认的是 NTP=0 计算路径及 CPU/GPU 工程检查可用，没有证明棋力改善、退化或非劣。生产默认 NTP=0.02 不变，500-update 状态永久保留为中止证据，不续训、不作为新协议起点。

后续如复用历史 Baseline，必须先完成[执行前会话历史一致性门禁](../experiment-evidence.zh-CN.md#执行前会话历史一致性门禁)，联合核验训练日志、checkpoint metadata 与执行记录。现有 `audit-restarts.py` 仅覆盖日志边界且针对本轮输出固定的失败结论，不能把它的退出码 0 当作准入 PASS，也不能把它称为已实现的联合门禁。原脚本及冻结协议保留不改；将来另获授权后，在新协议中复用其边界提取并补齐交叉核验。

新 B 必须从零初始化，计划会话边界为 `0→8→2,000→20,000`。匹配这些边界仅排除本次发现的重启差异，不保证自博弈轨迹逐位一致，也不消除单训练 seed 的不确定性。本轮仅强化准入要求，不改生产训练器、不增加通用框架、不自动重试。

## 停止原因与实际结果

执行代理在 Gate 0 核验了端点配置和完整状态，却遗漏了历史日志中的训练会话边界。A 的实际轨迹为 **0→8→2,000→20,000**，8-update smoke 状态曾直接恢复为正式 Baseline；本轮冻结的 B 为 **0→2,000→20,000**，独立两更新 smoke 不参与正式训练。`dougpu.train` 每次恢复都会重新创建 actor、抽取 worker seeds，且不恢复在途牌局。额外重启可能影响自博弈轨迹，因此该差异不能归因于 NTP。

发现遗漏后立即发出 STOP，B 在 **500 成功更新 / Adam.step=500** 安全保存，端点状态为 **INCOMPLETE / signal**，没有重开或挑选另一条训练轨迹。原协议及哈希保持不变，以下原冻结设计作为偏差审计记录，**不可直接再次执行**。这是协议核验错误，不是 NTP 无效或训练数值失败的证据。

| 阶段 | 实际结果 | GPU 进程墙钟 |
|---|---|---:|
| Gate 0 | 两端完整状态与身份核验通过；会话边界漏核，准入不完整 | 0 |
| Gate 1 CPU | NTP-only 自检通过，58 项现有测试通过 | 0 |
| Gate 1 GPU preflight | 通过 | 38.014 秒 |
| Gate 1 独立 smoke | 两周期、2 更新，NTP=0，Q/Belief/梯度正常 | 10.004 秒 |
| Gate 2 正式 B | 500 更新停止，nonfinite=0 | 42.156 秒 |
| Gate 3 / Gate 4 | 未运行 | 0 |
| 合计 | 502 次新增更新，未产生棋力比较 | **90.174 秒 / 1.503 分钟** |

smoke 与中止状态都已在 Mac、LocalServer 从冻结 ZIP 独立解压并加载，完整参数、Adam、champion、replay、主 RNG、配置、日志及清单一致。500-update checkpoint SHA256 为 `ce0889c69a5b852ab474055d2f66db4ef387c085b264f59c7f46885e2232aea3`；其冻结归档 SHA256 为 `0c8c86d482b8902b010d6a0fc77e77f63d5c877eb2fccd5a04d775516efce924`。GPU 退出后空闲。恢复 PASS 仅指状态完整，不表示实验有效或达到 2k。

证据入口为 `reports/ntp-ablation-20261008/completion.json`、`restart-audit.json` 及两端恢复回执；`audit-restarts.py` 留下了可运行的会话边界检查。总归档与恢复回执为 `reports/ntp-ablation-20261008.zip`、`reports/ntp-ablation-20261008-archive-verification.json`，包含原 Baseline 输入归档、执行源码、配置、日志、自检及中止状态。`stage2000-full.zip` 沿用目标阶段名，实际只含 500-update 中止端点，不得按文件名认定完成 2k。

保持 NTP=0.02 及生产默认配置不变。若另行批准重试，应在任何 GPU 工作之前验证 A 的全部会话边界，冻结匹配的 `0→8→2k→20k` 协议并重新初始化 B；不能继续此 500-update 状态，也不能将本次开销从实验历史中删去。原 A 的 8-update smoke 属于正式轨迹，其训练进程耗时应计入 A：三段合计 **704.791 秒**。下文仅两段 694.550 秒的原预算参考不是完整 A 训练时间。没有可报告的棋力或效率优势。

## 原冻结设计（存在上述遗漏）

A 使用 10 月 8 日既有 seed `20261009` 的 20,000-update latest；策略 SHA256 `575b7d57a1aa03f69e23374b5402ac8ec6eb4739f32fa699cc646f20f92ed1a8`。B 从同 seed 全新初始化，唯一正式配置差异为 `ntp_weight: 0.02 → 0`。保持 DMC、Transformer、Belief=0.05、有效 batch256、优化器、采样、replay 和训练内评估不变。配置来自实际 Baseline 归档，见 [NTP-off 配置](../../configs/experiments/ntp-off-20261008.json)，不使用当前默认值推测旧设置。

运行源码采用原归档 `03f414021d51c2e50db70d0e83d93ad4d7abaad7`；与当前 main 的运行文件逐项比对一致。隔离目录为 `reports/ntp-ablation-20261008/` 和 `LocalServer:/root/dougpu-ntp-ablation-20261008/`。原 Baseline 只读，禁止导入到 B 或改变普通续训语义。

B 与原 A 一样在 2,000 成功更新停止、冻结并恢复核验，之后恢复到绝对 20,000 成功更新。2k 仅作工程节点；正式只比较 20k latest，不使用 best、champion 或中间棋力选优。两周期 smoke 独立设置 `updates_per_cycle=1`、`max_cycles=2`、`target_updates=2`，确保两周期恰好两次成功更新；不把 smoke 状态用于正式训练。正式每周期更新数仍为 4。

## Gate 与停止条件

- Gate 0：已在 Mac 与 LocalServer 重新加载原 2k/20k 完整状态，核对参数、Adam、champion、replay、主 RNG、日志、规则及对手哈希；两份结果除 NumPy 环境版本外一致。旧角色报告及选优报告已提交，原结论不变。
- Gate 1：CPU 自检要求关闭 NTP 后 loss/梯度为零、Q 与 Belief 有效、两次 Adam 更新有限、普通恢复拒绝更改 NTP；再执行 GPU preflight 与隔离两周期 smoke。任一步失败即停止。
- Gate 2：仅新增 B 一条轨迹。端点要求 `Adam.step == successful updates == 2000/20000`、非有限更新为零。每个端点完成跨主机复制及完整恢复核验后才进入下一阶段。
- Gate 3：WP、ADP 各使用同一批新冻结的 **2,000 副牌**；A/B 每副分别担任地主与农民团队，共 16,000 局。发牌 seed `239960424249927`，在可访问的历史文本中未见重复；不声称穷尽已删除历史。复用 `paired_difference` 的 2,000 次完整 deal 配对 bootstrap，不另增统计实现。
- Gate 4：仅全部筛选门槛通过才记录为复现候选；两个新训练 seed 的配对训练及最后留出须另立协议并另行授权。本轮不运行最后留出。

差值统一为 **NTP-off − NTP-on**，以下条件对 WP、ADP 都必须成立：

| 指标 | 进入复现候选的门槛 |
|---|---|
| Balanced | 名义双侧 95% CI 下界 > 0 |
| 地主 | 单侧 99% 下界 ≥ −1pp |
| 农民团队 | 单侧 99% 下界 ≥ −3pp |
| 正确性及归档 | 恰好 20k 更新、nonfinite=0、完整证据在两故障域恢复一致 |

失败、身份变化、预检不通过、备份失败或预算耗尽时立即停止；不追加牌局、不换 seed、不调 NTP 权重、不扩大 batch。筛选未过则保留现有配置；速度优势不能代替非劣性证据。单 seed 名义区间不覆盖训练 seed 方差或多重检验的整体错误率。

## 预算、备份与效率口径

60 分钟统计所有 GPU Docker 进程墙钟，包括预检、smoke、训练及正式评估；每阶段预留 30 秒预算用于停止，CPU 验证与传输另计总墙钟。Store 镜像目录仅作服务器本地暂存，`remote_ok=True` 不冒充跨故障域成功；监督脚本持续复制已提交 ZIP/marker 至 Mac 并核对 SHA256，复制或校验失败通知 GPU 进程停止。冻结端点还须在两端独立完整加载核验。故障域为 Mac 和 LocalServer 两块独立主机磁盘。

报告分别列出成功更新/训练进程秒、训练日志阶段时间、全部 GPU 阶段墙钟以及含备份的总墙钟。原 A 两个训练阶段进程合计 **694.550 秒**，全部 GPU 阶段 **750.309 秒**，含恢复核验总墙钟 **1,223.419 秒**。A 为历史运行，硬件负载、缓存、备份开销及自博弈轨迹可能不同；即使 B 更快，也不能据此直接声称等 GPU 时间棋力提升。

机器可读冻结协议为 `reports/ntp-ablation-20261008/protocol.json`，SHA256 `c9728e3d6b1c39e64752abb3b636ff3bde69212541499392c983a88460aad1d6`。原始证据被 Git 忽略，最终报告需附实际归档和恢复回执。

## 实施范围与证据限制

生产训练、模型和评估源码不修改，不加依赖。现有 `sample_losses` 支持 NTP=0，`check_training_semantics` 禁止恢复时改权重，部署策略已剔除辅助头。仅增加实验配置、协议与一次性监督/校验脚本。GPU 环境复用 RTX 5070、JAX 0.7.2、现存冻结依赖与 Docker 镜像。

本实验检验当前配方下的条件性收益，不把其他任务的辅助训练论文当作斗地主效果证明；此执行协议不新增未经核验的文献结论。WP/ADP 六角色权重沿用已冻结输入，官方发布身份仍未独立认证。归档采用哈希及恢复校验，不是 WORM；GPU 基础镜像记录身份但未整镜像导出。
