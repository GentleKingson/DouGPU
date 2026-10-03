> 发布版说明：本文整理自 `reports/eight-hours-seed44-20261002/final-report.md`。正文中的同目录文件、bundle、脚本和归档均指该原始实验目录，不是本文所在目录；它们未包含在本次文档提交中。历史状态、数值与产物哈希保持原样。

历史阶段报告：以下状态和建议截至本轮结束。后续 seed43/45 复现已完成，当前保留 6h、停止预算扩张；原始证据未随 Git 发布，见[当前决定与证据获取](../experiment-evidence.zh-CN.md)。

6h -> 8h continuation: PASS
preregistered gates: 5/5 passed
landlord negative-point warnings: none
12h / seed45: NOT STARTED
next step: independent preregistered replication, not automatic longer training

# seed44 八小时续训验证报告

预注册：2026-10-02；完成：2026-10-03，Asia/Hong_Kong。

## 1. 结论

在不改模型和训练配置的条件下，从固定 seed44 完整 6h 状态追加两小时训练，WP、ADP 的 balanced 与地主点估计均提高，五项预注册 gate 全部通过。本轮没有触发地主负点差警示。

地主改善还不能称为稳定复现：WP 与 ADP 地主的 nominal95% CI 下界虽为正，单侧99% 下界仍略低于零。本轮证明的是通过预登记的 -1pp 非劣风险线，并非在更严格水平上证明地主必然提高。WP balanced 的99% lower 也只有 +0.0667pp，通过幅度较小。

下一步优先预注册其他独立训练 seed 的同档 6h -> 8h 复现，例如现有 seed42/43 的完整 6h 状态，分别使用新牌集。暂不修改模型，也不直接进入 12h。本任务没有执行这些后续实验。

## 2. 冻结协议与执行边界

- 协议 SHA256：`a6b9d49660a57a2e33ced3d10b1ca8d8f40250f12750c1936754ff82e2a54806`。协议和执行脚本哈希在容器启动及任何本轮新训练/评估前保存，交付时逐文件核验未变。
- 固定起点为最近完成的 seed44 完整 6h session-end checkpoint，不按胜率挑选。旧结果启发了这一选择，本轮是同轨迹条件性时间增量实验，不是新的独立训练 seed。
- 生产源码 SHA256：`6f495604bf9fc703694eac0be08df582e61619e0112f73833c9edeabac5399e5`；源包 SHA256：`2c343e62d126d0f5b70588e063d2eb792962954a7278ad591fa7f4fb4c4ae7e1`。完整规范化 ModelConfig、TrainConfig 均与 6h 状态相同。未改变模型、lr、损失、采样、奖励、replay、epsilon、batch 或硬件执行配置。
- 镜像复用 `douzero-test:latest`，ID `sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e`。未重建镜像或升级组件。
- LocalServer RTX5070 / 12227 MiB / driver595.91.07；Python3.12.14、JAX/JAXLIB0.7.2、NumPy2.5.3、Torch2.8.0+cpu、CUDA runtime13.0.96、cuDNN9.18.1.3。完整包版本匹配冻结清单；安装仅发生在私有 Docker deps。
- WP/ADP 六份权重与前轮字节一致、严格加载及 SHA 校验通过，只读挂载。它们是用户提供的冻结权重，未独立认证为官方发布版本。
- 本轮只在隔离 Docker 挂载目录 `/var/tmp/dougpu-eight-hours-Qsc2Zj` 执行。服务器原仓库 `/opt/DouZero` 未写入。没有宿主机 Python 安装、驱动或 Docker daemon 修改、重启、prune、提交或 push。

## 3. 训练与恢复

命令：`python run_local.py train --run-dir /app/runs/seed44-8h --hours 2`。这里的 2 是追加 session 预算，未使用 `--hours 8`。

| 状态 | 6h 起点 | 8h 终点 |
|---|---:|---:|
| total_seconds | 21600.137396 | 28800.208220 |
| successful updates / Adam step | 832536 / 832536 | 1108427 / 1108427 |
| cycle | 208136 | 277109 |
| frames | 426265344 | 567522432 |
| complete_samples | 426260536 | 567515166 |
| games | 11910312 | 15746551 |
| replay size / roles | 65536 / 0,1,2 | 65536 / 0,1,2 |
| collection/sample credit | 2104 | 542 |
| champion_cycle | 0 | 0 |
| nonfinite updates | 0 | 0 |

追加 7200.070824 秒、275891 次成功更新，以生产时间预算安全结束。total_seconds 是 session 内生产口径，包含采样、学习、等待/编译与保存等开销，不是纯 GPU kernel 时间。停止时间不要求完整四次更新一组的周期边界，未按更新数择优停止。

导入前后完整参数及辅助头、champion、Adam m/v/step、replay、主 RNG、训练计数、credit 和累计时间核验相同。实际 learner session-start checkpoint 也通过逐数组与元数据比较，累计时间仅自然增加 0.000051811 秒。源 6h ZIP 前后 SHA 不变；完整 8h 状态 finite、schema 与 manifest 均有效。未结束的 actor 牌局恢复后重新开始，不声称与不中断训练逐位一致。

只评估最后时间预算 session-end policy，不选中间 checkpoint。复用的旧 export helper 中 `metadata.selection=predeclared_update_node` 是历史通用标签，实际选择规则以本轮预注册的时间终点为准。

## 4. 新牌配对结果

新评估 seed984001-984048，48 chunks x 250 = 12000 副独立 deal，与登记前全部历史 report JSON 和 checkpoint metadata 的 seed 集合无重叠。6h/8h x WP/ADP 四组使用相同有序牌，每副候选作为地主和农民团队各打一局，共实际 96000 局，不是 96000 个独立样本。

统计复用冻结生产 paired bootstrap，2000 次重采样，以完整 deal 为单位。192 个 chunk 的身份、ordered-deal hash 与逐 deal outcome 均保留；运行时审计重新构造牌序哈希，并从全部原始 outcome 复算统计。

胜率单位为百分比；差值与区间单位为百分点 pp。区间是 paired 差值的 nominal95% CI，不是两个绝对胜率区间相减。

| 对手 / 角色 | 6h % | 8h % | 8h-6h pp | nominal95% CI pp | 单侧99% lower pp |
|---|---:|---:|---:|---:|---:|
| WP landlord | 16.3000 | 16.9750 | +0.6750 | [0.0083,1.3002] | -0.1167 |
| WP farmer team | 25.8667 | 26.5917 | +0.7250 | [-0.0169,1.4752] | -0.1750 |
| WP balanced | 21.0833 | 21.7833 | +0.7000 | [0.1708,1.2002] | +0.0667 |
| ADP landlord | 18.9167 | 19.7333 | +0.8167 | [0.1167,1.5502] | -0.0168 |
| ADP farmer team | 27.0750 | 27.9667 | +0.8917 | [0.1083,1.6667] | -0.0167 |
| ADP balanced | 22.9958 | 23.8500 | +0.8542 | [0.3249,1.3750] | +0.2458 |

绝对胜率各自的完整95% CI 也保存在 statistics.json。6h policy 没有变化，本轮绝对胜率与前轮有差别是因为牌集不同，不能用跨牌集相减替代本轮 paired 对照。

## 5. 地主风险与决策

本轮在新结果前把地主 margin 从历史 -3pp 收紧为 -1pp，农民仍为 -3pp。该值是保守的操作性风险容忍线，不是从结果拟合的阈值，亦不追溯修改旧实验结论。

| Gate | 要求 | 实测 lower pp | 结果 |
|---|---:|---:|---|
| WP balanced 改善 | > 0pp | +0.0667 | PASS |
| WP landlord 非劣 | >= -1pp | -0.1167 | PASS |
| ADP landlord 非劣 | >= -1pp | -0.0168 | PASS |
| WP farmer team 非劣 | >= -3pp | -0.1750 | PASS |
| ADP farmer team 非劣 | >= -3pp | -0.0167 | PASS |

两个 balanced 点差均为正、两个地主点差均非负，额外方向检查也通过，地主负点差警示为空。输出 `PASS / CONSIDER_PREREGISTERED_REPLICATION_ONLY`，不触发任何自动后续训练。

前轮 seed44 4h -> 6h 的 WP 地主为负点差，本轮 6h -> 8h 转正；只能说明本次条件对照的方向变化。不同牌集、不同时间区间不能相加或拿来证明总体稳定提高。旧三 seed 逐角色对照原样保留；seed42 原 validation 的失败不改写。

WP 农民本轮95% CI 跨零，不宣称每个角色都已显著提升。地主95% 下界接近零或99% 下界仍负，因此优先独立 seed 复现，而不是立即扩大模型、改学习率、改损失或进入 12h。近似五项单侧界限只覆盖本次固定观察，不覆盖所有历史/未来反复查看；一个训练 seed、两个固定对手也不能代表实际玩家综合水平、总体收敛或模型容量。

## 6. 测试与异常

冻结版本 GPU Docker 测试命令：`python -m pytest -q tests --junitxml=/app/evidence/gpu-tests.xml`，251 passed、1 skipped、0 failed，154.24 秒。唯一 skip 为 GPU backend 下的 CPU 条件检查 `test_cudnn_bf16_requires_cuda_backend`。未额外执行 CPU-only 全套。

doctor 的实际 GPU/BF16 检查、生产 preflight 的完整512-token backward/Adam、cuDNN/manual parity 与推理形状验证均通过。正式 pipeline exit=0，13 个要求的实验子进程均 exit=0，未发生 OOM、非有限更新或正式训练/评估失败。

准备扫描的 AppleDouble 编码异常、tar xattr 提示，以及交付核验初次检测同期 Colab 改动导致拒绝归档的经过，保存在 verification-notes.md。它们没有改变冻结协议、训练源码、学习语义、权重或原始评估结果。

本地同期有 README 的 Colab 文档变化、新 `scripts/colab_benchmark.py` 和 `tests/test_colab_benchmark_cli.py`，均保留，未覆盖或回滚。原有训练源码和原有测试字节未变；新 Colab 测试不在本次冻结252项套件中，本报告不宣称已经验证这些新增代码或 Colab 可用性。

## 7. 完整状态与归档

| 产物 | SHA256 |
|---|---|
| 6h full checkpoint | `fa739f1aa0a50441da9d70cd08814e935befd0fc53d73979bebbcf3670f20061` |
| 8h full checkpoint | `26e7a6beea08bcafa7a75b7322d8a746721de1adbbbd1fe3734a7c99c0f8965f` |
| 6h policy | `18963726daa9a682ad9e91ff7a8af2d58f10b1ff75dc3ff8d36892d4cf60aa63` |
| 8h policy | `2d878e2037a0a9552d8ab343963d06a1fa708f3e7b3cc996a3b59e5549527a1e` |
| 实际恢复 session-start full state | `79a008d4d77ac78ebd946e1cc03d9a6146855c3127abd5c28a4706d4b1fa560e` |
| evidence.tar.gz | `16e94d0f16364a7b0c512232a66a93f895b8d106420b5d310b996b242cb96c58` |

证据目录（相对仓库根目录）：`reports/eight-hours-seed44-20261002/`。归档含 406 个 manifest 文件，包括 source lock、完整配置/源码、冻结脚本、权重身份、完整起点及终点、恢复审计、测试、原始评估、统计和工作区核验。永久保留位置逐文件核验无缺失或额外文件，完整状态的 ZIP 内部 manifest、原始胜率差值与 gate 再次通过独立 stdlib 检查。可运行 `python3 verify_delivery.py` 复核；继续训练必须使用完整 ZIP 与 marker，不能用 policy NPZ 恢复。

seed42 的399、seed43 的490、seed44 前轮的492个证据文件在开始和交付时均复核，全部保留。清理后报告和交付补充在本地另外保存，不伪称它们原本就在远端实验归档内。

## 8. 清理与后续边界

永久保存与核验完成后，仅删除本轮服务器 `/var/tmp/dougpu-eight-hours-Qsc2Zj` 临时目录。本轮主容器和归档容器已自动删除；原服务器 HEAD、完整 status 和 tracked/untracked 文件字节与开始相同，原镜像、共享缓存/卷及历史 checkpoint 未删除。具体核验见 cleanup-verification.json。

本轮本地临时 staging 也只在交付文件保留后清理。同期 Colab 工作保留，不能称整个本地工作区仍与开始完全相同。本次没有执行 12h、seed45、其他 seed 的 8h 或未登记的追加评估，到此停止。
