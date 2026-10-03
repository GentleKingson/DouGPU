> 发布版说明：本文整理自 `reports/six-hours-seed44-20261002/final-report.md`。正文中的同目录文件、bundle、脚本和归档均指该原始实验目录，不是本文所在目录；它们未包含在本次文档提交中。历史状态、数值与产物哈希保持原样。

历史阶段报告：以下状态和建议截至本轮结束。后续实验已完成，当前保留 6h、停止预算扩张；原始证据未随 Git 发布，见[当前决定与证据获取](../experiment-evidence.zh-CN.md)。

seed44 4h -> 6h replication: PASS
five preregistered gates: 5/5 passed
8h / 12h training: NOT STARTED
recommended next step: PREREGISTER_6H_TO_8H

# seed44 六小时训练与三 seed 对照报告

日期：2026-10-02。结论：三个独立长跑 seed 在 WP、ADP 的 balanced 4h→6h 点估计方向一致，农民团队收益较一致，地主收益不稳定。值得预注册下一档 6h→8h 小幅续训，暂不改模型，不直接进入 12h。本任务没有启动 8h、12h 或 seed45。

## 1. 冻结身份与完整配置

- 本地仓库根目录，HEAD `7dba89617006f0e4baa824d777c61fae19037439`；原有未跟踪报告未改，未提交或 push。
- LocalServer 原仓库 `/opt/DouZero`，HEAD `ce6c4237fd18207b9dd8c3c15b94e9cdaa8fdac5`；本轮不写该用户工作区。实验仅在 `/var/tmp/dougpu-seed44-ywoR4s` 的隔离 Docker 目录执行。
- 使用前两轮冻结的训练源码，生产 hash `6f495604bf9fc703694eac0be08df582e61619e0112f73833c9edeabac5399e5`；源码包 SHA256 `2c343e62d126d0f5b70588e063d2eb792962954a7278ad591fa7f4fb4c4ae7e1`。当前生产源码与冻结源码相同，不因 main 变化换实验版本。
- 镜像 `douzero-test:latest`，ID `sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e`；没有 rebuild 或升级组件。
- RTX5070 / 12,227 MiB / driver595.91.07，实际 JAX GPU backend；Python3.12.14、JAX/JAXLIB0.7.2、NumPy2.5.3、Torch2.8.0+cpu、CUDA runtime13.0.96、cuDNN9.18.1.3。完整依赖清单逐项匹配冻结版本，仅安装到私有容器 deps。
- 新协议 SHA256 `f3824e230a0e46f33096701aec2ea41163cc5ca9d6347cf0b6bf208f3782e316`，在训练与正式评估前登记。五项门槛及两个 balanced 方向检查未根据结果修改。
- 完整规范化 ModelConfig 不变；基础完整规范化 TrainConfig 相对于 seed42/43 只改变 training seed=44，而非仅人工比较几个字段。BF16、cuDNN、8 workers、16 envs/worker、有效 batch256、micro64、accumulation4、lr/epsilon/loss/replay/采样/奖励及 offline eval 均保持原值。完整配置在 `bundle/input.config.json`。
- 按任务要求由 CLI 明确设置 session 时间预算 `--hours 4` 和随后 `--hours 2`。session 的 max_hours 因停止预算分别为 4、2；其余完整字段逐项相同，不修改学习语义。
- WP/ADP 六份权重与前两轮字节一致，严格安全加载及 SHA256 检查通过，评估时只读挂载。身份见 protocol 的 opponent_hashes 和 `weight-provenance.json`。它们是用户提供的冻结权重，不宣称独立认证为官方发布版本。

## 2. 独立初始化、四小时与恢复

历史搜索发现 17 份 seed44 checkpoint，均不匹配冻结 source lock，未复用。本轮从零独立初始化；零更新 checkpoint 的 params 与生产 `init_params(ModelConfig,44)` 相同，champion 相同，Adam m/v 为零、step=0，训练计数为零，没有从 seed42/43 fork。

四小时由 time limit 安全停止，冻结最后 session-end 完整 checkpoint，不从附近状态挑选。旧 export helper 的通用 metadata.selection 标签仍为 `predeclared_update_node`，但本轮实际停止条件是预登记时间预算，不是固定更新数。

| 状态 | 4h | 6h |
|---|---:|---:|
| total_seconds | 14400.071747 | 21600.137396 |
| successful updates / Adam step | 552276 / 552276 | 832536 / 832536 |
| cycle | 138070 | 208136 |
| frames | 282768896 | 426265344 |
| complete_samples | 282766521 | 426260536 |
| games | 7986192 | 11910312 |
| replay size / roles | 65536 / 0,1,2 | 65536 / 0,1,2 |
| collection/sample credit | 1209 | 2104 |
| nonfinite updates | 0 | 0 |

从本轮冻结 4h 完整状态恢复，追加 7200.065649 秒、280260 次成功更新。导入前后 params、champion、完整 Adam、replay、RNG、所有训练计数、credit 和 prior total_seconds 完全相同；实际 learner session-start checkpoint 也逐数组和计数器通过核验，时间计数仅自然增加 0.0000525 秒。原 4h 文件只读保存，SHA256 前后不变。

完整 auxiliary heads、params、champion、Adam m/v/step、replay、RNG/counters 均保留，数组 finite。未完成 actor 牌局恢复后重新开始，不声称与不中断训练逐位一致。total_seconds 沿用生产口径，包含 session 内采样、学习、等待/编译及 checkpoint 开销，不是纯 GPU kernel 时间。

## 3. 新牌确认评估

评估 seed983001–983048，48 chunks × 250 = 12000 副独立 deal。训练前核验与历史 selection/validation/holdout 清单无重叠。4h/6h、WP/ADP 使用相同有序发牌，每副候选控制地主、农民团队各一次，总实际 96000 局；不能称为 96000 个独立样本。

每 chunk 的 ordered-deal hash、逐局二元 paired outcome 均保存。审计重新构造 192 份 chunk 的 deal hash，并从完整原始 outcomes 复算统计。保持冻结生产统计：以完整 deal 为单位 paired bootstrap，2000 resamples。

下表是绝对胜率百分比与 nominal paired-bootstrap 95% CI。

| 对手 / 角色 | 4h % [95% CI] | 6h % [95% CI] |
|---|---:|---:|
| WP landlord | 16.792 [16.133,17.483] | 16.283 [15.608,16.959] |
| WP farmer team | 24.117 [23.350,24.867] | 26.058 [25.242,26.809] |
| WP balanced | 20.454 [20.013,20.913] | 21.171 [20.700,21.638] |
| ADP landlord | 19.142 [18.433,19.875] | 19.242 [18.550,19.925] |
| ADP farmer team | 24.367 [23.575,25.109] | 28.083 [27.258,28.875] |
| ADP balanced | 21.754 [21.288,22.208] | 23.663 [23.196,24.100] |

## 4. Paired 差值与五项 Gate

单位为百分点 pp；99% lower 为 one-sided paired bootstrap lower。

| 对手 / 角色 | 6h-4h pp | nominal 95% CI pp | 99% lower pp |
|---|---:|---:|---:|
| WP landlord | -0.5083 | [-1.1333,0.1333] | -1.2418 |
| WP farmer team | +1.9417 | [1.1331,2.6750] | +0.9999 |
| WP balanced | +0.7167 | [0.2083,1.2000] | +0.1250 |
| ADP landlord | +0.1000 | [-0.5333,0.7500] | -0.6583 |
| ADP farmer team | +3.7167 | [2.9665,4.4502] | +2.8333 |
| ADP balanced | +1.9083 | [1.4333,2.3959] | +1.3374 |

WP balanced 99% lower >0；WP/ADP 的地主和农民团队 99% lower 均 >=-3pp，5/5 通过。门槛未改变，没有将 ADP balanced 偷加为第六项正式 gate；如实报告它并检查预登记的方向条件。近似 Bonferroni 解释只适用于这次固定五项决定，不覆盖所有历史和未来重复查看。

## 5. 三 Seed 逐角色对照

各数值来自每个 seed 自己的冻结独立留出/确认集，以 6h-4h 差值 pp 表示。

| 对手 / 角色 | seed42 差值 [95% CI] | seed43 差值 [95% CI] | seed44 差值 [95% CI] |
|---|---:|---:|---:|
| WP landlord | +0.030 [-0.610,0.720] | +1.342 [0.741,1.950] | -0.508 [-1.133,0.133] |
| WP farmer team | +1.570 [0.740,2.410] | +1.375 [0.608,2.084] | +1.942 [1.133,2.675] |
| WP balanced | +0.800 [0.280,1.350] | +1.358 [0.850,1.829] | +0.717 [0.208,1.200] |
| ADP landlord | +0.610 [-0.080,1.390] | +0.492 [-0.142,1.150] | +0.100 [-0.533,0.750] |
| ADP farmer team | +2.130 [1.240,3.000] | +1.525 [0.808,2.233] | +3.717 [2.966,4.450] |
| ADP balanced | +1.370 [0.810,1.945] | +1.008 [0.521,1.504] | +1.908 [1.433,2.396] |

农民团队的增益方向在三个 seed、两个固定对手上均一致。地主结果明显不如农民稳定：seed44 的 WP 地主为负点估计，其 CI 跨零，不能断言显著退化，也不能说已经证明没有差异；只可说通过本次 -3pp 非劣风险门槛。ADP 地主三条轨迹的 CI 均跨零，不把 balanced 改善外推为两个角色都稳定改善。

seed42 的独立留出五项通过，但原 validation gate 未全通过，原整体扩展门槛仍失败，本轮不追溯改写。seed43/44 各自新确认集五项通过。不同 seed 的牌集不同，不能根据点估计大小给模型排名、宣称某 seed 显著更强，或挑最高绝对胜率的 checkpoint 作为新实验起点。

这里是三条独立训练轨迹，不是数万条独立训练 seed；不将三个点估计简单平均后宣称总体方法必然改善或稳定收敛。逐 seed、角色的 99% lower 和风险标志完整保存在 statistics 的 three_seed_role_comparison。

## 6. 决策与下一阶段边界

本轮按结果前冻结的规则输出 `PREREGISTER_6H_TO_8H`：seed44 五项通过、两个 balanced 点估计为正，且前两条冻结轨迹方向一致。继续训练尚有收益证据，不优先扩大模型或修改学习算法。

建议下一任务先冻结新的 6h→8h 协议、追加两小时预算、未使用的牌集和角色风险规则，再从事先选定的完整 6h 状态续训。重点监测地主，不能只看 balanced 或农民提升。若地主风险越界或收益不能复现，停止扩大预算，调查角色与 seed 方差；不能据失败直接推断模型容量不足。

本轮到此结束，未启动 8h、12h、seed45，没有自动执行下一阶段。

## 7. 完整状态与 Policy SHA256

| 产物 | SHA256 |
|---|---|
| seed44-4h full checkpoint | `7ae00c78f430a94351fabda289b52d490e6530ad8f17410fe7c9d909bfdbe58c` |
| seed44-6h full checkpoint | `fa739f1aa0a50441da9d70cd08814e935befd0fc53d73979bebbcf3670f20061` |
| seed44-4h policy | `abb0ccc3d779ac0795a4bddfeee67a16116516615b74abaacf1f7c3a6b6a9880` |
| seed44-6h policy | `18963726daa9a682ad9e91ff7a8af2d58f10b1ff75dc3ff8d36892d4cf60aa63` |
| seed44 initial full state | `7ceb9b276a22ecd302f93822cc8e92c7adbc7c04e50c80fbd7850fc334dea508` |

继续训练必须用 full checkpoint，不能只用 policy NPZ。

## 8. 测试、异常与证据

同一原镜像的 GPU Docker 环境运行 `python -m pytest -q tests --junitxml=/app/evidence/gpu-tests.xml`：251 passed、1 skipped、0 failed，155.14 秒。唯一 skip 为 GPU backend 下不能执行的 CPU 条件检查 `test_cudnn_bf16_requires_cuda_backend`。无需额外 CPU-only 全套。doctor 的真实 BF16 kernel、GPU backend 与 preflight 完整 512-token backward/Adam、cuDNN/manual parity、推理形状检查均通过。

正式 pipeline exit=0，各训练/评估/分析/运行时审计过程 exit=0；无 OOM、非有限更新或训练评估失败。Linux tar 忽略 macOS provenance xattr header，不影响源码字节；已确认没有 AppleDouble sidecar，源码哈希独立通过。独立的本地交付 verifier 初稿漏改一个 seed=43 期待值，被正确的实际 seed44 配置拦截；只将该期待值修为 44 后重新完成全部审计，首次失败脚本和记录保留在 delivery-audit-attempt1 文件中。未改冻结协议、生产源码、训练评估算法、数据、权重或组件版本。

完整证据位于仓库相对路径 `reports/six-hours-seed44-20261002/`，包括冻结协议、完整配置/source lock、环境/镜像身份、源码、权重身份、两个完整状态与初始状态、恢复审计、日志、exit 记录、192 个 chunk、outcome、deal hash、统计、三 seed 对照、completion audit 和 manifest。seed42 的 399 个旧证据文件、seed43 的 490 个旧证据文件在开始及归档前均再次核验。

`evidence.tar.gz` SHA256：`7d56eb418356f59fca928e780544420eded7575aa5a1acea99b739b0e831e9a9`。传输后和最终保留位置均核验 492 个 manifest 文件，无缺失或额外文件；两个 full checkpoint 的内部 manifest、六份权重及 policy 身份再次通过。可在最终证据目录执行 `python3 verify_delivery.py` 复核。

本轮临时容器均自动删除，仅删除本轮 `/var/tmp/dougpu-seed44-ywoR4s`，已验证路径不存在。原服务器 HEAD、完整 status、tracked/untracked 文件字节哈希与初始快照相同；标准镜像、共享 cache/volume、旧 checkpoint、seed42/43 证据及原有未提交文件均保留。见 `cleanup-verification.json`。

清理记录及最终报告生成于归档传输、验证之后，作为本地交付补充保存，不伪称它们原本包含在远端归档中。`delivery-manifest.json` 覆盖这些补充文件，`completion-audit.json` 逐项关联完整实验和三 seed 角色对照证据。本任务到此停止，不运行下一档预算。
