> 发布版说明：本文整理自 `reports/six-hours-seed43-20261002/final-report.md`。正文中的同目录文件、bundle、脚本和归档均指该原始实验目录，不是本文所在目录；它们未包含在本次文档提交中。历史状态、数值与产物哈希保持原样。

历史阶段报告：以下状态和建议截至本轮结束。后续实验已完成，当前保留 6h、停止预算扩张；原始证据未随 Git 发布，见[当前决定与证据获取](../experiment-evidence.zh-CN.md)。

seed43 4h -> 6h replication: PASS
five preregistered gates: 5/5 passed
12h training: NOT STARTED
recommended next step: RUN_SEED_44

# seed=43 六小时独立长跑复现报告

日期：2026-10-02；执行机器：LocalServer。seed=44 仅为建议，本任务未启动 seed=44、12h 或其他后续训练。

## 1. 环境与源码身份

- 本地仓库根目录，HEAD `7dba89617006f0e4baa824d777c61fae19037439`。开始和归档前均只有原有未跟踪报告 `docs/SIX_HOURS_2026-10-02.zh-CN.md`；未提交、未 push。
- LocalServer 原仓库 `/opt/DouZero`，HEAD `ce6c4237fd18207b9dd8c3c15b94e9cdaa8fdac5`。原有脏工作区未写入；开始和归档前的 HEAD、完整 status，以及原 tracked/untracked 文件逐文件 SHA256 全部相同。
- 本轮仅在 `/var/tmp/dougpu-seed43-j713bI` 隔离执行，使用上一轮冻结 `source.tar.gz`，未使用当前 main 更新替代冻结源码。
- 当前生产源码与冻结生产源码哈希相同：`6f495604bf9fc703694eac0be08df582e61619e0112f73833c9edeabac5399e5`；源码包 SHA256 `2c343e62d126d0f5b70588e063d2eb792962954a7278ad591fa7f4fb4c4ae7e1`。保存了 current-vs-frozen diff，生产源码和冻结 tests 无差异。
- Docker `douzero-test:latest`，ID `sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e`，与基准一致；未 rebuild、未更换镜像。
- RTX 5070，12,227 MiB，驱动 595.91.07；真实 JAX backend 为 GPU。Python 3.12.14、JAX/JAXLIB 0.7.2、NumPy 2.5.3、Torch 2.8.0+cpu、CUDA runtime 13.0.96、cuDNN 9.18.1.3。所有冻结 dependency versions 逐项匹配，无宿主环境安装或升级。
- 协议在任何训练和正式评估前冻结：SHA256 `cdfaee0ffff095c4827f2c4757d2d49b4e9a98262553c35533f8d4ec116efbec`。

## 2. 独立起点与完整配置

搜索本地 reports 和服务器原仓库、临时目录，发现 17 份历史 seed=43 完整 checkpoint。其 source lock/训练源码不匹配，因此全部拒绝复用；不是仅凭 seed 相同就继续旧状态。逐份路径、SHA256、配置差异和拒绝原因保存在 `bundle/evidence/startpoint-search.json`。

本轮全新初始化 seed=43。零更新 session-start checkpoint 的参数与冻结生产 `init_params(ModelConfig, 43)` 逐数组相同；Adam m/v 为零、step=0、所有训练计数为零，没有 seed=42 fork 来源。完整 auxiliary heads、champion 和初始状态均保留。

完整规范化 ModelConfig 相同；基础完整规范化 TrainConfig 只将 seed=42 改为 43，未仅检查几个选定字段。模型 width128/layers3/heads2/ffn512，BF16、cuDNN、8 workers、每 worker 16 envs、有效 batch256、micro64、accumulation4、offline eval_every=0、无历史对手列表，均不变。

明确的操作性例外：按照任务要求使用 `train --hours 4`，随后 `train --hours 2`。生产 CLI 将本次 session 的 max_hours 写为 4 和 2，完整 session 配置按这两个预登记预算逐字段审计；基础 run 配置保持 max_hours=2。这是停止时间预算，不是学习参数修改。lr、epsilon、loss、replay、奖励、采样、模型、对手均未修改。

## 3. 四小时训练状态

| 项目 | seed43-4h |
|---|---:|
| total_seconds | 14,400.07865647 |
| successful updates / Adam step | 552,427 / 552,427 |
| cycle | 138,107 |
| frames | 282,845,440 |
| complete_samples | 282,843,164 |
| games | 8,121,124 |
| replay size / roles | 65,536 / 0,1,2 |
| collection/sample credit | 540 |
| nonfinite updates | 0 |

四小时以生产 time limit 安全停止，完整状态审计通过，冻结最后 session-end checkpoint 为 seed43-4h。并未从四小时附近的多个 checkpoint 中挑选；成功 update 数并非预登记停止条件。沿用旧 export helper 的通用 metadata.selection 标签 `predeclared_update_node`，该历史标签并不代表本轮按固定 update 数选点；实际冻结条件以协议、session-end 状态与日志为准。

## 4. 恢复与六小时状态

从冻结 seed43-4h 完整 checkpoint 导入独立 seed43-6h run，再追加两小时。导入前后 params、champion、Adam m/v/step、replay、主 RNG、cycle、updates、frames、games、complete_samples、credit、total_seconds 均完全相同。另检查了实际 learner 的 session-start checkpoint，而不只检查启动前拷贝；数组、RNG、计数器相同，新 session 的时间计数仅自然增加 0.000054 秒。源四小时状态设为只读文件且 SHA256 前后不变。

| 项目 | seed43-6h |
|---|---:|
| total_seconds | 21,600.15254086 |
| additional seconds | 7,200.07388439 |
| successful updates / Adam step | 831,788 / 831,788 |
| additional successful updates | 279,361 |
| cycle | 207,948 |
| frames | 425,881,600 |
| complete_samples | 425,877,043 |
| games | 12,030,015 |
| replay size / roles | 65,536 / 0,1,2 |
| collection/sample credit | 1,587 |
| nonfinite updates | 0 |

参数、auxiliary heads、champion、完整 Adam、replay 数组、RNG 和计数器均通过审计，数组全部 finite。policy 与对应完整状态参数逐数组相同。未完成 actor 牌局在恢复后重新开始，不能声称与不中断的六小时轨迹逐位一致。

## 5. WP 绝对胜率

数值为百分比；括号为以完整 deal 为单位的 nominal paired-bootstrap 95% CI。

| 角色 | 4h | 6h |
|---|---:|---:|
| landlord | 15.833 [15.192,16.458] | 17.175 [16.492,17.808] |
| farmer team | 24.883 [24.100,25.675] | 26.258 [25.467,27.050] |
| balanced | 20.358 [19.900,20.825] | 21.717 [21.283,22.167] |

## 6. ADP 绝对胜率

| 角色 | 4h | 6h |
|---|---:|---:|
| landlord | 18.692 [18.000,19.375] | 19.183 [18.458,19.867] |
| farmer team | 26.133 [25.342,26.933] | 27.658 [26.908,28.458] |
| balanced | 22.413 [21.954,22.850] | 23.421 [22.975,23.879] |

## 7. Paired 6h - 4h

所有差值单位为百分点（pp），未更改生产统计：完整 deal 配对 bootstrap，2,000 resamples。

| 对手 / 角色 | 差值 pp | nominal 95% CI pp | one-sided 99% lower pp |
|---|---:|---:|---:|
| WP landlord | +1.3417 | [0.7415,1.9500] | +0.6417 |
| WP farmer team | +1.3750 | [0.6083,2.0835] | +0.4248 |
| WP balanced | +1.3583 | [0.8500,1.8292] | +0.7875 |
| ADP landlord | +0.4917 | [-0.1419,1.1502] | -0.3084 |
| ADP farmer team | +1.5250 | [0.8081,2.2333] | +0.6999 |
| ADP balanced | +1.0083 | [0.5208,1.5042] | +0.4541 |

ADP landlord 的 nominal CI 跨零，因此不能单独断言地主明显改善，也不能解释成已经证明没有差异。其 one-sided lower 仍高于预登记 -3pp 风险边界。

评估使用新 seed 982001–982048，共 48 chunks，每 chunk 250 副牌。已与历史 selection/validation/holdout seed 清单核对，无重叠。四组评估使用相同有序发牌，每副候选控制地主及农民团队各一次；192 份 chunk 文件保存 deal hash 和逐局二元 outcome。已重建每 chunk 发牌 hash，并从全部原始 outcome 重算绝对胜率和 paired difference。

总实际局数为 96,000，统计独立单位是 12,000 副 deal，不是 96,000 个独立样本。

## 8. 五项预登记 Gate

| 判断 | 99% lower pp | 固定阈值 | 结果 |
|---|---:|---:|---|
| WP balanced superiority | +0.7875 | >0 | PASS |
| WP landlord noninferiority | +0.6417 | >=-3 | PASS |
| WP farmer noninferiority | +0.4248 | >=-3 | PASS |
| ADP landlord noninferiority | -0.3084 | >=-3 | PASS |
| ADP farmer noninferiority | +0.6999 | >=-3 | PASS |

5/5 通过。沿用每次固定决定的近似 Bonferroni 解释，不声称覆盖所有历史或未来重复查看。ADP balanced 如实报告，但没有添加为第六项正式统计 gate；两个 balanced 点估计方向检查同样是在结果前登记。

## 9. seed42 与 seed43

| 独立训练轨迹 | WP balanced 6h-4h pp [95% CI] | ADP balanced 6h-4h pp [95% CI] |
|---|---:|---:|
| seed42 冻结独立 10,000-deal 留出 | +0.800 [0.280,1.350] | +1.370 [0.810,1.945] |
| seed43 新 12,000-deal 确认集 | +1.358 [0.850,1.829] | +1.008 [0.521,1.504] |

增益方向一致，两个对手的 balanced 均改善，农民团队改善在两条轨迹中一致。seed42 的 WP 地主差值 +0.03pp，seed43 为 +1.34pp；ADP 地主分别为 +0.61pp、+0.49pp。seed43 未触发预登记角色风险，但不能据两条轨迹断言所有 seed 都安全。

seed42 当时独立留出五项通过，但先前 validation gate 未全部通过，因此原整体扩展门槛仍是失败；本轮独立 seed43 确认结果不追溯改写它。两轮使用不同新牌集，不能从点估计大小直接推断某 seed 显著优于另一 seed。

本轮支持继续第三条独立长跑 seed=44，按新的预注册协议确认 seed 方差。没有将两个 seed 当成大量独立样本，没有平均两个点估计后声称整个方法必然改善，更没有证明稳定收敛或模型容量充足/不足。至少第三个长跑 seed 和新的预注册决策完成后，才重新讨论 12h。

## 10. Checkpoint、Policy 与权重身份

| 产物 | SHA256 |
|---|---|
| seed43-4h full checkpoint | `28b9ecfdbc0697a3302eb5ebf543e84bcdc7c4c017ae99c50ef2674271b0fd7e` |
| seed43-6h full checkpoint | `6f70883ecb90996abd8c1eb3b3a1da67e4ec34b696b335adcc7bc97cedfd12f3` |
| seed43-4h policy | `db88c0942ec87e09ee52dab464f97ceb5f2a50f37f5e13b5a0a76cc267e6d31d` |
| seed43-6h policy | `940a9a0c32f990b1c6e794eb67ce58275a9799e248fbf6f4fee8348573d79c41` |
| seed43 initial full checkpoint | `2216cb421a9c773e3e68d247f4f2f7d7997b9ea5b2d89b7ed042d091b4f4f9f3` |

WP/ADP 六份权重与 seed42 冻结字节相同，严格安全加载通过；来自用户提供文件，未经独立官方发布来源认证，不称其为官方认证版本。

| 对手 / 角色 | SHA256 |
|---|---|
| WP landlord | `132e26479fcb69f457ddbf1ad32a7bdc3aa73cde80a28cc91f29a62c6075955a` |
| WP landlord_down | `964720faf583f4905662c94aaf73b1a97352b76ae3686fcc949beabbacdd95d1` |
| WP landlord_up | `b849ea518e66f088d635d3a29956da880f7b5ca89fabc5f70fc356c106ca6a15` |
| ADP landlord | `6f2971813495e9c509cbd4a101213c49587472942fa0f018d04c139a6a647c2d` |
| ADP landlord_down | `c173dbe30258f19b6392c30ab1e330b6b3090bbd9ccdc89a9b18123a6ab6e2d9` |
| ADP landlord_up | `7fb7c495ff095db2ddadd8f9f8fd53cad840211578dabc83bbcc8d75085f843e` |

## 11. 测试、异常与限制

全部测试、doctor、preflight、训练和评估在原 GPU Docker 镜像中执行，未运行宿主 Python。主要命令：`python -m pytest -q tests --junitxml=/app/evidence/gpu-tests.xml`；结果 251 passed、1 skipped、0 failed，155.34 秒。唯一跳过为 GPU backend 下不能运行的 CPU-only 条件检查 `test_cudnn_bf16_requires_cuda_backend`，不另跑 CPU 全套。doctor 真实 BF16 kernel 通过；preflight 实际完整 512-token backward/Adam、cuDNN/manual parity 与推理容量检查通过。完整回归并不等同于棋力证明。

正式 pipeline exit=0，训练、评估、分析和审计过程 exit 均为 0。没有 OOM 或非有限更新，没有修改训练/评估语义或生产源码。两个普通启动问题均在训练前解决并留证：macOS tar 生成的 AppleDouble sidecar 在本轮隔离目录清除；`--no-deps` 安装遗漏 MarkupSafe/mpmath，被版本核验拦截后按旧精确版本补齐。首次失败日志保存在 `bundle/evidence/environment-setup-attempt1/`，未隐藏。

时间预算沿用生产 total_seconds 口径，包含训练 session 内采样、学习、编译等待和 checkpoint 开销，不是纯 GPU kernel 时间。seed42 与 seed43 的训练路径、更新数和新牌集不同，只有两条独立长跑轨迹，尚不能精确估计训练 seed 总体方差。96,000 局结果只支持这套冻结源码、配置、对手及预登记比较。

## 12. 证据与清理核验

最终证据目录（相对仓库根目录）：`reports/six-hours-seed43-20261002/`。原始文件在 `bundle/`，包括协议、环境/硬件、源码包和 hash、完整配置/source lock、镜像信息、版本、权重、完整 4h/6h/初始 checkpoint、policy、训练日志、exit 记录、192 个评估 chunk、deal hash、paired outcome、统计与审计。manifest 对不可重新生成证据逐文件 SHA256，排除私有 deps/cache/bytecode。

`evidence.tar.gz` SHA256：`f7076868733aa51246f4d9c8dc0a8fceb2adc935cef2a4447c52749b21d7dc12`。传输后和最终保留位置均核验 490 个 manifest 文件，无缺失或额外文件；两个完整 checkpoint 内部 manifest、六份权重、policy 和所有关键审计再次通过。manifest SHA256 及交付后的报告/清理记录身份保存在交付清单中。`python3 verify_delivery.py` 可在最终证据目录重新核验。

本轮 Docker 临时容器均已自动删除；仅删除本轮 `/var/tmp/dougpu-seed43-j713bI`，已检查路径不存在。保留标准镜像，不触碰共享 cache/volume、原仓库、旧 checkpoint 或 seed42 证据。清理后原服务器 HEAD、status 与 tracked/untracked 文件字节哈希仍完全相同。seed42 原 399 个 manifest 文件在归档前再次全部核验通过。详情见 `cleanup-verification.json`。

远端归档含原始实验证据；清理核验发生在归档传输及验证之后，因此清理记录与最终报告作为本地交付补充文件保存，不伪称它们早已存在于归档内。交付 `delivery-manifest.json` 进一步覆盖这些补充文件，`completion-audit.json` 将逐项要求与对应证据关联。任务到此结束。
