> 发布版说明：本文整理自 `reports/eight-hours-seed43-20261003/final-report.md`。正文中的同目录文件、bundle、脚本和归档均指该原始实验目录，不是本文所在目录；它们未包含在本次文档提交中。历史状态、数值与产物哈希保持原样。

# seed43 八小时独立复现与两 seed 汇总

历史阶段报告：以下建议截至本轮结束，第三 seed45 现已完成。当前保留 6h、停止预算扩张；原始证据未随 Git 发布，见[当前决定与证据获取](../experiment-evidence.zh-CN.md)。

日期：2026-10-03，Asia/Hong_Kong。

## 结论

seed43 的 6h 到 8h 复现结果为 **INCONCLUSIVE，预注册五项门槛通过 4 项**，不能记为第二次 PASS。WP、ADP 的地主与 Balanced 点差均为正，两个地主单侧 99% 下界均转正；但 WP Balanced 只有 +0.2000pp，99% 下界为 -0.3917pp，未通过严格大于零的改善门槛。WP 农民团队点差为负，95% 区间跨零。

与 seed44 汇总后，ADP Balanced 和两个对手的地主改善方向在两条独立训练轨迹上相同，但 WP Balanced 的统计通过结果没有复现。当前证据不足以把 8h 设为默认，更不足以进入 12h；也不能把 WP 门槛未过解释为已经证明无收益或明显退化。

按用户给定路线，下一步可单独预注册第三 seed 的同档确认，重点检查 WP Balanced 与农民团队；本轮没有启动第三 seed、追加评估或 12h。冻结程序的单轮决定仍保留 `STOP_BUDGET_EXPANSION_AND_INVESTIGATE_ROLE_VARIANCE`，跨 seed 建议不是对该决定的追溯改写。

## 起点与训练语义

使用 seed43 自己的六小时 session-end full-state，SHA256 `6f70883ecb90996abd8c1eb3b3a1da67e4ec34b696b335adcc7bc97cedfd12f3`。ZIP 内部 manifest、全部内容 SHA、marker、模型与完整 TrainConfig 均验证，未改 seed44 状态、未人工拼造 6h。

历史口径需要校正：seed43 与 seed44 都有独立从零初始化证据，但均在四小时停止后以完整状态恢复到六小时，不是无中断的六小时进程。两者恢复结构可比；seed42 还有更早的诊断续训历史。恢复会重开未完成 actor 牌局，不声称与不中断运行逐位相同。

模型、Adam、replay、探索、lr、损失、采样、有效 batch、执行配置、offline eval 均未变。完整规范化配置与 seed43 6h 相同；把 training seed 仅用于配置比较时替换为 44 后，也与 seed44 对照配置完全相同，并未修改任何 checkpoint。命令为 `python run_local.py train --run-dir /app/runs/seed43-8h --hours 2`，只追加两小时。

| 状态 | 6h | 8h |
|---|---:|---:|
| 累计生产训练秒数 | 21600.152541 | 28800.221944 |
| 成功更新数 / Adam step | 831788 | 1107860 |
| cycle | 207948 | 276967 |
| frames | 425881600 | 567232768 |
| complete samples | 425877043 | 567225628 |
| games | 12030015 | 15890419 |
| sample credit | 1587 | 1308 |
| replay 有效行数 | 65536 | 65536 |
| 非有限更新 | 0 | 0 |

追加 7200.069403 秒、276072 次成功更新；按预注册时间终点停止，不按胜率或更新数择点。导入前后，以及实际 learner session-start 边界，均核验 params/辅助头、champion、Adam m/v/step、replay、主 RNG、计数器与 credit。源 6h SHA 前后不变。复用导出 helper 的 `predeclared_update_node` 是历史通用标签，本轮实际选择依据为时间预算 session-end。

## 冻结协议

协议 SHA256 为 `877e913aaea5a067babb9af9681701d3d96607e522e509cf8ad821817de83039`，在容器启动、新训练与新评估之前冻结；驱动文件哈希一并登记。未在看到结果后改变任何门槛。

完全沿用 seed44 6h 到 8h 的五项检查：单侧 99% 配对 bootstrap 下界，WP Balanced > 0pp，WP/ADP 地主 >= -1pp，WP/ADP 农民 >= -3pp。另保留两个 Balanced 点差为正、两个地主点差非负的方向要求。2,000 次完整发牌对重采样；表中 95% 区间是逐项名义区间。五项单侧界限的近似 Bonferroni 解释只针对该次固定观察，不控制整个历史及未来序贯决策的总体错误率。

新留出 seeds 985001 至 985048，48 chunks，每 chunk 250 副，共 12,000 个独立发牌单位，已扫描全部已有 report JSON 与 checkpoint metadata，和 `eval_seed=900001`、历史 selection/validation/holdout seeds 隔离。6h/8h 与 WP/ADP 四组共用同一有序牌集，每副交换候选地主与农民团队两种安排，共 96,000 局，不能当作 96,000 个独立样本。

WP/ADP 六份角色权重及 SHA 沿用 seed44，安全 `weights_only=True` 与 strict 加载通过。这些是冻结的用户提供权重，不宣称本轮独立认证了官方发布身份。全部 192 个 chunk 的对手/策略/协议身份、逐副 outcome 与 ordered-deal hash 均保留，生产审计重建牌序、重新汇总及计算 paired difference。

## seed43 结果

绝对胜率单位为百分比，8h 减 6h 差值及区间单位为百分点 pp。

| 对手 / 角色 | 6h % | 8h % | 差值 pp | 配对 95% CI pp | 单侧 99% 下界 pp |
|---|---:|---:|---:|---|---:|
| WP 地主 | 16.8583 | 17.6417 | +0.7833 | [0.1581, 1.3833] | +0.0166 |
| WP 农民团队 | 26.2333 | 25.8500 | -0.3833 | [-1.1083, 0.3669] | -1.2083 |
| WP Balanced | 21.5458 | 21.7458 | +0.2000 | [-0.2793, 0.6708] | -0.3917 |
| ADP 地主 | 19.3000 | 20.9833 | +1.6833 | [1.0500, 2.3500] | +0.9333 |
| ADP 农民团队 | 28.0000 | 28.2500 | +0.2500 | [-0.5000, 1.0250] | -0.6417 |
| ADP Balanced | 23.6500 | 24.6167 | +0.9667 | [0.4707, 1.5000] | +0.3791 |

两个地主、两个农民非劣检查通过；WP Balanced 改善检查未通过，方向检查均通过。因此为 INCONCLUSIVE，不是 PASS，也不满足预注册的明确 FAIL 条件。ADP Balanced 的 99% 正下界作为完整报告指标保留，但它不是五项 gate 中额外加入的第六项。WP 地主的正下界只有 +0.0166pp，不能忽略其接近零的幅度。

## 两 seed 汇总

每个 seed 均用自身同牌的 8h 减 6h 差值，不能把跨 seed 的绝对胜率相减。下表括号内为各轨迹条件下的单侧 99% 下界，单位 pp。

| 对手 / 角色 | seed44 差值（下界） | seed43 差值（下界） | 两 seed 等权点差均值 |
|---|---:|---:|---:|
| WP 地主 | +0.6750 (-0.1167) | +0.7833 (+0.0166) | +0.7292 |
| WP 农民团队 | +0.7250 (-0.1750) | -0.3833 (-1.2083) | +0.1708 |
| WP Balanced | +0.7000 (+0.0667) | +0.2000 (-0.3917) | +0.4500 |
| ADP 地主 | +0.8167 (-0.0168) | +1.6833 (+0.9333) | +1.2500 |
| ADP 农民团队 | +0.8917 (-0.0167) | +0.2500 (-0.6417) | +0.5708 |
| ADP Balanced | +0.8542 (+0.2458) | +0.9667 (+0.3791) | +0.9104 |

seed44 为 PASS 5/5；seed43 为 INCONCLUSIVE 4/5。两 seed 的地主与 Balanced 点差同向，但农民收益不稳定，尤其 WP 农民点差反向。本轮没有多个指标反转，也没有证据把 WP 称为明确退化；主要缺口是 WP Balanced 的小收益尚未跨 seed 重复达到原统计门槛。

等权均值仅为描述性汇总，不提供假装拥有大量独立 training seeds 的 pooled-deal CI。两条训练轨迹和不同牌集不足以可靠分解训练 seed 方差与评估抽样方差。可继续第三 seed 确认，但不得通过重抽本轮留出、放宽门槛或挑中间 checkpoint 补成 PASS；默认预算不变，不进入 12h。

## 验证与保留

复用 LocalServer RTX 5070、既有镜像 `sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e` 与冻结依赖。生产源码 SHA 为 `6f495604bf9fc703694eac0be08df582e61619e0112f73833c9edeabac5399e5`，没有修改生产代码、驱动、镜像、宿主依赖或训练设置。仅在 `/var/tmp/dougpu-eight-seed43-Hdhgt4` 隔离 Docker 执行。

GPU 回归 251 passed、1 skipped、0 failed；doctor 与完整 preflight 通过。冻结套件不包含后来添加的 Colab benchmark 测试，没有额外声称 CPU-only 全套或 Colab 已验证。正式 pipeline 及 13 个要求的子进程 exit=0，实际恢复审计、完整状态与统计重算通过。

| 保留产物 | SHA256 |
|---|---|
| 6h full checkpoint | `6f70883ecb90996abd8c1eb3b3a1da67e4ec34b696b335adcc7bc97cedfd12f3` |
| 8h full checkpoint | `9323d2b8f4e2f0196c8f7cceb01606eda4caa2aa3bdc93f3ef83145443734005` |
| 6h policy | `940a9a0c32f990b1c6e794eb67ce58275a9799e248fbf6f4fee8348573d79c41` |
| 8h policy | `4efde851e44b51d138a852d887baddd7f899367ffe691998e662a74333205b21` |
| evidence.tar.gz | `c24314581435f3b583a39176c1ba21049c6724be529f5def8475644a4d71793b` |

归档含 403 个 manifest 文件，包括完整 6h/8h 状态、全部中间 checkpoint、原始 outcomes、冻结协议/脚本、源码、对手权重、配置及测试日志。`bundle/evidence/cross-seed-summary.json` 保存机器可读汇总，`bundle/evidence/statistics.json` 保留全部绝对胜率区间及配对差值。可运行同目录 `python3 verify_delivery.py` 重验交付；生产审计已从原始 outcomes 重算 bootstrap，独立 stdlib 交付检查重验 ZIP 内容、身份、原始胜率差值、gate 与均值。

本地归档 403 个文件及完整 checkpoint 核验通过后，已只删除本轮远端临时目录；容器自动移除，原镜像、共享缓存/卷保留。清理与工作区保持检查见同目录 `cleanup-verification.json`。历史 seed42/43/44 六小时及 seed44 八小时的 399/490/492/406 个证据文件已再次逐项核验。本地原有未提交文档和远端用户仓库不改动，不 commit、不 push。
