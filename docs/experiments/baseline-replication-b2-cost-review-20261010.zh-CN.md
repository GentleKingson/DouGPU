# B2-R0 成本复核与 B2-R1 独立协议候选

日期：2026-10-10，Asia/Hong_Kong。**B2_R0_COST_REVIEW_COMPLETE；B2_R1_FROZEN_CANDIDATE_NOT_AUTHORIZED**。本轮 GPU 成本为 0、新对局为 0，未读取预检胜率或以预检结果选择模型。原 B2 **TECHNICAL_FAIL / STOP** 和全部原始协议、控制脚本、STOP、失败 ZIP 哈希不变；当前继续 **TRAINING_PAUSED / STRENGTH_NOT_EVALUATED / G_NOT_ENTERED**。

已按现有 24 份预检记录完成离线成本复核，并封存独立身份的下一轮协议候选。**3,000 秒只是待独立审批的候选硬上限，授权仍为 0，R2 未启动。** 原“批准”仅适用于已结案的 1,800 秒 B2，不转用于本候选，不借用 B1 或 B2 余额。机器数据及封存身份见[JSON 回执](baseline-replication-b2-cost-review-20261010.json)，原结案见[B2 STOP 报告](baseline-replication-b2-20261010.zh-CN.md)。

## R0：输入、复核方法和 12 组耗时

读取失败 ZIP 的 206 文件 manifest 并逐文件校验 SHA；比对本地原始 input/control/STOP 字节。原 ZIP SHA256 为 `380968e70fadb1dd36a3cd43535a37bf8a1e84f88580efa3bed78b96587d5291`，原 protocol 为 `fdcbf9d555aa6f666fe1abe431a79058f93cd08874742a8fe62d2f587fd360c5`，原 seal 为 `2f008af1a404f68f449c998349a6089d7bf721f7fe0cf6e583b31af233d65c33`。

成本脚本只解析日志中 `finished/seconds` 和原结果文件中的 `wall_seconds` 字段，不反序列化逐副 outcomes 或胜率。使用原冻结协议确定每组正式数量为 4,000 局，总计 48,000 局；不生成新牌局或改变原牌序。本机普通、`-O` 和 LocalServer 标准库 Python 的 R0 JSON 完全一致。自检覆盖不等数量的加权、1.5×与 60 秒常量以及非法数量拒绝。

每组 first 和 warmed 均为同一组合的 8 局墙钟耗时。表内正式预测为 `warmed / 8 × 4000`，未加安全系数。

| 训练链 | 对手 | 端点 | first 秒 | warmed 秒 | 4,000 局原始预测秒 |
|---|---|---|---:|---:|---:|
| C0-S1 | WP | 2k latest | 6.630017 | 0.178557 | 89.279 |
| C0-S1 | WP | 20k latest | 0.159649 | 0.154934 | 77.467 |
| C0-S1 | ADP | 2k latest | 0.180592 | 0.177801 | 88.900 |
| C0-S1 | ADP | 20k latest | 0.169287 | 0.165566 | 82.783 |
| C0-S2 | WP | 2k latest | 0.183420 | 0.181395 | 90.697 |
| C0-S2 | WP | 20k latest | 0.178765 | 0.180563 | 90.282 |
| C0-S2 | ADP | 2k latest | 0.171764 | 0.173674 | 86.837 |
| C0-S2 | ADP | 20k latest | 0.186250 | 0.186242 | 93.121 |
| C0-S3 | WP | 2k latest | 0.174719 | 0.177867 | 88.934 |
| C0-S3 | WP | 20k latest | 0.162604 | 0.160897 | 80.448 |
| C0-S3 | ADP | 2k latest | 0.178372 | 0.208161 | 104.081 |
| C0-S3 | ADP | 20k latest | 0.198318 | 0.197367 | 98.684 |

按正式数量加权：`Σ(warmed_i / 8 × 4000) = 1071.512 秒`；沿用原安全系数与余量得到 **1,667.268 秒**。原最慢组公式为 `max(warmed_i / 8) × 48000 × 1.5 + 60`，仍为 **1,933.451 秒**。这两个数来自相同原始计时；加权值只是诊断参考，未追溯替换原门禁，也未改变原 STOP。

12 组 warmed 均值 0.178585 秒、中位数 0.178212 秒、范围 0.154934–0.208161 秒、样本标准差 0.014663 秒。这里的标准差描述不同工作负载之间的分布；**每组只有一次 warmed 记录，不能估计同组计时波动。** 最慢组第二次比第一次慢约 0.029789 秒；不得把全部 first-minus-warmed 差值都当作编译成本。预热和重复测量的用法参见 [Google Benchmark 官方指南](https://google.github.io/benchmark/user_guide.html)。

## 耗时分解能够支持到哪里

| 阶段 | 现有记录支持的结论 |
|---|---|
| 24 次评估调用合计 | 10.716781 秒；其中 12 次 first 合计 8.573757 秒，12 次 warmed 合计 2.143023 秒 |
| 首个调用额外耗时 | first 6.630017 秒减 warmed 0.178557 秒 = 6.451460 秒；混合冷启动效应，不能单独归因于编译 |
| 容器评估进程 | 13.560155 秒；扣除调用后剩余 2.843375 秒，未被子计时细分 |
| 监督器启动到门禁 | 13.197123 秒；与上述进程及调用时间有重叠，不能再相加 |
| CPU Torch 推理 | 未单独计时；源码确认 WP/ADP 为 CPU Torch，但不能给出占比 |
| JAX 编译、执行、同步、传输 | 未分别计时；现有候选推理经 `jax.device_get(outputs)` 回到主机后才形成动作，因此每次评估墙钟已包含相应等待 |
| 失败包写入、传输、Mac 回读 | 没有各阶段独立计时，无法分离 |
| LocalServer 普通失败回读 | 容器包络 0.376891 秒；此前预审回读包络 0.727726 秒 |
| 失败结案过程 | 原授权记录到四次回读完成 228.625718 秒；设备开始到回读完成减评估进程为 214.464688 秒，含人工与编排等待，不能作为纯归档耗时 |

[JAX 官方性能测量说明](https://docs.jax.dev/en/latest/benchmarking.html)要求区分首次编译、异步执行同步和数据传输。本轮只检查已有实现和记录，未新增推理或编译试验。不能由上述混合耗时反推出 CPU/GPU/归档各自的百分比。

原 [`paired_evaluate`](https://github.com/GentleKingson/DouGPU/blob/main/dougpu/evaluation.py) 已支持端点配对。四副预检与 2,000 副正式输入的批处理规模不同；计时还含原统计汇总。**很小样本的线性外推及 1.5×系数不能保证正式规模、游戏长度变化、额外编译或全量统计归档的实际耗时。** 未重写评估算法、削减样本、新增依赖或并行框架。

## R1：独立冻结候选与预算口径

新修订身份 **B2-R1-20261010-v1**，未来执行身份 **B2-R2-20261010-v1**。新 protocol SHA256 为 `2eff2acdb4a4fd4ffdb69cfb2b57db233ee2afd5d218efbe30b3d4cf9b2ae50f`；新 seal 为 `8d9ef4b1ac1cdb24e5f3d5fcde88885eb40f82259d16762a071234a2b98469ff`。两者独立于原失败身份；输入中保留原 protocol/seal/ZIP 父身份与 R0 成本回执 SHA。

逐字段核验三条 seed、六份 latest 策略、WP/ADP 六份权重身份、规则锁、环境源码、正式及预检有序牌数组、角色映射、执行配置、正式 48,000 局和统计规则均与原协议一致。全部源码文件字节一致；生产评估算法未修改。完整政策、权重和牌序哈希仍在候选 `input/protocol.json`、`input/seal.json` 和报告 JSON 中可复核。

若将来获准执行，192 局资格预检会重复原预检数组；这些数组已在失败 B2 用过，**不能称为未见留出或用于选模型**。正式数组未改、正式局数仍为 0，正式结果尚未出现。原训练选优与可取得历史牌序隔离核验继续沿用[输入预审](baseline-replication-b2-preaudit-20261010.zh-CN.md)，不声称覆盖无法取得的历史材料。WP/ADP 官方身份仍为 **NOT_VERIFIED_USER_PROVIDED**。

仅执行预算、成本方法说明及新身份/路径发生变化：

| 项目 | R1 候选值 |
|---|---:|
| 全块硬上限（含本机入口、传输、预检、正式评估、统计、归档、两地回读） | 3,000 秒 |
| 远端评估容器截止（含 TERM 宽限、清理） | 启动后 2,500 秒 |
| 结案预留 | 500 秒，原为 300 秒 |
| TERM 宽限 / 清理预留 | 90 / 10 秒，均在 2,500 秒内 |
| 额外准入缓冲 | 10 秒 |
| 评估器软截止 | 远端启动后 2,390 秒 |
| 正式准入门禁 | 原最慢 warmed 公式，小于当时剩余软窗口才执行 |
| 加权预测用途 | 诊断，不用于放宽门禁 |
| 当前授权 | 0 秒；要求新 `B2_R2_ONLY` 授权及精确 protocol/seal SHA |

预算控制同步修改本机控制器、远端监督器、评估入口的授权检查与统计收据的时间资格上限；每份改动相对原脚本的 diff 已归档。软截止比 TERM 时间提前，90 秒宽限与清理都计入容器窗口；本机入口更早开始的 3,000 秒独立硬上限仍覆盖最后 Mac 回读。原 `readback.py` 字节未变。若失败，同一 R2 身份不允许重启。

用户建议的约 **2,547 秒**可以在新增 500 秒结案预留的口径下复算：`1933.450890 + 13.197123 + 500 + 90 + 10 = 2546.648013`，向上取整为 2,547。原 300 秒预留下同一名义计算为 2,347 秒。R1 再留 10 秒准入缓冲，名义计算为约 2,557 秒。**这些是基于旧门禁耗时的规划数，包含混合墙钟、严格小于门禁和整数 timeout 的近似；本机/远端入口时间还不同，不能称为测得的总耗时下界或可完成保证。** 500 秒是事前保守预留，不是成功归档耗时实测。3,000 秒仅作为供审阅的候选硬上限。

统计停止规则原样保留：六组 Balanced 点差均 > 0，且 12 项地主/农民团队点差均 ≥ −5pp，才报告 **DIRECTIONAL_SIGNAL_ONLY / STOP**；否则 INCONCLUSIVE_STOP，技术失败则 TECHNICAL_FAIL / STOP。无模型晋升、追加 seed/牌局、更多训练或 G。

## 封存、CPU 检查和审阅入口

R1 准备包：`reports/b2-r2-candidate-20261010/revision-evidence.zip`，22,337,072 字节，manifest 170 文件，SHA256 **`b95c5214fac1c0345349806c9e612d4009c39d6e54f7cecccfd04cb9659d1cfb`**。副本：`LocalServer:/var/tmp/dougpu-b2-r2-candidate-20261010/revision-evidence.zip`。包含冻结源、输入、五份控制脚本、完整 Docker argv 前缀及动态 timeout 计算、原控制 diff、R0 脚本/回执、修订记录和 CPU 输入审核收据。

LocalServer CPU 输入审核 PASS，DeviceRequests=null、State.Pid=0、无 JAX 导入、无 forward 或真实对局。候选 seal 在两地核验一致。统计自检在普通/`-O` 模式均通过，覆盖配对身份、六组判断、零差及 −5pp 边界；预算控制自检验证 CPU/GPU 请求分离与截止拆分。准备包在 Mac 和 LocalServer CPU 容器解压、逐文件校验并运行普通/`-O` 统计自检，两地回执字节一致，decision 均 **NOT_EVALUATED**。回读收据在 ZIP 外；未将它们追加进已封存包。

R0 标准库检查可运行：

```sh
python3 reports/b2-cost-review-20261010/cost-review.py check
python3 reports/b2-cost-review-20261010/cost-review.py > /tmp/b2-r0-cost-review.json
```

第二条使用保留在本地的原失败归档；结果应与候选包的 `input/cost-review.json` 一致。R1 准备包可用已有 NumPy 的 Python 运行候选 `control/readback.py <ZIP> <回执路径> <SHA>`，只做 CPU 校验。未来完整执行入口及精确 argv 已冻结于 `input/execution-commands.json`；**当前没有 authorization.json 或 output/budget.json，不运行控制器。**

本次 Git 发布范围为报告、JSON 与索引；reports 被 Git 忽略，没有公开 ZIP 下载地址。本轮不重复 B1 的 60,000 次训练更新，不把 192 局预检解释为棋力失败。原 B2 STOP 保持结案；下一次 GPU 执行仍需人类独立批准本候选预算和协议。
