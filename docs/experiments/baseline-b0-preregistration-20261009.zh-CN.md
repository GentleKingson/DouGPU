# B0：新 Baseline 零 GPU 预注册草案

日期：2026-10-09，Asia/Hong_Kong。状态：**PROTOCOL_DRAFT / TRAINING_PAUSED；B1_NOT_ENTERED / B2_NOT_ENTERED / G_NOT_ENTERED**。

源码基线：`8f9bfcf6f6b7d2afa15b8feec466fd6075ff88a5`。本轮开始时 HEAD、origin/main 与实时远端 main 一致，工作区干净。本文件是待审阅的设计冻结草案，尚非获准执行的预注册；**当前获准 GPU 预算=0 秒，实际使用=0 秒**。不申请预算，不执行 B1/B2，不提交 Git。本轮只新增此文件，不修改配置、算法、依赖、框架或旧结案。

## 1. 科学问题与范围

唯一问题：保持有实际运行来源的 DouGPU 配方和三段恢复结构，在 **3 个全新、分别初始化的训练 seed** 上，20,000 latest 相对该 seed 的 2,000 latest，能否在两组哈希固定的 WP、ADP 对手上重复观察到 Balanced 胜率改善，同时未触发角色退化规则？这是有限 seed 的 Baseline 重复性筛选，不是机制归因、模型晋升或 Ataraxos 复现。

逐项阅读并保留以下结案：

| 材料 | 本草案不能越过的边界 |
|---|---|
| [Ataraxos 可行性](ataraxos-feasibility-20261009.zh-CN.md) | NO_GO / STOP，成套权重与可复算性 NOT_VERIFIED；本 B0 不重启该方向 |
| [R0–R2](cross-replay-20261009.zh-CN.md) | R2=RESEARCH_SCREEN_INCONCLUSIVE / STOP；跨 Replay 误差交叉不构成新机制或干预依据 |
| [旧 Baseline 棋力报告](baseline-strength-20261008.zh-CN.md) | 单 seed 探索性 POSITIVE_SIGNAL，只作为问题和成本的来源；名义 CI 不涵盖训练 seed 总体 |
| [E/F 会话机制](session-mechanism-20261009.zh-CN.md) | ENGINEERING_CLOSED；新的执行证据门禁已验收，但旧 Baseline 仍 **INVALID_PROTOCOL** |

**旧 seed `20261009` 不可充当合规独立 seed，不计入 n=3，不用于合并置信区间或达标票数，也不作为新运行的初始化来源。** 旧参数数值可读、旧对局统计可复算，均不补足当时缺失的 Actor seeds、实际源码身份和恢复输入摘要。旧 NTP、batch512、角色机制结论全部不变。

B0 仅做本草案。为避免授权歧义，本文件把可能的未来 B1 定义为输入资格、正式执行意向冻结与三条新训练链，把 B2 定义为独立强对手评估和统计闭环；两者均须另行明确批准范围与预算，当前不启动。

## 2. 实际配置来源与唯一候选

先读取 `reports/baseline-20261008-seed20261009/` 的原始三阶段 ZIP，重算外层 SHA256，并检查内层 manifest 对 `run/resolved_config.json`、`run/session_config.json` 的哈希；两份配置相等。只读 ZIP/JSON，不装载参数、Replay 或优化器。本次不是重新执行完整历史门禁。

| 阶段 | 外层 ZIP SHA256 | 归档内 resolved_config SHA256 |
|---|---|---|
| smoke | `3c1da4e7fc97ddd9ee3c5d08cfba58ccda51c6032f9f9c73a58653247e6989a0` | `1c0b2eae9614eac16ca32a1628717f6260d9148213a18920154c76820a0ad025` |
| stage2000 | `fcf71d8cb9c249353c3df2bc3de6511dc6901a95e7108accf25fa5304902b99d` | `ecd8552b00b1794800e71e388f4538bbc5ec0e94c184659729e497f513030092` |
| stage20000 | `1702ae91c4aca6dceff455f4d96bc14b7f6b112e942905d3402f56bfdfc6a1ad` | `f834d4bf1b7cbcce73e02b9f7a34b2b870012107ab49ee84b32e4b4674daf406` |

旧外层 `smoke.json/stage2000.json/stage20000.json` 写 `max_hours=1.0`，但实际会话分别为 `0.978735892507765 / 0.9542678677373462 / 0.8592265039682389`；旧 `run-stage.py` 用剩余预算改写 `--hours`。因此不将外层草稿冒充实值。实际配置还展开了 `historical_opponents=[]`、`historical_fraction=0.5`。

唯一候选 C0：复用上述 resolved 配方，在当前修复后的源码上从零训练；下表完整列出 9 个 ModelConfig 和 53 个 TrainConfig 字段。只有 seed、selection seed、绝对 target 与新执行时间上限按本草案取值。新 seed 和时间上限是设计选择，不是历史测量值；不生成可执行 JSON。

| ModelConfig 字段 | 冻结候选值 |
|---|---|
| width / layers / heads / ffn / q_hidden | 128 / 3 / 2 / 512 / 256 |
| max_seq / vocab / bf16 / remat | 512 / 32 / true / true |

| TrainConfig 字段 | 冻结候选值 |
|---|---|
| engine / backend / require_tpu / attention_impl | douzero / cuda / false / cudnn |
| seed / eval_seed | 第 4 节按 seed 行指定；同一训练链三段不变 |
| workers / envs_per_worker / actor_groups | 8 / 16 / 2 |
| infer_batch / action_chunk | 64 / 2048 |
| infer_history_buckets / infer_fused / eval_optimized_inference | false / false / false |
| learner_remat / learner_prefetch / history_groups | false / true / 2 |
| ready_first / actor_learner_overlap / selfplay_kv_cache | false / false / false |
| historical_opponents / historical_fraction | [] / 0.5（空池，无历史对手训练） |
| sample_credit / replay_version_updates | true / true |
| micro_batch / accumulation | 64 / 4；effective batch=256 |
| replay_capacity / replay_max_age / fresh_samples / updates_per_cycle | 65536 / 64 / 2048 / 4；资格年龄为 64×4=256 次成功更新 |
| lr / weight_decay / grad_clip | 0.0001 / 0.00001 / 1.0 |
| ntp_weight / belief_weight | 0.02 / 0.05 |
| epsilon_start / epsilon_end / epsilon_frames | 0.10 / 0.03 / 1000000 |
| max_hours / max_cycles / target_updates | 1.0（每会话内部次级限时）/ 100000000 / 8、2000、20000 |
| checkpoint_seconds / keep_checkpoints / save_replay / resume | 300 / 3 / true / true |
| eval_every / eval_deals / eval_workers | 25 / 256 / 4 |
| promotion_role_margin / eval_bucket_batch / eval_kv_cache | 0.05 / true / false |
| worker_timeout / log_every | 300.0 / 25 |

此配方与 [rtx5070_throughput.json](../../configs/rtx5070_throughput.json) 的差异仅是 seed、eval_seed、max_hours、target_updates 和默认字段展开；不能误用 balanced 配置或 TrainConfig 的其他默认执行参数。该已跟踪配置 SHA256 为 `b3d547dda91518cb8f7bd32f750ca5de8402dc2cfcf29ee55fd0cf4cef1bf484`。

来源运行的 Python/JAX/NumPy 为 `3.12.14 / 0.7.2 / 2.5.3`；配置 backend 为 `cuda`，JAX 实际 `versions.backend` 为 **`gpu`**，门禁应冻结后者，不能误填 `cuda`。拟复用既有 RTX 5070 单 GPU 和报告中的镜像 `sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e`；本轮未探测机器或镜像。未来须核对完整包版本、驱动、硬件、CPU Torch 及镜像可用性，不许隐式升级或回退 CPU。完整新环境身份目前 **NOT_VERIFIED**。

## 3. 真实执行流程与三段门禁

| 追踪源码 | 对未来执行意向的约束 |
|---|---|
| [run_local.py](../../run_local.py)：load_config → prepare_run → prepared → main → invoke | prepare 展开 dataclass 并保存配置、规则锁和 trainer_source.zip；运行有目录锁和 CUDA 项目锁。`--cycles` 是追加 cycle，不是成功更新数；target_updates 必须在会话配置中。`--hours` 会覆盖实际配置，不允许事后用模糊命令代替冻结值 |
| [train.py](../../dougpu/train.py)：main / save / stop_reason / endpoint | 从 Store 恢复完整状态、主 RNG、collection_credit、累计日志；Actor 在途牌局重启。成功更新才递增 updates；到 target 才 COMPLETE。先写 session_end，再保存最终 ZIP；session_end 日志本身不证明保存成功 |
| [checkpoint.py](../../dougpu/checkpoint.py)：Store.save / load_latest | ZIP、manifest、最后写入的 marker 才组成完整代际；NPZ policy 不是恢复状态。Store 可能回退较旧有效代际，也可能从镜像选更新代际；必须把实际装载 SHA 与冻结输入比较。独立端点不能依赖 keep=3 的滚动保留 |
| [protocol_gate.py](../../scripts/protocol_gate.py)：audit_execution / execution_sessions | 使用 `--execution` 语义，核查三段事件、完整配置、版本、源码、Actor 顺序/seeds、恢复 SHA、Adam.step、Replay/主 RNG 和独立端点。旧 audit/compare 是历史 NTP 专用，不能拿它审新 Baseline |
| [runtime.py](../../dougpu/runtime.py)：source_identity | 核对根目录及 dougpu 的 Python 字节和已装载模块路径；不能将 git unknown 当来源证明。还需外层冻结 scripts、配置、依赖与规则锁；不是所有执行组件都有运行时装载证明 |
| [check_session_chain.py](../../scripts/check_session_chain.py)：check | 除 execution gate 外，还比对事前意向 SHA 与事后回执的全部冻结字段。`audit_execution` 单独不验证意向链接，也不代表训练授权；该脚本的 `--train` 是 CPU 0→1→2 验收，不能用于本草案的 GPU 三段运行 |

每个新 seed 独立新目录；三段仅在该目录内恢复，不 `import-checkpoint`，不 fork 旧 seed。`resume=true` 仍要求第一段目录及镜像没有旧代际；第一段 `resume_input=null`，参数由本 seed 初始化、Adam.step=0、Replay 空，主 RNG 从本 seed 开始。

| 会话 | 起止成功更新 | 增量 | 输入 | 离开该段的必要证据 |
|---|---:|---:|---|---|
| S0 | 0→8 | 8 | 全新初始化 | 完整 8-update ZIP+marker、actor_start/start/session_end、退出码 0、COMPLETE |
| S1 | 8→2000 | 1992 | S0 实际 SHA | S0 独立副本已读回；恢复 SHA/边界正确；完整 2k 端点 |
| S2 | 2000→20000 | 18000 | S1 实际 SHA | S1 独立副本已读回；恢复 SHA/边界正确；完整 20k 端点 |

每段返回后、下一段前，把 ZIP+marker 复制到 `endpoints/<sha256>/` 并在两个独立存储位置读回校验；冻结 2k/20k 的 latest policy 与它们对应 params 的关系。完整保留参数、Adam m/v/step、champion、Replay、主 RNG、collection_credit、累计日志、源码与退出/限时回执。失败即 STOP，不补第 4 会话，不用周期 checkpoint 冒充预定端点。

事前意向冻结每 seed 的三段全配置、版本、边界、源码快照 SHA、source_lock、预算与统计口径；事后仅追加 UUID、真实输出 SHA、实际输入 SHA、退出事实，引用原意向 SHA。未来 checkpoint、worker seeds、实际输出和 deal 哈希本轮不存在，不伪造为事前测得。worker seeds 必须记录实际启动值，按第一段初始 RNG/后段输入 checkpoint RNG 独立复算；同 seed 三段恢复不等于三次独立训练。

## 4. 候选 seed 与数据隔离

下面整数为本草案新指定的候选，不是旧结果筛出来的。已在可访问 `docs/configs/reports/dougpu/scripts/run_local.py` 的 JSON、JSONL、Markdown、Python 中用 `rg --no-ignore` 检索，未见匹配；未扫描压缩包内部、已删除数据或其他机器，不能宣称全球未用过。B1 前任何已用证据或碰撞均阻断，不能运行后换 seed。

| 新链 | train.seed | 训练 selection eval_seed | B2 预检 seed | B2 正式 seed |
|---|---:|---:|---:|---:|
| C0-S1 | 2026101101 | 2026101201 | 2026101301 | 2026101401 |
| C0-S2 | 2026101102 | 2026101202 | 2026101302 | 2026101402 |
| C0-S3 | 2026101103 | 2026101203 | 2026101303 | 2026101403 |

依次 S1、S2、S3；训练链共享算法和硬件，不共享参数、Replay、优化器、RNG 或 champion。数值不同的 seed 只是可审计的随机流约定，不证明逐局绝无重复或跨硬件逐位复现。

- 保留每链既有 256-deal 训练选优流程，但不据其结果停早、挑 seed 或选择正式候选。主比较只用该链 2k latest 与 20k latest；best 不进入 B2。
- 三链全部完成并通过资格后才开启正式评估结果。每 seed 正式 **2000 副牌**，两端点、WP/ADP、两阵营共用同一有序 deal 列表；不同训练 seed 使用不同正式列表。总正式对局 `3×2000×2端点×2对手×2阵营=48000`，不是 48000 个独立训练样本。
- 预检每 seed 4 副，两个端点×两对手×两阵营重复两次，总 192 局；只验身份、合法性和重复一致性，不计统计、不据胜率调参。本 B0 未生成这些牌局或牌序。
- 生成规则沿用 `np.random.default_rng(seed).permutation(DECK)`，固定 NumPy 版本；记录实际有序 rank 数组（拟统一 int16、小端、C 顺序、形状 N×54）及 SHA256。仅在后续获准时生成；开局前核对所有候选共用数组和角色切片，检查正式/预检/selection/可取得历史评估的实际 deal 交集。若发生碰撞，不静默重抽，STOP 后修订未执行协议。
- 旧 selection `202610091`、工程 `202610092/202610093`、正式 `10759382473515`、预检 `272920744840185` 及既有最终留出不复用。无法证明所有历史训练随机发牌与新评估绝无偶遇；隔离承诺是无主动复用、无 Replay 注入、无结果调参，不作绝对不重叠声明。
- 这批正式结果用于一次性重复性筛选，使用后不再称为未见 holdout；现有 final holdout 本轮及本草案 B2 都不用，不追加“最终测试”来救结果。

## 5. WP/ADP 输入资格与评估接口

WP、ADP 是两组对手权重族的标签；本方案测量二者上的**胜率**，不把“ADP 对手”误写成已经计算了 ADP 分数指标。来源为旧报告中用户 Drive 文件，官方发布身份仍 **NOT_VERIFIED**。若研究目标必须是“官方认证 DouZero”，则当前直接 NO_GO，不能用本组固定权重替代认证。

本次只重算磁盘文件 SHA256，六项全部与 `reports/independent-strength-20261008/inputs/opponent-sources.json` 相等；未反序列化模型、未加载 Torch、未做前向。该来源清单 SHA256 为 `eb573d9c642ad1010da9d6771ae9ca4ea585c19910f1ff3485450db98fa69a05`。

| 对手 | 角色 | SHA256 | 原始 Drive 文件 ID |
|---|---|---|---|
| WP | landlord | `132e26479fcb69f457ddbf1ad32a7bdc3aa73cde80a28cc91f29a62c6075955a` | `16_ilJ6VCU-6uY3uDkcvHqZci-IAUM81N` |
| WP | landlord_down | `964720faf583f4905662c94aaf73b1a97352b76ae3686fcc949beabbacdd95d1` | `1RMXLj_2gxNiZK-oo-EkxIZBKlKiV_zBP` |
| WP | landlord_up | `b849ea518e66f088d635d3a29956da880f7b5ca89fabc5f70fc356c106ca6a15` | `1GWyHguO8K7MYqFRQ-RDG7HNp0vMgzL7N` |
| ADP | landlord | `6f2971813495e9c509cbd4a101213c49587472942fa0f018d04c139a6a647c2d` | `1k3pKAynRLX4vUWPfborKvvPaJv33T3R8` |
| ADP | landlord_down | `c173dbe30258f19b6392c30ab1e330b6b3090bbd9ccdc89a9b18123a6ab6e2d9` | `1gOh9a0YsWRmCvugiDD6wqCncvNljPb5N` |
| ADP | landlord_up | `7fb7c495ff095db2ddadd8f9f8fd53cad840211578dabc83bbcc8d75085f843e` | `1lhG3g_lczgJV57Wiz_ZuGoIc2sXB_3gY` |

未来资格检查要求六文件齐全、哈希精确匹配、角色顺序正确、规则锁一致，并对可信权重用 `weights_only=True`、`strict=True` 和有限值检查；禁止 pickle 不安全回退、替换某角色、只拿文件名作证。后两项当前只沿用旧报告验收事实，未在本轮重做。

[evaluate.py](../../dougpu/evaluate.py) 的 DouZeroOpponent 安全加载和 strict 已存在，但没有通用的全参数有限性审计；旧独立评估调用方曾补该检查。未来还须用已冻结的检查流程核验参数前后不变。[evaluation.py](../../dougpu/evaluation.py) 的 `paired_difference` 只检查 seed/deals/outcomes 形状，不验证对手哈希、牌序或规则身份；调用方必须补足这些资格，不能把函数返回结果当完整可比证明。

B2 候选执行设置沿用旧强对手评估：DouZero 引擎，候选 CUDA、BF16/manual attention，batch=64、action_chunk=2048、workers=0、KV=false，CPU Torch 对手；与训练 BF16/cuDNN 分开记录，不声称逐位等价。`run_local evaluate` 默认 best，未来必须显式选择 **latest** 并指向已冻结的 2k/20k 导出，不能从持续变化的 latest 路径取样。

每份逐副 outcome 保留两列：候选地主对对手两农民、候选两农民对对手地主。只有来源 checkpoint/政策 SHA、对手六角色 SHA、规则源码、牌序 SHA、角色映射、全执行设置和环境身份一致时，才允许配对。两个候选参数若完全相同亦 STOP，不用重复导出冒充学习。

## 6. 跨 seed 统计与冻结判据

每 seed、每对手、每副牌先计算 `d_L = win20k_L − win2k_L`、`d_F = win20k_F − win2k_F`，以及 `d_B=(d_L+d_F)/2`。报告各自均值×100（百分点）。同一副两角色及两端点必须作为一个配对单元，不能把农民两位当独立观察。

每 seed 分别使用既有 `paired_difference` 的 2000 次完整 deal bootstrap、RNG=`formal_seed+12345`；名义 95% CI 仅作描述。未来分析对同一重采样分布取**单侧 alpha=0.05/18** 的下/上界：18 项=3 seed×2对手×3指标。这样对正向筛选使用的 18 个下界统一校正；上界另作同样校正的负向筛选，不把两族合称联合双侧 95% CI。2000 次下极端分位数约由第 6 个排序值决定，尾部精度有限；不因结果临界而追加 bootstrap 或牌局。

跨 seed 等权报告 WP/ADP 各自的三项均值、三 seed 标准差、范围及每 seed 完整结果；不能按局数加权、把三链 Replay 行或 48000 局当训练重复。**训练 seed 总体 CI 与检验力 NOT_ESTIMATED**：3 seed 不支持可靠估计总体分布，本草案不伪造总体显著性。以下 POSITIVE_SIGNAL 只称“在预定三个新 seed 上重复观察到”，不等于总体稳定提升。

决策按优先级一次作出，所有差值为 20k−2k：

1. 资格缺失、任一链不完整或任一正式 outcome 缺失：`INVALID_PROTOCOL / STOP`；不在幸存 seed 上改成 n=2 分析。
2. 任一 seed、任一对手的 Balanced 校正上界 <0，或任一角色校正上界 <−5pp：`NEGATIVE_SIGNAL / STOP`。−5pp 沿用配方中 promotion_role_margin=0.05 作为本草案预先指定的容忍线，不是已证明合理的业务损失阈值。
3. 仅当六个 seed×对手 Balanced 校正下界全部 >0，且十二个角色下界全部 ≥−5pp：`REPEATED_POSITIVE_SIGNAL / STOP`。不自动晋升、不进入 G、不追加预算。
4. 其余完整情形：`INCONCLUSIVE / STOP`。不换 seed、不加第 4 seed、不补抽、延长训练或改 LR/NTP/belief/batch。

没有功效保证，也不使用旧 +6.625/+7.000pp 点估计当作真实效应来承诺通过。样本量 2000 副/seed 来自已运行评估规模，是预算受限的设计选择。现有 helper 不直接提供上述 18 项校正界或跨 seed 汇总；未来需先冻结一个最小离线统计入口并留下配对/分位数自检，本 B0 不新增代码、不假装已经实现或验证。

## 7. 预估成本与未批准的硬上限

**获准预算仍为 0；下列数字仅为草案估算与候选上限，不是申请、预约或执行许可。** B0 不运行 doctor/preflight/selftest：它们分别可能执行 kernel、前反向或新牌局，不能因标称“检查”而绕过范围。

历史成本来源：`reports/baseline-20261008-seed20261009/completion-timing.json`（SHA256 `7ba603edb32c6d01064a39d93a721127294537f428fc4186c75af2cecf62be11`）记录三段训练进程 10.242+93.445+601.105≈704.792 秒，preflight 40.227 秒；GPU 阶段总进程墙钟 750.309 秒，至恢复验证累计 1223.419 秒。`reports/independent-strength-20261008/output/exit.json`（SHA256 `8552b882044c3be409192c196dcc2dbc71cc47ef8be3ad45c797dc310805c2c1`）记录 24000 正式局及预检约 378.263 秒。

粗略线性估算：三 seed 训练含 preflight 约 `3×(704.792+40.227)=2235.057秒`，B2 48000 局约 `2×378.263=756.526秒`，合计约 **50 分钟单 GPU 进程墙钟**；若按旧训练恢复验证的完整耗时计，约 `3×1223.419+756.526≈74分钟`。新 seeds、编译缓存、机器负载、归档和检查流程均可能改变耗时；这些不是新测量或性能保证。

| 未来工作块 | 草案硬上限（包含等待、保存及退出） |
|---|---:|
| 每 seed：preflight及 0→8 块 | 600 秒 |
| 每 seed：8→2k 块 | 900 秒 |
| 每 seed：2k→20k 块 | 2100 秒 |
| 三 seed B1 合计 | 10800 秒（3 小时） |
| B2 全部预检、12 组正式评估及退出 | 1800 秒（30 分钟） |
| 总计 | **12600 秒（3.5 单 GPU 小时），未批准** |

单 GPU 串行，块间不能挪用剩余额度。每块从首次设备进程启动前开始计时，内部校验/传输和安全退出计入上限；最后 90 秒预留停止与保存，不能用 grace 延长硬截止。预算余额不足则不启动下一块；达到限时或非 target 终止均为失败，不缩小正式规模来凑完成。

`tc.max_hours` 从训练内部初始化后的计时点开始，且不覆盖全部最终保存，不能单独充当外部硬限时。旧 `run-stage.py` 固定旧目录/seed，并允许额外 90 秒停止，不可直接复用来宣称满足本草案。**符合新上限的外部监督执行意向和终止验证尚未冻结，属于准入缺口**；不为解决该缺口在 B0 写新框架或运行代码。

## 8. 全局 STOP 与尚未确定项目

任何运行资格不通过、非零退出、任一已记录 nonfinite step、源/依赖/权重/规则哈希漂移、端点/Adam.step 不符、fallback 到错误代际、角色数据缺失、恢复链断裂、镜像或独立读回失败、牌序不一致、未批准重启/配置变更或超时，均停止整个三 seed 方案。训练器连续三次 nonfinite 才退出，但本协议比它严格：出现一次已记录事件即不准继续下一段；不能把“程序没报错”当门禁通过。日志不能覆盖全部失败尝试的限制继续保留。

| 已确定 | 尚未确定 / B1、B2 准入缺口 |
|---|---|
| 科学问题、唯一配方、3 条候选 seed 链、三段边界、评估规模/隔离、统计与 STOP 口径 | 用户执行批准及 GPU 授权；当前为 0 |
| 原始配置片段与六权重字节哈希匹配 | 权重官方发布认证；新运行装载、有限性与接口验收未做 |
| 当前 main 和所追踪源码 | 新 trainer_source.zip、执行意向及其 SHA、完整环境快照、镜像/设备可用性未核验 |
| 事前输出/输入身份应如何串联 | 未来 UUID、worker seeds、checkpoint/政策/牌序 SHA 均未产生；执行前值与事后回执须分开 |
| 新候选 seed 在指定可访问文本中无匹配 | 压缩包内部、删除历史、远端未扫描；无全历史唯一性保证 |
| 旧配方耗时与保守候选上限 | 新运行成本、外部硬预算监督和最小统计入口未验收 |
| 有限三 seed 的重复性决策 | 训练 seed 总体显著性/功效没有估计，不据此推断机制或更广棋力 |

这些缺口不得用“合理默认”自动补成 PASS。之后即使全部补齐，仍只允许重新审阅该草案；没有明确批准，不执行 B1/B2，也不提交 Git。

## 9. 本轮检查与证据可达性

本轮仅读源码、Markdown、JSON、ZIP 中的文本成员以及权重原始字节。历史包配置 manifest 检查、六权重 SHA256 检查、候选 seed 文本检索已执行；首次尝试把外层配置当实际值的断言因 max_hours 差异失败，随后按归档实值更正，未掩盖差异。没有导入项目模块、JAX/Torch，未创建运行目录、配置或执行意向。

`reports/` 原始材料被 Git 忽略，不随克隆发布；旧报告提供归档获取与恢复说明。上述身份只说明本次可访问本地字节，完整旧证据链不在 B0 重验。源码引用均相对于本文，绑定页首 main；原始配置 SHA 和权重 SHA 已直接写入本文，便于独立审阅。

最小检查仅限本文的 UTF-8、表格/链接、字段覆盖、预算与对局数算术、空白及单文件差异范围。检查结果不构成执行准入；**PROTOCOL_DRAFT / TRAINING_PAUSED，当前 GPU 预算和使用均为 0。**

实际结果：标准库静态检查退出码 0；9 个 model 字段、53 个 train 字段覆盖完整，13 个本地链接有效，表格列数及预算/对局数算术通过。`git diff --check` 退出码 0；新文件 `git diff --no-index --check /dev/null <本文>` 退出码 1（文件有差异），无空白诊断。tracked diff 与暂存区均空，唯一未跟踪文件为本文。未运行训练、前向、反向、新牌局、门禁执行器或应用测试。
