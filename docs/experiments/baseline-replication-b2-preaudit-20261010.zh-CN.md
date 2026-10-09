# B2 独立强对手评估：零 GPU 输入预审

日期：2026-10-10，Asia/Hong_Kong。结论：**B2_INPUT_PREAUDIT_PASS / AWAITING_SEPARATE_AUTHORIZATION**。B1 保持 COMPLETE / EXECUTION_QUALIFIED，训练 PAUSED；B2、G 仍 NOT_ENTERED。本轮获准和实际 GPU 时间均为 **0 秒**，真实新对局、前向推理及训练更新均为 0。

用户要求先做零 GPU 输入预审，再单独批准最多 1,800 秒 B2。本文完成第 1 步；**1,800 秒是待批准的独立申请，不能使用 B1 剩余预算**。原 B0/B1 结案及历史实验结论不改写。完整机器回执见 [JSON](baseline-replication-b2-preaudit-20261010.json)。

## 输入和执行身份

评估源码绑定开始时干净 HEAD `d7e6e124229a056f936596d26e4b452f45aff064`，用 Git archive 建立独立 source；与 B1 冻结清单逐项比较，Python、配置、requirements/constraints 字节相同。生产源码、配方和依赖未修改。本轮仅在被忽略的 reports 中保存一次性审计、现有评估函数调用及统计/预算/读回脚本。

B1 401,337,416 字节归档重新计算 SHA256，仍为 `4d86e41e0a178f458ae3580f0d53f69453f91141850455d50ab795b28acf1480`。六份政策来自各链独立封存的 latest 导出；逐份核对文件 SHA、模型/selection 元数据、有限值、完整 checkpoint 的 SHA、updates/Adam.step，部署参数逐数组相等；每链 2k/20k 的实际参数不同。best 不进入评估。

| 策略 | latest policy SHA256 | 来源 checkpoint SHA256 |
|---|---|---|
| C0-S1-2000-latest | `08fde85a272649dbf92fa83231ef82ded2ad614dc57143cb66db30833ef7ffa3` | `54346127d66d8686174f7ed22ce072ed605c353f35a31f3f92ecd5e474d3d52b` |
| C0-S1-20000-latest | `175f3f7b58d73bc3e18774cb74be6d6240cd13cb5cb64ebed628236c60f91316` | `356b9c66cfb5e1dc64c1bd98e74bbbcf38a99668af032c004c3f7ed60a4f7faa` |
| C0-S2-2000-latest | `19d04b3a57f9510d45526fb7cd1610f4bfa0b5151b9179c4bd4b8e9bea62d41f` | `128343d3d6675d75737948de21c919098dba1e2599cc186bb1011da09c20254b` |
| C0-S2-20000-latest | `7c8015c1946904be923b5bc7e33a035214ccb49cab0f7cb9bb1f9a07e713b567` | `fc5e44530d90495cffc508395e077ae702fa633532d83e5dd20f1789234d49b0` |
| C0-S3-2000-latest | `f1003d321604c9b240d1e59037aa88325f4132ffcc3b022466609293536a6207` | `b588473b824635ecacc5df335dc354fbe04ef75a87b04e0de5a9f1f204ea8d2a` |
| C0-S3-20000-latest | `b6745810ad3aace417092ef7041a3bd410f9ae331f24eb0b881ff169b40b0bfa` | `cbf45197f9d4c100f5c8e5a6c382fb047b6663c35a73ececc117169be8b3c104` |

六份 WP/ADP 的 SHA 与 B0 和原 opponent-sources 清单全部一致。本轮实际通过共享 `DouZeroOpponent` 在 CPU 上 `weights_only=True` / `strict=True` 装载六角色，检查有限且留在 CPU；没有 forward、模型替换或不安全 pickle 回退。**官方发布认证仍 NOT_VERIFIED，评估对象是用户提供的固定字节**。

- 规则运行锁 SHA256：`40f47b2cd94a76fe7112140d41b85af0b7462b6c35197241fe71fc506348e1ac`；DouZero commit `718a5c920bf3361e34178a38f3b80458e176b351`，encoding=1。vendor 11 文件及独立模型导入校验通过。
- 现有 Docker image：`sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e`。只读查询设备为 RTX 5070 / `GPU-46dce763-982c-cb97-2004-b7770ccdd7eb` / 驱动 595.91.07，与 B1 一致。未初始化 JAX；本次 GPU kernel/推理能力由获准后的 192 局预检验证。
- CPU Python 3.12.14、NumPy 2.5.3、Torch 2.8.0+cpu；JAX/JAXLIB 0.7.2 仅读包元数据。完整 package 清单及 14,522 个依赖文件集合/字节精确匹配既有冻结环境；未安装、升级或构建。
- 成功 CPU 输入审计退出码 0，1.781 秒；runc、network=none、DeviceRequests=null、无 NVIDIA 节点，退出后 State.Pid=0、命名容器删除。首试因只读 source 缺少 `.cache` 挂载点，Docker 在启动 Python 前以 125 退出；保留原 inspect/log，仅补挂载目录后重验。首试也未请求 GPU。

## 实际牌序隔离

按 B0 固定整数生成真实 rank 数组，dtype=`<i2`、C 顺序、形状 N×54，保存 NPY 文件与原始数组 SHA。仅生成数组，不构造游戏。服务器 NumPy 2.5.3 已逐数组复生，与 Mac 生成的内容一致。

| 链 | train seed | selection seed / 副数 | 预检 seed / 副数 | 正式 seed / 副数 |
|---|---:|---:|---:|---:|
| C0-S1 | 2026101101 | 2026101201 / 256 | 2026101301 / 4 | 2026101401 / 2000 |
| C0-S2 | 2026101102 | 2026101202 / 256 | 2026101302 / 4 | 2026101402 / 2000 |
| C0-S3 | 2026101103 | 2026101203 / 256 | 2026101303 / 4 | 2026101403 / 2000 |

九组列表内部无重复，彼此交集为零。交集按实际初始状态归一化：三角色各自排序手牌，加公开底牌；数组顺序不同但手牌相同仍算复用。原始有序 rank 数组同时保留，不用归一化取代执行时的顺序核验。

扫描本机 reports 的 1599 份保留 JSON/JSONL/ZIP 成员，以及 LocalServer `/root/dougpu-*`、`/var/tmp/dougpu-*`、`/opt/DouZero`、`/tmp` 的 6317 份记录，提取并复生 **51 条历史/selection RNG 流**。另外校验所有可取得原始 NPY（包含 ZIP 成员），去重后 **21 份、62,112 副**，其中也包括保留的训练采集牌序。正式/预检与这些历史流、原始数组交集均为 **0**。B0/B1 中仅预定的 B2 seed 单独标为 design_only，不冒充已经使用。

初扫读取异常来自故意损坏的 pytest 夹具、macOS metadata 和实际为完整 JSON 的 `.jsonl` 文件；按来源排除测试夹具/metadata，修正 JSON 读取后最终扫描错误为 0，过程说明保留。没有读取旧 sealed final holdout 的结果或使用它救本轮实验。**已删除记录、只存在于未解包 tar 的历史及所有训练随机发牌无法穷尽**；此处证明保留范围内的交集为零，不声称全球无偶遇或完全统计独立。

## 冻结命令及独立预算申请

协议 SHA256：`fdcbf9d555aa6f666fe1abe431a79058f93cd08874742a8fe62d2f587fd360c5`。输入/控制 seal SHA256：`2f008af1a404f68f449c998349a6089d7bf721f7fe0cf6e583b31af233d65c33`。完整 Docker argv、挂载、执行顺序及限时公式在归档 `input/execution-commands.json`，也完整写入机器回执；实际运行仍须追加原始 argv、退出/inspect 和计时事实。

候选设置固定为 CUDA、BF16/manual attention、batch=64、action_chunk=2048、workers=0、KV=false，CPU Torch 对手，单 GPU 串行。顺序为 S1→S2→S3，每 seed 比较 2k latest 与 20k latest，在 WP/ADP 各 2,000 副、两阵营，共 **12 组 / 48,000 局正式对战**。预检各 seed 4 副、两端点×两对手×两阵营重复两次，共 **192 局**，全部排除统计。

获准后唯一控制入口：

```sh
python3 /Users/kingson/GitHub/DouGPU/reports/b2-preaudit-20261010/control/controller.py
```

入口必须读到独立 `authorization.json`，其中 scope=B2_ONLY、authorized_seconds=1800，且 protocol/seal SHA 精确匹配本报告；**当前没有该授权文件**。controller 限定 LocalServer 对应独立目录，拒绝再次启动同一 B2 block。Docker 固定 image ID、runc、network=none、24g RAM、2g shm，政策/对手/deps/source 使用只读入口；启动前检查设备/驱动及 GPU 无其他计算进程，完整日志和 Docker inspect 保留。

申请 **1,800 秒全块硬上限**，从本地 controller 首次授权运行前计时，包含启动、哈希检查、预检、正式评估、退出、统计、封存及 Mac/LocalServer 读回。远端从启动计时，评估窗口截止在 +1500 秒，GNU timeout 的 TERM→90 秒强杀宽限及 10 秒清理从该窗口扣除；剩余 300 秒用于结案。Mac controller 采用更早起点且不延长远端截止。SSH 中断时远端 timeout 仍约束设备进程。

预检以最慢 warmed 8-game 组合×48000×1.5+60 秒预测；不适配剩余评估窗口即 STOP，不缩小正式规模。任何非零退出、超时、身份漂移、缺结果、重复不一致、封存/双位置读回失败均 TECHNICAL_FAIL / STOP；不自动重启、加第四 seed、加牌或追加预算。源码和 GPU 配置的实际运行能力仍要通过获准后的预检，不把本轮 CPU 成功算作 GPU 执行成功。

## 统计及封存

复用 `_summarize` / `paired_difference`，按完整副牌的两角色及两端点配对，2000 次 bootstrap、RNG=formal_seed+12345，给各 seed 名义 95% CI；CI 和 helper 的 promotion_lower_bounds 不参与方向筛查。各对手跨三 seed 等权报告均值、样本标准差 ddof=1、范围；训练 seed 总体 CI=NOT_ESTIMABLE，power=NOT_ESTIMATED。

仅六个 Balanced 点差全部 >0，且全部 12 个地主/农民团队点差 ≥−5pp，才报告 **DIRECTIONAL_SIGNAL_ONLY / STOP**；任一角色 <−5pp、Balanced 为零/负或方向混合则 INCONCLUSIVE_STOP。资格失败优先 TECHNICAL_FAIL / STOP。普通及 `-O` 合成自检均通过，覆盖缺组、零、混合、恰 −5pp、低于 −5pp、CI 跨零及配对身份错误。没有真实 B2 结果，没有晋升、G 或更多训练。

预审归档：`reports/b2-preaudit-20261010/preaudit-evidence.zip`，**25,082,244 字节**，SHA256 **`141840f61f2877797d7b1fdd9296a9293e092c9aaf42826811b93166b37a1bd6`**；manifest 覆盖 199 文件，含冻结源码/输入/政策/牌序、历史 inventory、一次性控制、失败/成功 CPU 回执和最小复核脚本。Mac 与 LocalServer 分别解压到新临时目录，验证外层 SHA、完整 manifest，并普通及 `-O` 运行统计合成自检，结果完全相同。两地读回回执在 ZIP 外，避免改写封存包。

归档存于本机及 `LocalServer:/var/tmp/dougpu-b2-preaudit-20261010/preaudit-evidence.zip`。reports 被 Git 忽略，没有新的公开下载链接；六原始对手及完整依赖沿用既有冻结存储，未重复打包。Git 发布范围为本报告、JSON 与索引；原始封存材料保留在两地存储。

结案：**零 GPU 预审通过，等待独立 B2 最多 1,800 秒明确批准；TRAINING_PAUSED / B2_NOT_ENTERED / G_NOT_ENTERED。**
