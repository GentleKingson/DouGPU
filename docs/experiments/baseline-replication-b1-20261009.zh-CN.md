# 新 Baseline B1：三 seed 训练与执行资格结案

日期：2026-10-09，Asia/Hong_Kong。结论：**B1_COMPLETE / EXECUTION_QUALIFIED / TRAINING_PAUSED；B2_NOT_ENTERED / G_NOT_ENTERED**。三条独立 seed 均完成 0→8→2000→20,000，九个预定端点全部 PASS；本轮共 60,000 次真实成功更新。**棋力结果 NOT_EVALUATED**，没有方向性复现、晋升或机制结论。

用户先授权“在 LocalServer 中训练”，随后明确批准“完整 B1，硬上限 10,800 秒，不执行 B2”。这次直接授权替代上一轮零 GPU 限制，仅限 B1；[原 B0 协议](baseline-replication-protocol-20261009.zh-CN.md)和[零 GPU 审计](b1-readiness-audit-20261009.zh-CN.md)保持历史原文。旧 Baseline/NTP INVALID_PROTOCOL、E/F ENGINEERING_CLOSED、R2 INCONCLUSIVE / STOP、Ataraxos NO_GO 均不变。

## 1. 冻结身份与实际设备

开始时 main 干净，HEAD 与远端 main 均为 `e4fd3d421f56c5f341fc34b5e30a27824c98dcd7`。用此提交的 Git archive 在 LocalServer 建立独立目录；不更新旧实验、不导入旧 checkpoint、不装依赖或构建镜像。训练源目录运行时只读，缓存独立挂载；事前和事后核对 135 个 tracked 文件，已记录的 14,522 个依赖文件字节无漂移。

| 身份 | 记录 |
|---|---|
| 完整执行意向 SHA256 | `ad151b84ad8e3a6646a110115744b595070128bf6620fc6ae6956cef271ad6ae` |
| main 源码 tar SHA256 | `43a9c8a580b144b13e43f9da75a82e4580f47620073345a2a024c8d629b53619` |
| 三链共同 trainer_source.zip SHA256 | `2956a5848b999f655a9ddf8320ec4a1fefb58365eb215bc0b073e16a4389e9fe` |
| 规则运行锁 SHA256 | `40f47b2cd94a76fe7112140d41b85af0b7462b6c35197241fe71fc506348e1ac` |
| Docker 镜像 ID | `sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e` |
| GPU / UUID / 驱动 | RTX 5070 / `GPU-46dce763-982c-cb97-2004-b7770ccdd7eb` / 595.91.07 |
| Python / JAX / NumPy | 3.12.14 / 0.7.2 / 2.5.3；配置 backend=cuda，实际 backend=gpu |
| CUDA / cuDNN 包 | jax-cuda13-plugin/pjrt 0.7.2，nvidia-cuda-runtime 13.0.96，nvidia-cudnn-cu13 9.18.1.3 |

准备三条空目录并分别冻结全配置及 source SHA；实际 UUID、worker seeds、恢复/输出 SHA、退出事实只在运行后填写。镜像及该设备/驱动组合通过三次真实 preflight 和九次训练，资格为 **VERIFIED_THIS_RUN_CONFIGURATION**。源码 tar 内没有 .git，原生 best-effort git 字段为 unknown；来源证明使用干净提交、外层文件清单和真实 runtime_source 校验，未把 unknown 当证明。

## 2. 执行顺序与硬上限

唯一配方保持 B0 的 9 个 ModelConfig 和 53 个 TrainConfig 字段：NTP=0.02、belief=0.05、64×4=batch256、CUDA/cuDNN/BF16、workers=8、history_groups=2、resume/save_replay=true。三段 max_hours 均为 1.0，没有用 `--hours` 临时覆盖；绝对 target 分别为 8/2000/20000。只共享编译缓存，参数、Replay、Adam、主 RNG、champion 均分别初始化。

执行沿用 run_local prepare/preflight/train、Store 与 execution gate；一次性控制及归档脚本在被忽略的 reports 中封存，不增加生产框架或依赖。GPU 容器固定 image ID、runc、`--gpus device=0`、network=none、24g RAM、2g shm；每次启动先检查 GPU 无其他计算进程。GNU timeout 的 TERM→90 秒强杀宽限包含在段预算内，预留归档/读回时间，容器退出后核对 State.Pid=0 并删除。CPU gate/readback 不请求设备、不初始化 JAX。

| 链 / train.seed / selection seed | 预检与 0→8（≤600 秒） | 8→2000（≤900 秒） | 2000→20k（≤2100 秒） |
|---|---:|---:|---:|
| C0-S1 / 2026101101 / 2026101201 | 246.636，PASS | 101.209，PASS | 618.003，PASS |
| C0-S2 / 2026101102 / 2026101202 | 19.518，PASS | 85.542，PASS | 603.630，PASS |
| C0-S3 / 2026101103 / 2026101203 | 19.506，PASS | 81.965，PASS | 596.826，PASS |

首块原控制器记录 56.853 秒，仅包含启动后的执行及读回，未包含 prepare 至启动间等待。结案按原始 setup-start 与首块结束时间重计为 **246.636 秒**，仍小于 600 秒；原始回执不改写。在首链末段运行期间追加全局预算计时补充，以最早 prepare 起点约束 10,800 秒，并留 10 秒清理余量，只收紧上限，不改配置或延长授权。以后复用监督命令时，应在首块启动前直接扣掉这段等待；本轮证明实际未超预算，不声称原首块控制器完整覆盖了此前等待。

三次 preflight 均 PASS：cuDNN 对 manual 的激活/全梯度数值检查、64/128/256/512 生产形状推理与越 chunk 动作选择、完整非 PAD 512-token 反向及 Adam 更新检查。每次 16 次合成更新，合计 48；这些状态丢弃，不计入 60,000 次 Baseline 更新。

GPU 阶段进程墙钟合计 **2144.240 秒**；这是容器进程包络，不是 GPU 利用率积分。最早 prepare→最后端点两地读回 **2372.844 秒（39.55 分钟）**；连最终整包封存及两地读回 **2846.969 秒（47.45 分钟）**，均未超过 10,800 秒。没有失败重启、第四会话或超时后继续。

## 3. 初始化、三段恢复及政策资格

Mac CPU 独立读回三份初始 checkpoint：resume_input=null、updates/Adam.step=0、Replay 空、Adam m/v 全零、champion 与初始参数相等、主 RNG 从对应 seed 开始。按各自 seed 复算 embedding，全部逐值匹配且三者不同；这证明指定初始化及无旧状态注入，不声称游戏样本绝无重叠。

每段在下段前核对实际 Store（含 mirror）候选与上一端点 SHA；保存独立 ZIP+marker 后复用 audit_execution，检查源快照/已加载模块路径、全配置/版本、Actor 顺序及实际 seeds、输入 checkpoint RNG、连续日志、准确更新边界、Adam、完整有限 FP32 状态、Replay 和 sample credit。九个端点均在 LocalServer 与 Mac 独立读回 PASS；每链均记录 0 个 nonfinite step、退出码 0、未超时、镜像保存成功。

| 链 | 2k checkpoint SHA256 | 20k checkpoint SHA256 |
|---|---|---|
| C0-S1 | `54346127d66d8686174f7ed22ce072ed605c353f35a31f3f92ecd5e474d3d52b` | `356b9c66cfb5e1dc64c1bd98e74bbbcf38a99668af032c004c3f7ed60a4f7faa` |
| C0-S2 | `128343d3d6675d75737948de21c919098dba1e2599cc186bb1011da09c20254b` | `fc5e44530d90495cffc508395e077ae702fa633532d83e5dd20f1789234d49b0` |
| C0-S3 | `b588473b824635ecacc5df335dc354fbe04ef75a87b04e0de5a9f1f204ea8d2a` | `cbf45197f9d4c100f5c8e5a6c382fb047b6663c35a73ececc117169be8b3c104` |

各链冻结 2k/20k latest 和 best，但下一阶段候选仅为 latest。逐参数核对 latest 与对应完整 checkpoint 的部署参数相等（政策按既有格式省略 NTP/belief 辅助头），并验证每链 2k/20k 的真实政策参数不同，不仅比较含元数据的文件 SHA。初始/8-update SHA、真实 UUID、恢复输入、六个政策 SHA、Replay 角色计数和全部包版本见 [机器回执](baseline-replication-b1-20261009.json)，完整状态在证据包。

## 4. 封存与可运行复核

归档 `reports/b1-training-20261009-e4fd3d4.zip`：**401,337,416 字节**，SHA256 **`4d86e41e0a178f458ae3580f0d53f69453f91141850455d50ab795b28acf1480`**。manifest 覆盖 161 个文件，包含源 tar/清单、完整意向、事后计时补充、一次性命令脚本、九份逐段 evidence tar、原始 GPU/CPU 日志及退出回执、完整三段端点和初始化状态、政策、两地逐段读回与结案事实。

证据包保留封存时的中间 block 状态；最终状态由 `server-receipts/C0-S*/block-*.json` 与各 `complete-block-*.json`、`progress.json` 联合给出，不能只拿 archive 内某一份 RUNNING 回执结案。最后 GPU 计算进程及任务命名容器均为空，全局 watchdog 收到完成标记后退出。

Mac 和 LocalServer 分别从同一封存 ZIP 解压到新临时目录，校验外层 SHA、161 文件清单、135 原提交源码文件，对三链分别普通及 `python -O` 验读，**六次/地均 PASS，结果完全一致**。Mac NumPy=2.3.5，服务器=2.5.3；仅记录此次一致性。最终读回回执留在 ZIP 外，避免改写封存包：`b1-training-20261009-e4fd3d4-archive-readback-{mac,server}.json`。

最低成本复核使用已有 NumPy 的 Python，仅读 ZIP/NPZ、不调用训练或评估入口：

```sh
python reports/b1-training-20261009-e4fd3d4/readback-archive.py \
  reports/b1-training-20261009-e4fd3d4.zip \
  4d86e41e0a178f458ae3580f0d53f69453f91141850455d50ab795b28acf1480 \
  /tmp/b1-independent-readback.json
```

该脚本从归档恢复原源码，内部运行普通及优化解释器的 `endpoint.py verify`。初始化检查另有 `verify-initial-seeds.py` 的普通模式 assert 自检；预算命令自检为 `python control/stage.py check`。正式运行 argv、超时和 exit/inspect 事实均在原始回执，不重跑 stage，不使用 `check_session_chain.py --train`。生产源码未修改，本轮 Git diff 仅本报告、JSON 与实验索引；前轮 22 项回归不冒充本轮重跑。

ZIP 保留在 Mac 与 LocalServer 两个独立存储位置；reports 被 Git 忽略，没有新公开下载链接。完整 Docker image、挂载依赖及旧 WP/ADP 原始权重不重复打包，复算 GPU 执行须另取这些已冻结输入；上述 CPU 证据读回不需要 GPU 或 Torch。

## 5. 下一阶段边界

**可以申请 B2 资格准备及评估审批，当前 B2 授权仍为 0。** B1 端点具备执行资格，尚未生成或验证正式有序 deals、预检/selection/可取得历史的真实交集，尚未冻结 B2 全部实际命令、政策/对手/规则/牌序/角色映射与执行环境身份。B0 候选 B2 1800 秒需独立明确批准，不借用 B1 余量。

WP/ADP 沿用前轮 CPU 资格通过的用户固定字节，其官方认证继续 **NOT_VERIFIED**；若研究要求官方认证，仍 NO_GO。B2 才能评估三条链 20k latest 相对 2k latest 的方向，遵守原描述性判据，不提供训练 seed 总体显著性结论。

结案：**B1 执行 PASS；TRAINING_PAUSED；B2_NOT_ENTERED / G_NOT_ENTERED，无自动续训、评估或晋升。**
