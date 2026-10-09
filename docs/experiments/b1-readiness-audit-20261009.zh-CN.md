# B1_READINESS_AUDIT：零 GPU 结案

日期：2026-10-09，Asia/Hong_Kong。结论：**B1_READINESS_AUDIT_PASS_WITH_BLOCKERS / TRAINING_PAUSED**。B1、B2、G 均 NOT_ENTERED；获准及实际 GPU 预算均为 **0 秒**，本轮真实训练更新、新牌局和正式评估均为 0。

用户授权本轮 P1→P5 审计；[B0 协议](baseline-replication-protocol-20261009.zh-CN.md)及附件中的未来执行指令是审阅材料，不构成训练批准。本文补记本轮新事实，不改写 B0 的历史状态。旧 Baseline/NTP 的 INVALID_PROTOCOL、E/F ENGINEERING_CLOSED、R2 INCONCLUSIVE / STOP 和 Ataraxos NO_GO 均不变。

开始时工作区干净，HEAD、origin/main、远端 main 均为 `2553aa354be36b2831a984a76668e8b45dcaea9e`。事前身份按此提交冻结；之后仅在 P4 加入已记录的最小有限值检查和测试，不把修改后的身份冒充事前原值。完整机器回执见 [JSON](b1-readiness-audit-20261009.json)。

## 1. 核对的实际调用路径

| 源码 | 实际路径及审计约束 |
|---|---|
| [run_local.py](../../run_local.py) | prepare 展开配置、规则锁、源码快照；运行经目录锁、运行时配置和 GPU 锁进入 invoke。invoke 新建 learner session，SIGINT/SIGTERM 只转发 learner PID，随后等待；没有外部硬期限或全进程树强杀保证。`--hours` 覆盖会话实值，`--cycles` 为追加 cycle，不是成功更新数。 |
| [compose.yaml](../../compose.yaml) | 当前服务申请一张 GPU，镜像为可变标签 `dougpu:local`，挂载可写 /app，停止宽限 120 秒。仅解析配置，未启动服务；不能直接作为零 GPU 验收命令，也不能把其宽限默认值计作协议内 90 秒。 |
| [train.py](../../dougpu/train.py) | main 才导入 JAX；恢复 Store、完整状态及主 RNG，Actor 启动记录实际 seeds。成功更新到绝对 target 才 COMPLETE。finally 先关闭 Actor，再写 session_end、保存最终 ZIP；日志 COMPLETE 不证明最后提交或镜像成功。内部 max_hours 不覆盖启动、预检、最终归档。 |
| [checkpoint.py](../../dougpu/checkpoint.py) | ZIP 内 manifest、外部 SHA 和最后写入 marker 验证代际。NPZ 仅为政策导出；load_latest 可回退旧有效代际或选镜像更新代际，须核对实际恢复 SHA。keep 滚动剪枝不能替代独立端点归档。 |
| [protocol_gate.py](../../scripts/protocol_gate.py) | 本轮复用 audit_execution/execution_sessions，核对完整配置、版本、源码、Actor、三段恢复输入、端点、Adam.step、Replay 和 RNG。旧 audit/compare 是历史 NTP 路径。执行门禁本身不核对外部超时/退出事实，也不授予实验预算。 |
| [check_session_chain.py](../../scripts/check_session_chain.py) | check 将事前 intent SHA/字段与事后 plan 串联，再审独立端点和负例。只调用离线 check；未使用 `--train`，未调用 train_chain。 |
| [evaluate.py](../../dougpu/evaluate.py) | CLI 经政策读取、selection seed 检查、JAX/Inference 后才评估，故本轮不调用 CLI。直接调用共享 DouZeroOpponent，复用安全 CPU load 和 strict 模型接口，补全参数有限值检查；未做 forward。 |
| [evaluation.py](../../dougpu/evaluation.py) | 仅调用 _summarize/paired_difference/validate_holdout_seed。按整副两列 outcome 配对 bootstrap；调用方仍须验证对手、规则、政策和有序 deal 身份，helper 的 seed/shape 一致不证明完整资格。未调用 paired_evaluate 或构造新牌局。 |

## 2. P1：事前身份冻结 — PASS_KNOWN_IDENTITIES

`frozen-audit-intent.json` SHA256：`721b767200c12ee933c76399af99cad13311a6d46b22635faa1ea0b662e26f20`。冻结全部 133 个 tracked 文件、9 个 ModelConfig/53 个 TrainConfig 字段、三 seed 的三阶段候选实值、规则锁、六权重 SHA、镜像身份、零 GPU 授权和审计范围。候选配置不是获准训练命令。

- 训练 seed：2026101101/1102/1103；selection：2026101201/1202/1203；预检：2026101301/1302/1303；正式：2026101401/1402/1403。每链候选边界 0→8→2000→20000。
- 规则运行锁 SHA：`40f47b2cd94a76fe7112140d41b85af0b7462b6c35197241fe71fc506348e1ac`；DouZero commit `718a5c920bf3361e34178a38f3b80458e176b351`，encoding=1。缓存锁的额外字段身份与运行锁分开。
- 服务器现有镜像 ID：`sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e`。Docker 29.8.2；CPU Python 3.12.14、NumPy 2.5.3、Torch 2.8.0+cpu；JAX/JAXLIB 0.7.2 仅读包元数据，不导入。
- 未产生的真实 UUID、worker seeds、输入/输出端点、政策和 deal SHA 留待真实回执。合成夹具另有 SYNTHETIC scope 和意向，不混入新 Baseline。

Mac Docker daemon 不可用，使用 LocalServer 已有 Docker、只读 CPU `/deps` 和固定镜像，不构建、不拉镜像、不装依赖。CPU 容器启动已验证；**GPU 设备、驱动兼容性及该镜像 GPU 运行能力均 NOT_VERIFIED**。

## 3. P2：外部超时与进程树清理 — PASS_SIMULATION_ONLY

复用 Docker 隔离及真实 `run_local.invoke`，learner/actor 均为只记录 PID 和信号的 Python dummy；部分 actor 新建 session，验证不能仅靠 launcher 进程组清理。每例硬上限 10 秒，2 秒宽限包含在上限内；预算耗尽在启动前拒绝。

| 场景 | 秒数 | 容器退出码 | 超时事实 | 验收 |
|---|---:|---:|---|---|
| 正常退出 | 0.562 | 0 | false | PASS |
| 预检挂起 | 9.087 | 137 | true | PASS，负例 TIMEOUT_FAIL |
| learner 挂起、actor 脱离 session | 9.109 | 137 | true | PASS，负例 TIMEOUT_FAIL |
| learner 退出、孤儿 actor | 0.592 | 0 | false | PASS，无残留 |
| 收到信号后写伪 COMPLETE | 7.114 | 0 | true | PASS，仍为 TIMEOUT_FAIL |

回执保留 Docker before/after inspect、host PID 与容器 PID 记录：无 DeviceRequests/Devices、runc、network=none、无 NVIDIA 节点，退出后 State.Pid=0，原 host 进程树消失，审计命名容器全部删除。`p2-frozen-controls.json` SHA 为 `c296c02d26a8abada90572f29be824c3d5a873efd94c392caeaf83e2d9e49414`。

这是正常 Docker/SSH 条件下的方法验收。真实各段 prepare/preflight/train/archive/readback 命令及总预算尚未冻结，远端故障和真实硬件运行仍未验证；不能把 10 秒模拟扩大解释为完整生产监督已交付。

## 4. P3：意向及三段恢复离线负例 — PASS_OFFLINE_ONLY

用现有 Store/Replay 和参数形状规则写零数组、空 Replay 的 **SYNTHETIC** 三段夹具，标称 0→8→2000→20000；不初始化真实模型/优化器、不运行 learner。keep=1 后独立端点仍能验读，证明检查不依赖滚动目录保留中间代际。合成 intent SHA 为 `1384afab2efeac6985f664fabdff377808d46eef02f9c6d7e3e9a990074b0939`。

合成链和原有真实 CPU E/F 0→1→2 归档均通过离线正例；两者各 9 项负例全部拒绝：缺中段结束、不完整端点、伪 SHA、缺 ZIP、恢复边界错误、marker 损坏、Actor seed 错误、源码损坏、JAX 版本变化。另 6 项拒绝 intent 哈希/数量/配置错误、Adam.step 不符、缺 Replay、Store 回退至错误旧端点。缺 Replay 当前以异常拒绝，不声称诊断文本友好。

`tests/test_protocol_gate.py` 的 8 个检查通过；夹具扩展至完整三段。Mac 与服务器普通/优化解释器均复算一致，包含各链 9 项负例；这是独立 CPU 读回，不是新增真实训练。旧 CPU 归档没有补成新的 GPU Baseline。

## 5. P4：对手输入与合成统计 — PASS_CPU_INPUTS_AND_SYNTHETIC_STATISTICS

六份 WP/ADP 原始权重 SHA 精确匹配 B0/来源清单；实际 Torch CPU `weights_only=True`、`strict=True` 加载全部角色，检查所有参数有限且在 CPU，加载前后源文件 SHA 不变。缺角色、哈希不符、错误角色、缺键、NaN、+Inf、−Inf 共 7 项负例拒绝。官方发布认证仍 **NOT_VERIFIED**，资格只适用于本组用户提供的固定字节。

最小生产改动：DouZeroOpponent 安全 strict load 后增加 **2 行**有限值检查，统一覆盖 evaluate CLI、旧独立评估调用方和测试。无新依赖、框架或生产监督器。测试加入三类 nonfinite 拒绝；恢复夹具扩展至三段。

合成两列 binary outcome 验收复用配对统计：同副角色抵消时 Balanced CI=[0.5,0.5]；同政策差值为 0；缺 seed 为 TECHNICAL_FAIL，混合方向/零差为 INCONCLUSIVE_STOP；角色恰 −5pp 可通过方向筛查，低于 −5pp 必停。点差 +2.5pp、CI=[−5pp,+10pp] 的例子仍为 DIRECTIONAL_SIGNAL_ONLY，验证 CI 是描述性结果，不作晋升或总体显著性门槛。跨 seed 等权与按局数池化区分；训练 seed 总体 CI 不估计。

错误 seed、deals、shape、非 binary、NaN、Inf 及 selection seed 重用共 7 项拒绝。未生成正式 deal，也未用已有 helper 返回的 promotion_lower_bounds 改写 B0 判据。Mac NumPy 2.3.5 与服务器 2.5.3 的指定配对例子一致，不推断一般版本等价。

第一次完整相关回归为 **1 FAIL / 21 PASS**：只读 /app 没有运行时需要的可写 `.cache`。保留失败回执；仅加 `--tmpfs /app/.cache`，不改源码或测试，重跑同组 **22 PASS（1.51 秒）**。

## 6. 验证命令与复算入口

完整 argv、stdout/stderr、退出码及阶段事前控制分别保存在 `p2.json`、`p3-command.json`、`p3-process.json`、`p4-command.json`、`p4-process-1.json`、`p4-regression*.json`；不是仅列预期命令。P2 服务器执行 `python p2-simulate.py`，它只启动命名 dummy 容器。

P3/P4 复用以下 CPU 隔离前缀；按归档 RESTORE.txt 恢复 SOURCE/CONTROL/OUTPUT、旧链和原六权重，只使用已有镜像与依赖。正式复算应另建 output，保留原回执。以下不含 GPU 请求，也不调用训练/评估入口：

```sh
AUDIT_ROOT=/var/tmp/dougpu-b1-readiness-20261009-2553aa3
CPU_DEPS=/root/dougpu-independent-strength-20261008/deps
CPU_WEIGHTS=/root/dougpu-independent-strength-20261008/input/inputs
CPU_IMAGE=sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e
docker run --rm --init --runtime runc --network none --cpus 4 --memory 4g \
  --read-only --tmpfs /tmp --tmpfs /app/.cache \
  -e JAX_PLATFORMS=cpu -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app:/app/vendor:/deps \
  -v "$AUDIT_ROOT/source":/app:ro -v "$AUDIT_ROOT/control":/control:ro \
  -v "$AUDIT_ROOT/output":/output -v "$CPU_DEPS":/deps:ro \
  -v "$AUDIT_ROOT/existing-cpu-chain":/oldchain:ro -v "$CPU_WEIGHTS":/weights:ro \
  --entrypoint timeout "$CPU_IMAGE" --signal=TERM --kill-after=2s 115s \
  python -m pytest -q -p no:cacheprovider /app/tests/test_official_opponent_import.py \
  /app/tests/test_model_selection.py /app/tests/test_protocol_gate.py
```

P3 将尾部 pytest 命令替换为 `python /control/p3-offline.py`；P4 为 `python /control/p4-inputs-statistics.py`。原命令由 Mac 以 120 秒 subprocess 超时监督。离线端点复算入口是 `python scripts/check_session_chain.py <恢复后的链目录>`，并重复 `python -O`；**不得加 `--train`**。

## 7. P5：封存、读回和下一阶段

本地归档：`reports/b1-readiness-audit-20261009.zip`，1,759,552 字节，SHA256 **`844dbe822e9e4832d89c70ab470795ee57e9c620f2e56e82ef6dec0d7d5497b4`**。包含原提交源码包、测试过的三文件覆盖层/patch、事前意向、阶段脚本与控制、原始失败/成功回执、合成链、旧 CPU 链和恢复说明；manifest 覆盖 77 个文件。

Mac 和 LocalServer 分别从封存 ZIP 解压到全新临时目录，验证外层 SHA、全部 77 文件、恢复源码覆盖层，并用普通/`-O` 模式重新验读两条链，各 9 项负例全部拒绝。外置 `b1-readiness-audit-20261009-archive-readback-{mac,server}.json` 保存最终封存读回结果，避免改写已封存包；两地匹配。最后 Docker 命名容器清单为空。P5 **PASS_ARCHIVED_READBACK**。

`reports/` 被 Git 忽略，证据 ZIP 目前保留在本机和 LocalServer，不随克隆下载、没有新建公开下载地址。完整镜像、挂载依赖及六原始权重不重复放入小归档；复算这些输入须取既有独立棋力证据包，身份见 B0。源码/测试及本报告、JSON、实验索引作为本轮 Git 发布范围；封存回执的“未提交”描述指封存时事实。

**可申请下一阶段受限真实设备 preflight 审批，但不能据本轮直接批准完整 B1/B2 训练。** 必须先审阅并冻结发布后的干净 commit、trainer_source.zip、真实 preflight 命令及其独立预算，再明确批准设备运行；CPU 模拟不代替真实验收。完整训练申请还被以下事项阻断：

1. GPU 设备、驱动/运行时、镜像 GPU 内核能力 NOT_VERIFIED；禁止隐式 CPU 回退。
2. 最终三段外部监督、归档/双位置读回命令及其时间分配未冻结；所有宽限和开销必须纳入硬上限。
3. 真实端点、政策导出关系、worker seeds、退出/超时/镜像回执尚不存在；按实际事实追加，不能预填为 PASS。
4. 实际预检/正式/selection 有序 deal 身份及可取得历史交集尚未生成、核对；不能只按整数 seed 声称无重叠。
5. 用户提供 WP/ADP 的官方认证 NOT_VERIFIED；若目标要求官方认证，则 NO_GO。
6. 预算批准尚无。B0 的 B1=10,800 秒、B2=1,800 秒、合计 12,600 秒只是候选，不是本轮授权。

结案后 **TRAINING_PAUSED，授权 GPU=0；不自动进入 B1/B2、不补第四 seed、不改旧实验结论**。
