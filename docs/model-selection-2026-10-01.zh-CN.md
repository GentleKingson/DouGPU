# DouGPU 角色评估与冠军选择验证报告

“未执行”仅指本阶段结束时的协议状态，后续执行结果见下列历史对手报告；最新预算决定与原始证据获取见[证据说明](experiment-evidence.zh-CN.md)。

项目名称已统一为 DouGPU。下文保留验证当时的包名、命令、路径、镜像与哈希，均为历史证据；当前入口见 [README](../README.md)，后续阶段结果见 [历史对手报告](historical-opponents-2026-10-01.zh-CN.md)。

本阶段补齐了角色置信区间，为冠军晋升增加角色回退检查，并将默认评估策略改为 best。学习率、模型、奖励、辅助 loss、探索和采样更新比例均未修改。

功能回归和完整模型短训通过。训练中的 20 次验证检查产生了 1 次晋升、19 次拒绝；按新门槛选出的 best 保留在第 275 周期，未被第 500 周期的 latest 替换。

截至本阶段，三 seed 的历史对手对照实验、一小时稳定性与恢复验证，以及检查通过后的训练预算扩展均未完成。这次功能回归尚不能说明新一轮棋力或长期稳定性。

## 晋升条件

RTX 5070 balanced/conservative 配置每 25 周期使用 256 对验证牌局。每对牌局交换地主和农民阵营，以发牌对为 bootstrap 单位，记录地主、农民团队、均衡胜率及各自名义 95% 区间，同时保存逐对结果。

一次晋升须同时满足：

1. 至少 128 对验证牌局，候选对冻结冠军的均衡胜率下界大于 50%。
2. 候选对冠军的地主、农民团队胜率下界分别不低于 45%。
3. 对固定规则对手，候选相对当前冠军的地主、农民团队胜率变化下界分别不低于 -5 个百分点。

第三项使用同一批规则发牌的逐对差值，不是把两个独立胜率区间相减。冻结冠军的规则结果只计算一次；晋升后复用已经测得的候选结果，避免重复评估同一策略。

`promotion_role_margin=0.05` 是本轮实验之前设置的绝对非劣性容差。五项统计检查使用单侧 99% percentile bootstrap 下界，按每次检查的 Bonferroni 分配控制名义显著性水平。该方法是有限重采样近似，不构成反复挑选模型后的整体错误率保证，也不保证对任意对手、任意牌局均无退步。容差相对当前冠军，不是全程累计回退上限。

牌局不足时仍输出评估，但不会晋升；验证途中收到停止请求、未完成全部检查时也一样。实验期间未放宽晋升门槛。

## 模型选择与留出

`run_local.py evaluate` 默认选择 `best_policy.npz`，latest 必须显式指定。best 在首次晋升前仍是初始策略，不能只凭文件名认定已达到部署棋力要求。

完整状态和两个新策略导出记录历史 `selection_seeds`。普通续训与 checkpoint 导入均保留历史 seed；更换内部验证 seed 不会抹掉旧记录。`run_local.py evaluate` 和直接 `python -m doutpu.evaluate` 两个入口拒绝把已记录的选择 seed 当作最终留出 seed。旧策略导出没有这些信息，结果会标记 `selection_seeds_known=false`，需要人工核对其实验记录。

这能防止重复使用已知内部选择 seed，不能阻止人在看到最终测试成绩后手动调整超参数。最终留出集只能在模型与方案选择冻结后使用，不能再按留出分数反选 checkpoint。

## 回归验证

通过 `ssh LocalServer` 执行。先检查 `/opt/DouZero` 的工作区，再将本地源码复制到独立的 `/tmp/dougpu-selection-20261001`；不覆盖服务器原仓库。

所有依赖安装、CPU/GPU 回归和训练均在 Docker 中运行。标准镜像及其依赖未变，也未重新构建；验证使用 bind mount 提供的当前源码，未使用镜像内的旧源码。

```text
image: douzero-test:latest
image ID: sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e
source SHA256: 852490f5aeb92a2e6c22a0572f07f35e94869b4a9cd88693472f1ea7daaf98e0
```

| 检查 | 本轮结果 |
|---|---|
| CPU 完整回归 | 212 passed，1 skipped，46.88 秒 |
| GPU 完整回归 | 212 passed，1 skipped，145.17 秒 |
| 全模型训练 | 500 cycles、2,000 successful updates，111.15 秒；20 次验证检查、1 次晋升 |
| 状态与导出 | Adam step=2,000，参数/冠军/Adam 有限；best 导出逐数组等于保存的冠军 |
| GPU 检测 | 训练报告 backend=gpu、devices=[cuda:0]；RTX 5070，12,227 MiB，驱动 595.71.05 |
| 旧农民回退结果复算 | 新规则角色检查拒绝，见下一节 |

新增测试覆盖分角色区间、同牌局配对差值、均衡改善但单角色退步时拒绝晋升、固定规则对手退步时拒绝、满足条件时通过、样本量不足、非法容差、seed 隔离、默认 best 和导入时历史 seed 保留。

CPU 跳过真实 CUDA cuDNN 数值检查；GPU 跳过要求 CPU 后端的 cuDNN 拒绝检查，均为预期。训练数值回归未删除或放宽。短训完成 21,146 局、1,024,054 条完整轨迹样本，非有限更新为零。第 275 周期晋升后，第 500 周期的规则验证分数为地主 21.48%、农民 42.58%、均衡 32.03%；保存的冠军为地主 26.56%、农民 37.11%、均衡 31.84%。候选的均衡分数稍高，但地主分数较低。这些是模型选择所用的验证分数，不能作为独立棋力测试结果。

复现命令如下。`verify.py cpu/gpu` 在临时容器内安装 `requirements-dev.txt`，随后执行 `python -m pytest -q tests --junitxml ...`；没有向宿主安装依赖。

```bash
ssh LocalServer 'docker run --rm --name dougpu-selection-cpu-20261001 \
  --mount type=bind,src=/tmp/dougpu-selection-20261001,dst=/app \
  --mount type=bind,src=/tmp/dougpu-selection-20261001/evidence,dst=/evidence \
  --entrypoint python douzero-test:latest /evidence/verify.py cpu'

ssh LocalServer 'docker run --rm --gpus all --name dougpu-selection-gpu-20261001 \
  --mount type=bind,src=/tmp/dougpu-selection-20261001,dst=/app \
  --mount type=bind,src=/tmp/dougpu-selection-20261001/evidence,dst=/evidence \
  --entrypoint python douzero-test:latest /evidence/verify.py gpu'

ssh LocalServer 'docker run --rm --gpus all --name dougpu-selection-smoke-20261001 \
  --mount type=bind,src=/tmp/dougpu-selection-20261001,dst=/app \
  --mount type=bind,src=/tmp/dougpu-selection-20261001/evidence,dst=/evidence \
  --mount type=bind,src=/tmp/dougpu-selection-20261001/runs,dst=/runs \
  --entrypoint python douzero-test:latest /evidence/verify.py smoke'
```

初次传输附带 macOS `._*` 元文件，干扰了来源哈希，但没有改变实际 Python 源码。发现后清除本任务的元文件，重新传输不含这些文件的源码，再重跑完整回归与训练。验证驱动现在启动时先核对本地预期源码哈希。早期记录单独放在 `transfer-metadata/`，不作为最终来源一致的验收记录。验证驱动的直接导入路径也已修正；这不是训练代码的运行错误。

## 旧回退的复算

只读取上一份报告中 seed 42 的 10,000-update 和 52,616-update 规则对手原始结果，发牌 seed=951102，1,000 对牌局。没有重新训练，也没有用这些旧留出牌局挑选本轮模型。

| 角色 | 长训相对 10,000-update 变化 | 本轮复算 95% 区间 | 晋升检查单侧下界 |
|---|---:|---|---:|
| 地主 | +3.0 个百分点 | [0.2, 5.7] | -0.401 个百分点 |
| 农民团队 | -8.3 个百分点 | [-11.6, -4.8975] | -12.2 个百分点 |
| 均衡 | -2.65 个百分点 | [-4.75125, -0.45] | -5.2 个百分点 |

农民下界 -12.2 低于 -5，规则对手角色条件因此不通过。这验证了该检查能拒绝已知的退步情形。两个旧策略之间未补跑直接对战，完整晋升门槛仍未在这组旧模型上验证。本轮重采样 2,000 次，上一份离线报告为 5,000 次，因此区间略有变化。

## 后续实验

机器可读协议已保存为 `reports/model-selection-20261001/experiment-plan.json`，状态为未执行。两组使用同样的 training seeds=42/43/44，在 0、2,000、10,000、20,000 次成功更新保存相同评估节点。

基线继续当前策略自博弈。另一组将 50% 的完整牌局改为对历史冻结策略，当前策略以相同概率担任地主或农民团队；每局固定对手和阵营，只将当前策略动作写入 replay。历史池事先固定为上一轮三个 seed 的 10,000-update 导出，记录 SHA256，不按新测试成绩挑选对手，也不在线更换冻结参数。

内部冠军 seed=900001。额外方案选择验证使用 960001/960002；最终规则留出 963001/963002、DouZero 留出 964001/964002 均与它们分开，须等两组模型和基于验证的方案选择冻结后才首次生成结果。

之后对选定方案运行 3,600 秒，逐 PID 跟踪 PSS、匿名内存、cgroup anon/file、GPU 显存、日志大小、检查点耗时和编译形状，并重复正常退出、SIGTERM、真实 periodic-learning 状态恢复。只有这些检查通过，才扩展到每个 seed 累计 40,000 updates，并使用另一套全新留出 seed 验收。协议中的这些预算是下一步执行方向，不是已完成的结果。

## 证据与清理

本地证据目录：`reports/model-selection-20261001/`。`verify.py` 是容器内复现驱动，`source.tar.gz` 是实际源码快照；测试 XML、完整日志、配置、逐牌局结果与训练状态保留在归档中。

归档下载后的 SHA256 为 `1bc9d12e68912bf43aea34eb3819a2eacffb124e1467013222c402a7717ee8dd`，与服务器一致；本地 `shasum -a 256 -c evidence/SHA256SUMS --quiet` 校验全部 47 项通过。完整状态、配置和日志已留在本地。硬件查询结果另存于 `hardware.log`。

服务器本轮 `/tmp/dougpu-selection-20261001` 已清理，所有本轮容器使用 `--rm`，最终没有运行中的容器；本地传输用临时 tar 也已清理。标准镜像与构建缓存保留，没有 prune、宿主依赖安装、驱动更改或服务重启。服务器原仓库仍为 detached HEAD，17 项 tracked 修改和 6 个 untracked 路径与本轮前相同；tracked binary diff SHA256 仍为 `5b685e4a4011e5b580b088e82f9efa7d11697aff8ae8903a274c097f29d19da0`。

下一阶段按已登记协议执行历史对手对照实验。
