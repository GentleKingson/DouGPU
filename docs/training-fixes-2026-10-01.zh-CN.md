# DouGPU LocalServer 训练修复与持续验证报告

本报告的 seed 连续运行说明仅适用于本轮短跑，不代表后续六小时轨迹无重启。后续预算决定与原始证据获取见[证据说明](experiment-evidence.zh-CN.md)。

项目名称已统一为 DouGPU。下文保留验证当时的包名、路径、镜像与哈希，均为历史证据，不代表当前项目名或最新源码；当前入口见 [README](../README.md)。

本次通过 SSH 在 LocalServer 的 RTX 5070 上完成了回归修复、旧检查点续训、三个训练 seed 的留出评估，以及半小时连续运行和恢复测试。

CPU/GPU 回归均通过。默认训练路径完成了学习，三个 seed 的地主与农民策略均显著优于各自初始策略；此前地主表现下降的旧检查点在续训后也有恢复。

此次工程稳定性检查通过，但棋力没有随训练单调提高：长训后的农民策略对规则对手出现回落。现有结果不足以证明达到 DouZero 水平，也未验证 12 小时或数天运行的稳定性。

验证日期：2026-10-01，Asia/Hong_Kong。核心 GPU 验证约于 01:30 完成，随后进行归档与清理。此前的 9 月 30 日报告保留为历史记录，本报告补充并更新其中的失败项与单 seed 限制。

## 验证对象

| 项目 | 本次实际设置 |
|---|---|
| 连接 | `ssh LocalServer` |
| 本地仓库 | 本地项目工作区（当时目录名为 `DouZero`） |
| 服务器隔离目录 | `/tmp/dougpu-goal-20260930`，归档后清理 |
| 服务器原仓库 | `/opt/DouZero`，未写入其源码或实验目录 |
| GPU | RTX 5070，12,227 MiB，驱动 595.71.05 |
| CPU / RAM | Intel Core Ultra 5 245KF，14 CPU，约 30 GiB |
| 运行环境 | Docker，Python 3.12.14，JAX/JAXLIB/CUDA plugin/PJRT 0.7.2 |
| 模型 | 922,401 参数，3 层，width=128，BF16，512-token 上限 |
| 默认训练 | manual attention，8 workers × 8 局，micro_batch=32 × accumulation=8 |
| 学习设置 | 原 balanced 配置的目标、学习率、探索、replay 与更新比例均保持不变 |

修复后的训练源码哈希：

```text
6131eb957ea69d6277b7c0e58fed7998aa0d2c243f0b92e78cf91741082efefc
```

规则仍固定到 DouZero 提交 `718a5c920bf3361e34178a38f3b80458e176b351`，没有换用简化测试引擎。

## 回归修复

修复保留了所有失败测试，并执行了真实 GPU cuDNN 检查。

| 问题 | 定位与处理 |
|---|---|
| FP32 的 XLA/manual、KV、历史分组、PAD 梯度或 Adam 更新不一致 | 默认 GPU 矩阵乘法精度造成误差放大。FP32 投影与 attention 显式使用完整精度；BF16 manual 路径保持原计算方式。修复位于 `model.py` 与 `kv_inference.py`。 |
| 不同 action chunk 的分数要求逐位相等 | 不同 GEMM 形状存在极小舍入差异。分数改为紧数值比较，选择结果、探索 RNG、真实相等时的首索引、非有限值检查仍严格保留。新增相邻 FP32 值检查，确保没有用“近似相等”吞掉真正较优的动作。 |
| preflight 的“完全相等动作”并未产生完全相等分数 | 随机投影的行舍入可不同。仅在合成 tie 检查中将 Q 输出投影清零，构造真正相等的分数；没有给生产 argmax 添加容差。 |
| GPU 套件内的 resume 测试声明 CPU，但 JAX 已初始化 GPU | 测试配置改为实际后端。生产代码仍拒绝不匹配后端，不允许 CPU fallback。 |
| cuDNN BF16 前向报 `No valid execution plans built` | 调试日志进一步定位到 `NVRTC` 运行时编译失败。只把 `nvidia-cuda-nvrtc` 从 13.0.88 升到 13.1.115，保持其他 CUDA 组件版本不变，问题消失。已更新依赖约束和 wheel 哈希记录。 |

JAX 文档说明，FP32 输入并不意味着默认 GPU dot 内部一定使用完整 FP32 精度；本次精度修复与该行为一致。[JAX Precision](https://docs.jax.dev/en/latest/201/precision.html)

cuDNN 的版本支持矩阵包含 CC 12.0，本次运行失败不足以判定 RTX 5070 不支持 cuDNN attention。修复依据是本机的版本对照实验；由于缺少更详细的编译器诊断，尚无法确定具体失败指令。[NVIDIA cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.18.1/reference/support-matrix.html)

## 完整回归与预检

| 检查 | 结果 |
|---|---|
| GPU 完整 pytest | **202 passed，1 skipped，0 failed**，145.61 秒 |
| CPU 完整 pytest | **202 passed，1 skipped，0 failed**，46.75 秒 |
| 两个 skip | GPU 上跳过“CPU 必须拒绝 cuDNN”的检查；CPU 上跳过真实 CUDA cuDNN 数值检查，均为预期后端条件 |
| NVRTC 版本对照 | 13.0.88 失败；13.1.115 的有/无 padding 前向、反向均通过 |
| manual 完整模型 preflight | 通过，64/128/256/512 全长非 PAD 训练批次，共 16 次有效更新 |
| cuDNN 完整模型 preflight | 通过，同样覆盖四档长度与完整 512-token backward，共 16 次有效更新 |
| 推理边界 | 第 2,049 个合法动作可被选中，跨 chunk 的真正相等分数选首索引 |
| 依赖一致性 | `pip check` 通过，保留实际 `pip freeze` |

完整回归在原标准镜像中临时安装修正的 NVRTC 与开发依赖后运行；后续训练使用由同一原镜像更新当前固定依赖所得的新镜像。没有使用全局 `JAX_DEFAULT_MATMUL_PRECISION` 环境变量掩盖问题。

这验证了上述配置与测试覆盖，不代表任意硬件、任意 shape 或所有可选优化组合均已验证。cuDNN 通过数值和容量检查；本次持续自博弈训练仍使用默认 manual attention。

## 旧地主下降检查点

续训沿用地主表现下降的旧模型。原 1,000-update 完整状态经已有 `import-checkpoint` 流程迁入新隔离目录，逐数组确认参数、冠军、Adam m/v 和 replay 与原状态相等后，再执行 4,000 次更新。

在新的诊断 seed=930103 的 1,000 对发牌上：

| 状态 | 地主胜率 | 农民团队胜率 | 均衡胜率 |
|---|---:|---:|---:|
| 原初始模型 | 16.5% | 16.4% | 16.45% |
| 原下降检查点，1,000 updates | 11.6% | 38.4% | 25.00% |
| 同一状态续训至 5,000 updates | 38.9% | 45.6% | 42.25% |

旧检查点的地主回升 **27.3 个百分点，配对差值 95% CI [24.0, 30.6]**。相对初始模型也提高 22.4 个百分点，区间 [19.0, 25.9]。

另一个从零开始的 seed 42 运行，在诊断 seed=930101 上从初始地主 16.4% 提高到 5,000-update 的 44.4%。

这些结果支持短训练阶段和轨迹随机性会造成角色表现波动的判断，1,000 步尚不足以评估长期地主效果。此次未发现需要翻转奖励、改变角色映射或重设 loss 权重的缺陷，但也未从因果上排除这些因素。

## 三个 Seed 的泛化

事先固定 training seeds=42/43/44，各训练到 **2,500 cycles、10,000 updates**，使用该固定终点的 latest 参数，不依据最终牌局成绩挑选 checkpoint。

| Training seed | 完成牌局 | 完整轨迹样本 | Adam step | 累计训练计时约 |
|---|---:|---:|---:|---:|
| 42 | 122,050 | 5,120,001 | 10,000 | 467 秒 |
| 43 | 126,729 | 5,120,013 | 10,000 | 433 秒 |
| 44 | 119,268 | 5,120,029 | 10,000 | 430 秒 |

42 从诊断阶段续训；43/44 从零连续运行。学习配置与总更新量一致，但重启安排不同，不把 seed 间差异当作纯 seed 因果效应。所有参数、冠军、Adam m/v 均有限，`nonfinite_steps=0`，replay 三个角色齐全。

全部模型完成固定训练量后，才首次生成规则留出 seed=951101 和 DouZero 留出 seed=952101。每组 1,000 对发牌、2,000 局，同一发牌交换地主/农民阵营。差值以完整发牌对为单位 bootstrap 5,000 次；以下为逐项名义 95% CI，不是多重比较调整后的联合区间。

| 对手 | Training seed | 初始均衡 | 10,000-update 均衡 | 提升百分点及 95% CI |
|---|---:|---:|---:|---|
| 简单规则 | 42 | 15.80% | 54.20% | +38.40，[36.20, 40.70] |
| 简单规则 | 43 | 0.95% | 59.05% | +58.10，[56.30, 59.95] |
| 简单规则 | 44 | 2.90% | 55.85% | +52.95，[51.00, 54.90] |
| 现有 DouZero 权重 | 42 | 3.25% | 12.85% | +9.60，[8.15, 11.05] |
| 现有 DouZero 权重 | 43 | 0.50% | 14.45% | +13.95，[12.55, 15.40] |
| 现有 DouZero 权重 | 44 | 0.55% | 11.45% | +10.90，[9.55, 12.25] |

分角色结果如下：

| 对手 | Training seed | 地主胜率 | 地主提升及 95% CI | 农民团队胜率 |
|---|---:|---:|---|---:|
| 简单规则 | 42 | 53.8% | +36.7，[33.3, 40.1] | 54.6% |
| 简单规则 | 43 | 61.8% | +60.9，[57.8, 63.9] | 56.3% |
| 简单规则 | 44 | 53.2% | +47.6，[44.3, 50.9] | 58.5% |
| 现有 DouZero 权重 | 42 | 13.2% | +10.0，[7.8, 12.3] | 12.5% |
| 现有 DouZero 权重 | 43 | 15.7% | +15.3，[13.0, 17.5] | 13.2% |
| 现有 DouZero 权重 | 44 | 10.9% | +10.2，[8.2, 12.1] | 12.0% |

三个 seed、两类对手的地主和均衡差值区间下界全部大于零；农民团队的差值区间下界也全部大于零。支持这三次运行具备留出牌局学习效果，不证明所有初始化或所有超参数都能如此。

服务器现有三个 DouZero 格式权重仅只读挂载，CPU PyTorch 2.8.0 使用 `weights_only=True` 和严格加载。权重 SHA-256 与上次一致；没有独立验证发布来源，不称为某一指定官方冠军模型。

## 连续运行与恢复

seed 42 在 10,000-update 状态后继续一个 **1,800 秒连续训练会话**，保持正常内部评估。为了反复覆盖周期保存，测试前将此阶段的操作性保存间隔从 300 秒改为 60 秒；目标、优化器、探索与采样没有改变。此前较短运行中的冠军晋升保存会重置计时，未实际触发周期保存，不能把它们算作周期 IO 验证。

| 项目 | 实际结果 |
|---|---|
| 训练会话 / 外层进程 | 1,800.85 / 1,801.94 秒，正常 time-limit 退出，退出码 0 |
| 最终状态 | cycle=13,155，updates=52,616，games=744,362，完整样本=26,940,716 |
| 周期保存 | 28 次，覆盖 collection 和 learning 保存路径 |
| 数值检查 | 无非有限更新；最终参数、冠军、Adam m/v 有限，replay 三角色完整 |
| 正常完整周期统计 | 去除 25 个预热周期和 1 个时限尾周期后，10,629 个周期，42,516 次更新，425 组内部评估对 |
| 完整周期吞吐 | 23.76 updates/s，计入周期内评估与保存；不是 learner-only 速度或配置排行榜 |
| GPU 监测 | 每 30 秒采样，共 61 条；暖态显存 8,480 至 8,482 MiB，温度 61 至 73°C，无 OOM |
| CPU 进程 RSS | 暖态总和峰值约 2.77 GiB，末段中位比初始暖态约高 229 MiB；含共享页重复计数、监测进程和日志/JIT 等影响，未证明不存在内存泄漏 |

已有 `metrics_summary` 将该完整周期窗口判定为 `eligible=true`。稳态窗口内没有新静态 shape、非有限指标或不完整内部周期；时限尾周期单独排除。

额外验证了三种恢复路径：

1. 正常退出后新进程续训，学习状态与计数继续增长。
2. 实际向 launcher 发送 SIGTERM，安全保存，退出码 0；恢复后 updates 从 53,169 增至 53,173，完成一个额外周期。
3. 从真实 `periodic_learning` 快照独立恢复，导入前后参数、冠军、Adam、replay、主 RNG 相等；继续三周期，Adam/updates 从 45,524 增至 45,536，参数实际改变，新增 12 步均有限。

周期快照可位于尚未写完周期日志的位置。本次该快照的完整周期日志比优化器计数少 4 步，续训后历史差额仍为 4；没有丢失或重置优化器更新。恢复验证以 Adam step、checkpoint counters 和新会话增量为准，**不能总用历史 `train.successful_steps` 之和代替更新计数**。在途 actor 对局重启，也不承诺逐位相同的 actor 延续。

## 长训后的棋力限制

使用另外两组此前未评估的留出 seed=951102/952102，同时评估初始、10,000-update 和 52,616-update 策略，没有根据这些结果改参数或重选终点。

| 对手 | 状态 | 地主 | 农民团队 | 均衡 |
|---|---|---:|---:|---:|
| 简单规则 | 初始 | 14.7% | 13.3% | 14.00% |
| 简单规则 | 10,000 updates | 57.3% | 55.5% | 56.40% |
| 简单规则 | 52,616 updates | 60.3% | 47.2% | 53.75% |
| 现有 DouZero 权重 | 初始 | 3.8% | 2.3% | 3.05% |
| 现有 DouZero 权重 | 10,000 updates | 11.7% | 11.4% | 11.55% |
| 现有 DouZero 权重 | 52,616 updates | 12.5% | 18.2% | 15.35% |

长训相对初始仍有显著均衡提升：规则 +39.75 个百分点，CI [37.60, 41.95]；DouZero 权重 +12.30，CI [10.75, 13.95]。

但相对 10,000-update 策略，规则对手上的农民下降 **8.3 个百分点，CI [-11.7, -4.9]**，均衡下降 2.65，CI [-4.80, -0.50]。对 DouZero 权重的均衡则提高 3.80，CI [2.25, 5.40]；地主 +0.8 的区间 [-1.3, 2.9] 跨零，不能声称该地主增幅显著。

在线自博弈中已观察到角色表现随对手和训练阶段波动，不能只凭 loss、步数或内部冠军分数部署 latest。延长训练和选择模型时，仍需同时检查各角色对不同对手的表现。本次未根据留出分数改动训练算法。

## 复现与证据

新标准镜像 `douzero-test:latest` 的 ID：

```text
sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e
```

原基础镜像为 `sha256:82ae6978a48c8607b74f8d83437bdeee8d2c8778529f369d8c6355581b5262b5`。原 Dockerfile 冷重建在下载 JAXLIB 时取消，随后复用已验证基础镜像，按当前 `requirements-gpu.txt` 更新固定依赖并执行 `pip check`，避免重下载整套未改变的 CUDA 组件。派生构建文件与日志均保留，不声称完成了一次从零的 OS 冷构建。

实际构建与主要检查命令形式如下，原临时目录已清理，复验需在新隔离目录恢复归档源快照：

```bash
docker build -f "$S/project/Dockerfile.fixed" -t douzero-fixed:20261001 "$S/project"
docker tag douzero-fixed:20261001 douzero-test:latest

# 在挂载项目、证据目录的容器内，PYTEST_DISABLE_PLUGIN_AUTOLOAD=1：
JAX_PLATFORMS=cuda python -m pytest -q --junitxml=/evidence/fixed-gpu-full.xml
JAX_PLATFORMS=cpu python -m pytest -q --junitxml=/evidence/fixed-cpu-full.xml
python /evidence/followup.py preflight
python /evidence/followup.py diagnostic
python /evidence/followup.py legacy
python /evidence/followup.py confirm
python /evidence/periodic-resume.py
```

pytest 前在临时容器安装 `requirements-dev.txt`；DouZero 权重评估另装 CPU PyTorch。两个依赖安装均不进入宿主 Python。`confirm` 前应执行已记录的 stability 保存间隔调整；完整挂载、环境、执行阶段和调整理由见归档脚本及日志。

本次独立评估共 48,000 局，其中诊断 12,000 局，事先固定的最终留出评估 36,000 局。原始逐发牌结果、配置、策略、完整学习状态、XML、依赖、硬件监测、源码快照与 SHA-256 清单均已归档。

以下证据路径相对于项目根目录，位于已忽略的 `reports/` 中，原始证据未随仓库发布：

- 本次证据目录：`reports/localserver-followup-20261001`
- 验证驱动：`reports/localserver-followup-20261001/evidence/followup.py`
- 连续运行指标汇总：`reports/localserver-followup-20261001/evidence/seed42/stability-metrics-summary.json`
- 周期恢复结果：`reports/localserver-followup-20261001/evidence/periodic-recovery.json`
- [历史初次验证报告](gpu-validation-2026-09-30.zh-CN.md)

## 清理与剩余边界

本次容器使用 `--rm`，结束后无遗留运行容器；仅清理本次创建的远端临时目录和临时镜像标签。更新后的标准镜像及已有构建缓存保留，没有执行任何 prune，也没有修改宿主驱动、CUDA、Docker daemon 或全局 Python。

`/opt/DouZero` 仍是原有 17 项已跟踪修改和 6 项未跟踪路径，tracked diff SHA-256 与初始验证相同。该机器上用户已有代码、权重和实验没有被覆盖。本地仓库仅修改此次精度、预检、测试和 NVRTC 约束及相应文档/源码锁；原回归失败证据保留。

本次默认配置通过了训练和恢复检查，所测留出表现也有改善。验证仍限于三个训练 seed 和半小时持续运行：强基准绝对胜率较低，角色表现随对手和训练阶段波动，长时间内存趋势及其余优化组合尚待验证。
