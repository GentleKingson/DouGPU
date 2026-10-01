# DouGPU LocalServer 训练有效性验证报告

项目名称已统一为 DouGPU。下文保留验证当时的包名、路径、镜像与哈希，均为历史证据，不代表当前项目名或最新源码；当前入口见 [README](../README.md)。

本次通过 SSH 在 LocalServer 的 RTX 5070 上验证本地仓库快照，未使用服务器已有的另一套 DouZero 代码。默认 manual 路径完成了真实自博弈、学习、保存和续训，训练后策略在独立牌局上的均衡胜率有所提升。但全套回归未通过，部分优化路径仍有问题。

验证时间：2026-09-30 23:24 至 23:44，Asia/Hong_Kong。验证期间未修改训练源码或覆盖用户已有实验，也未启动持续后台训练。

## 验证对象与环境

| 项目 | 实际信息 |
|---|---|
| 连接方式 | `ssh LocalServer` |
| 本地验证对象 | 本地项目工作区（当时目录名为 `DouZero`）的当前文件快照 |
| 服务器原仓库 | `/opt/DouZero`，detached HEAD，已有 17 个已跟踪文件修改和 6 项未跟踪路径；未覆盖 |
| 隔离工作目录 | `/tmp/dougpu-validation-20260930-232612/project`，验证后已清理 |
| CPU 和内存 | Intel Core Ultra 5 245KF，14 个可见 CPU，约 30 GiB RAM |
| GPU | NVIDIA GeForce RTX 5070，12,227 MiB 显存 |
| 宿主系统和驱动 | 原生 Linux x86_64，内核 7.0.0-30-generic，驱动 595.71.05 |
| Docker | 29.8.1，所有项目测试、依赖安装和训练均在容器中执行 |
| Python 和 JAX | Python 3.12.14，JAX/JAXLIB/CUDA plugin/PJRT 0.7.2 |
| 主要依赖 | NumPy 2.5.3，SciPy 1.18.1，CUDA runtime 13.0.96，cuDNN 9.18.1.3 |
| 训练配置 | 原样使用 `configs/rtx5070_balanced.json`，training seed=42 |
| 模型 | 922,401 个参数，width=128，3 层，BF16，完整 512-token 历史上限 |
| 执行设置 | manual attention，history_groups=1，8 workers × 8 局，micro_batch=32 × accumulation=8 |

训练源码哈希与仓库此前记录相同：

```text
fdcb6ac40db6d690e0b58e44c54e4e79cf6d77ae31eb756dcd707e3e3c7f370a
```

固定规则提交为 `718a5c920bf3361e34178a38f3b80458e176b351`。本地上传快照与服务器收到的压缩包 SHA-256 一致：

```text
52a3e9d27ba46cc2eb02d0c0f28d1b486887e9651bad4d43acb051de3004c6fa
```

Docker 镜像使用服务器要求的标准名称 `douzero-test:latest`，由当前仓库 Dockerfile 构建。测试工具及可选 CPU PyTorch 仅安装在临时容器中，不修改宿主 Python，也不写回镜像。

```text
sha256:82ae6978a48c8607b74f8d83437bdeee8d2c8778529f369d8c6355581b5262b5
```

## 规则与完整模型预检

规则自检通过：100 局、4,652 次合法出牌，观察到最长 226 token、最多 244 个合法动作；自检结束时 `jax_loaded=False`。

`doctor` 检测到唯一 GPU，真实 BF16 矩阵乘法通过，没有 CPU fallback。

默认完整模型 `preflight` 通过，耗时约 39.5 秒。四档历史均使用全长非 PAD token，检查完整反向传播、参数改变及 Adam 步数增长，共验证 16 次优化器更新。推理检查还确认能选中第 2,049 个动作位置上的最优候选，而不是截断到第一个 2,048 槽分块。

| 有效历史长度 | 预热后合成更新中位耗时 | 最小参数最大绝对改变量 |
|---|---:|---:|
| 64 | 8.48 ms | 0.0001000 |
| 128 | 11.94 ms | 0.0001011 |
| 256 | 20.33 ms | 0.0001035 |
| 512 | 61.52 ms | 0.0001071 |

这些数据来自合成内核和容量检查，不能用于比较实际训练吞吐。JAX 报告的内部 `peak_bytes_in_use` 为 271,482,880 bytes，预分配池约 8.64 GB；两者口径不同，前者不代表整张 GPU 的峰值显存占用。

## 真实训练与续训

使用同一独立实验目录，依次运行 2、1、22、225 个额外周期。保持原训练目标、学习率、模型与执行配置不变。

| 阶段 | 累计周期 | 累计成功更新 | Adam step | 累计完成牌局 | 累计完整轨迹样本 |
|---|---:|---:|---:|---:|---:|
| 两周期 smoke | 2 | 8 | 8 | 89 | 4,137 |
| 新进程续训 | 3 | 12 | 12 | 136 | 6,366 |
| 第一段短训 | 25 | 100 | 100 | 1,024 | 51,213 |
| 扩展短训 | 250 | 1,000 | 1,000 | 10,806 | 512,070 |

完整 checkpoint 通过归档校验；参数、冠军参数、Adam m/v 均有限，replay 可读取。续训后计数递增，参数实际改变，没有重新初始化学习状态。

全部 250 个周期共 1,000 次成功更新，`nonfinite_steps=0`。最终 replay 达到 65,536 条容量。累计记录 20 次内部评估事件，冠军在 cycle=125 晋升一次；独立棋力评估使用 cycle=250 的 latest 策略，而不是借内部结果选择 best 策略。

四次训练 CLI 墙钟分别约 5.2、5.1、18.4、48.7 秒。该短实验未达到 300 秒周期保存间隔，因此没有实际触发时间驱动的周期存盘；已验证启动、冠军晋升及正常退出存盘。没有据此宣称长期稳定性或最优硬件配置。

## 固定真实样本学习检查

从真实自博弈 replay 冻结抽取 256 条样本，三个角色分别占 86、85、85 条，最长有效历史 227 token。从相同 seed 的初始完整模型开始，对这批数据执行 100 次 GPU 更新，不再采样。

| 指标 | 初始 | 100 步后 |
|---|---:|---:|
| 总 loss | 1.086002 | 0.050233 |
| Q loss | 1.000091 | 0.012065 |
| NTP loss | 3.865369 | 1.741947 |
| belief loss | 0.172072 | 0.066579 |

Q loss 降低约 98.79%，每步均有限，Adam step 最终为 100，说明这套目标、梯度与优化器实现能够拟合固定样本。该实验仅检查过拟合能力，其参数未用于后续棋力评估，也未写回自博弈 checkpoint。

## 独立牌局棋力评估

每个对手和每个 seed 使用 1,000 对发牌，共 2,000 局；同一发牌交换策略的地主/农民阵营。初始策略来自最初冻结的冠军参数，训练后策略分别为 100 和 1,000 次自博弈更新后的参数。均衡胜率是地主胜率与农民团队胜率的平均值。

使用现有 `paired_evaluate` 执行对局，只额外记录其汇总前的逐对结果；没有改动策略选择、规则或胜负计算。增幅区间使用同发牌的前后策略差值，以完整发牌对为单位 bootstrap 5,000 次，避免将相关的两局当作独立样本。

| 对手 | 评估 seed | 初始均衡胜率 | 100 步 | 1,000 步 | 1,000 步相对初始增幅及 95% CI |
|---|---:|---:|---:|---:|---|
| 简单规则 | 910001 | 15.60% | 17.30% | 25.75% | +10.15 个百分点，区间 [8.05, 12.25] |
| 简单规则 | 910002 | 16.70% | 19.75% | 26.90% | +10.20 个百分点，区间 [8.00, 12.45] |
| 简单规则 | 910003 | 15.30% | 未评估 | 25.20% | +9.90 个百分点，区间 [7.80, 12.05] |
| DouZero 格式基准权重 | 920001 | 2.95% | 3.45% | 4.40% | +1.45 个百分点，区间 [0.35, 2.60] |
| DouZero 格式基准权重 | 920002 | 2.10% | 未评估 | 4.80% | +2.70 个百分点，区间 [1.60, 3.80] |

910001、910002、920001 在较早阶段已用于诊断；910003、920002 是最终训练结束后才首次评估的留出牌局，没有用于冠军选择或调整训练配置。全部独立评估共执行 26,000 局。

最终留出结果支持“本次训练后的均衡棋力优于初始模型”。但提升并非每个角色都一致：规则对手 seed=910003 上，地主胜率从 15.30% 降至 13.30%，农民团队胜率从 15.30% 升至 37.10%。前两组规则牌局的地主点估计也下降，应继续分别监控角色表现，不能只看总均值。

强基准上的最终胜率仍很低。这里只验证了一个训练 seed，不能据此声称已经达到 DouZero 水平，或证明每次从零训练都能得到相同提升。

服务器已有的三份权重只读挂载自 `/opt/DouZero/baselines`，使用 CPU PyTorch 2.8.0、`weights_only=True` 与严格 state_dict 加载，兼容性通过。未独立验证权重发布来源，因此称为“服务器现有 DouZero 格式基准权重”，不保证它们是某一特定官方发布版本。

```text
landlord       6f2971813495e9c509cbd4a101213c49587472942fa0f018d04c139a6a647c2d
landlord_down  c173dbe30258f19b6392c30ab1e330b6b3090bbd9ccdc89a9b18123a6ab6e2d9
landlord_up    7fb7c495ff095db2ddadd8f9f8fd53cad840211578dabc83bbcc8d75085f843e
```

## 回归失败与诊断

以下各行是不同执行批次，测试用例有重叠，不能把通过数相加当作唯一测试数。

| 批次 | 通过 | 失败 | 跳过 | 实际 pytest 耗时 |
|---|---:|---:|---:|---:|
| 完整 CPU 回归 | 200 | 2 | 1 | 44.16 秒 |
| GPU 注意力专项 | 8 | 3 | 1 | 25.56 秒 |
| 完整 GPU 环境回归 | 191 | 11 | 1 | 149.11 秒 |
| CPU 两项失败定向重跑 | 1 | 1 | 0 | 2.76 秒 |
| 显式 FP32 矩阵精度后的 XLA 定向诊断 | 2 | 0 | 0 | 6.27 秒 |
| 显式 FP32 矩阵精度后的分组/KV/PAD 诊断 | 4 | 0 | 0 | 24.19 秒 |

完整 GPU 环境批次的 11 项失败分为：

1. **cuDNN 路径不可用，1 项。** `test_cudnn_bf16_valid_outputs_losses_and_gradients` 报 `No valid execution plans built`。这是实机内核运行失败，不是仅靠放宽数值容差就能解决的问题。本次没有改动 cuDNN/JAX 包，也没有改用 cuDNN 训练。
2. **FP32 优化路径等价性失败，6 项。** XLA loss、XLA 单步优化器、history_groups=2/4、FP32 KV cache、PAD 后梯度比较超出已有容差。定向设置 `JAX_DEFAULT_MATMUL_PRECISION=float32` 后，这六项全部通过，说明误差与默认矩阵计算精度有关；尚不能据此宣称整套优化路径均已通过。训练和棋力评估始终使用原默认设置，该诊断开关没有写回配置。
3. **动作分块逐位一致性失败，2 项。** FP32/BF16 两个用例都使用 `assert_array_equal`，GPU 批次观察到约 `1.46e-11` 的差异。CPU 首轮 FP32 用例也失败，但独立定向重跑通过，不能声称该项在 CPU 上每次都必然失败。
4. **CPU 同分动作预检失败，1 项。** GPU 完整批次中也会启动一个显式 CPU 的预检子进程。独立诊断中，9 个相同动作的数值评分有约 `4.66e-10` 差异，选择结果为 index=6，而预检期望 index=0。这里不是动作被截断，而是数学上相同输入没有得到逐位相同输出；现有精确同分规则只处理实际计算出的分数完全相等的情况。
5. **测试后端前提不适配，1 项。** `test_resume_does_not_initialize_discarded_parameters` 在已初始化 GPU 的 pytest 进程中请求 CPU，后端保护正确拒绝切换。该用例在完整 CPU 批次通过；本次真实 GPU 新进程续训也通过，不应把这项解释成 GPU 续训损坏。

旧记录中的“202 项通过”不适用于这次实机验证：默认训练流程通过了，但完整回归仍有失败项。建议暂时使用 manual、history_groups=1，并关闭研究型 KV cache。cuDNN 和其他执行优化通过目标设备上的检查后再启用。

数值测试应先明确允许的误差与动作选择标准，避免把所有失败简单改成宽松容差；尤其需要把“数值接近”与“精确同分时选择最早动作”分别验证。

## 命令与证据

实际镜像构建命令：

```bash
ssh LocalServer 'docker build --progress=plain -t douzero-test:latest /tmp/dougpu-validation-20260930-232612/project'
```

三次验证容器均使用 `docker run --rm --init --gpus all --shm-size=2g`，挂载隔离项目到 `/app`、证据目录到 `/evidence`、旧基准权重到 `/weights:ro`。环境设置如下，容器入口分别为 `/evidence/verify.py`、`extension.py`、`holdout.py`：

```text
PYTHONPATH=/app:/app/vendor
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
JAX_COMPILATION_CACHE_DIR=/app/.cache/jax
```

CPU 批次通过子进程环境 `JAX_PLATFORMS=cpu` 明确选择 CPU，GPU 批次为 `JAX_PLATFORMS=cuda`。关键容器内命令：

```bash
python -m pytest -q --junitxml=/evidence/cpu-tests.xml
python -m pytest -q tests/test_gpu_attention.py --junitxml=/evidence/gpu-tests.xml
python -m pytest -q --junitxml=/evidence/gpu-full-tests.xml
python run_local.py selftest --games 100
python run_local.py doctor --output /evidence/hardware.json
python run_local.py prepare --config configs/rtx5070_balanced.json --run-dir /app/runs/validation
python run_local.py preflight --run-dir /app/runs/validation
python run_local.py train --run-dir /app/runs/validation --cycles 2 --hours .25
python run_local.py train --run-dir /app/runs/validation --cycles 1 --hours .25
python run_local.py train --run-dir /app/runs/validation --cycles 22 --hours .25
python run_local.py train --run-dir /app/runs/validation --cycles 225 --hours .25
```

完整命令、退出码和墙钟时间保存在 `stages.json`。CPU/GPU 原始日志及 JUnit XML、精度诊断日志、各策略和对手哈希、逐对对局结果、bootstrap 对比报告、完整配置、source lock、源码上传包均已归档。

本地证据目录（相对于项目根目录；位于已忽略的 `reports/` 中，原始证据未随仓库发布）：

```text
reports/localserver-20260930-232612
```

其中 `experiment/checkpoints/` 保留完整的 cycle=25、125、250 三代 checkpoint、对应 `.ok.json` 以及 latest/best 策略；`experiment/metrics.jsonl` 保留全部训练记录，`experiment/preflight.json` 保留完整 GPU 预检结果。固定样本学习曲线为 `fixed_batch.json`，最终留出对比为 `comparison_extended_rule_910003.json` 和 `comparison_extended_douzero_920002.json`。

从远端取回的 131 个证据文件逐一通过 SHA-256 比对，0 项不一致；原始远端文件哈希清单保存在 `remote-files.sha256`。

## 清理与验收边界

三次容器均已自动删除，服务器本次创建的整个 `/tmp/dougpu-validation-20260930-232612` 已删除，临时项目副本、训练目录、JAX 缓存及容器内测试依赖均不再留在服务器。保留标准测试镜像 `douzero-test:latest`，约 6.06 GB，并保留 Docker 构建缓存；没有清理其他任务的镜像或卷。

清理后 GPU 为 0% 利用率、18 MiB 显存占用。`/opt/DouZero` 的验证前后 git status 完全一致，已跟踪改动的 binary diff 哈希也一致。所有旧权重以只读方式访问，没有修改宿主驱动、CUDA、SSH、Docker 配置或全局 Python 依赖。

本地只新增本报告和被 `.gitignore` 忽略的证据目录，没有修改训练实现或替用户选择正式长训配置。

本次验收支持默认路径继续进行受控实验，但不支持“全套测试通过”“全部角色提升”“cuDNN 已兼容”“达到强基准水平”或“长期训练稳定”的结论。后续优先处理回归失败与地主表现下降，再以多个训练 seed、全新留出牌局和更长持续运行验证泛化与稳定性。
