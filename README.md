# DouGPU

DouGPU 由 `DouTPU_v6e1_Optimized_Final.ipynb` 迁移而来，目标硬件为 Intel Core Ultra 5 245KF、桌面 NVIDIA RTX 5070 12 GB 和 32 GB RAM。训练使用 JAX，由单个进程管理 GPU，多个 CPU actor 负责 DouZero 牌局和编码。

迁移保留了原模型、公开信息边界、DMC/NTP/belief 目标、完整合法动作和完整 checkpoint 格式。默认使用 balanced 基线配置，也可选用在 LocalServer 的 RTX 5070 上实测的吞吐配置。架构、硬件依据、迁移决策和验收边界见 [迁移报告](docs/gpu-migration.zh-CN.md)，实测过程见 [验证记录](docs/validation-notes.md)。

目前已验证基本学习能力和限定时长内的稳定性，但这些结果不足以证明达到强 DouZero 基准水平。训练默认采用自博弈，历史冻结对手池默认关闭。

## 1. 环境与安装

支持原生 Linux x86_64，或 Windows 下的 WSL2 Ubuntu。JAX 官方将 WSL2 NVIDIA GPU 支持标为 experimental；原生 Windows Python 不能运行本项目的 JAX CUDA 训练。[S1]

请先安装 Python 3.12，安装脚本不会代为安装。可用 `bash scripts/install.sh --python /path/to/python3.12` 指定解释器。

项目固定使用 JAX/JAXLIB/plugin/PJRT 0.7.2 和 cuDNN 9.18.1.3。CUDA 组件主要沿用 13.0 Update 2，其中 NVRTC 单独固定为 13.1.115，以修复本机 cuDNN attention 的运行时编译失败；这套依赖因此与原版 CUDA 13.0 Update 2 有所不同。22 个核心依赖的版本与 wheel 记录见 [依赖清单](docs/dependency_resolution.json)。安装约束以 `requirements-*.txt` 和 `constraints-cuda13.txt` 为准，实际安装结果见 `.venv/resolved-gpu.txt`。CUDA 库通过 pip wheel 安装，无须安装全套系统 CUDA Toolkit。

原生 Linux 驱动须支持 RTX 5070 与 CUDA 13；JAX 文档要求 CUDA 13 使用 **580 系列或更新**的驱动。CUDA 13.0 Update 2 的配套 Linux 驱动为 580.95.05，但不能把它当作本项目混合组件与全部内核的实测最低版本。本项目 GPU 验证使用过 595.71.05、595.91.07；你的环境仍须通过 `doctor` 与 `preflight`。WSL 使用 Windows 宿主 NVIDIA 驱动。[S1][S2][S3]

在 WSL/Linux 的本地磁盘中解压项目，然后执行：

```bash
# 替换为实际项目目录；旧检出目录 DouZero 无须为改名而移动。
cd /absolute/path/to/DouGPU
bash scripts/install.sh
source .venv/bin/activate

python run_local.py doctor --output reports/hardware.json
```

安装器创建或复用项目内的 Python 3.12 `.venv`，并生成 `.venv/resolved-gpu.txt`；已有环境中的其他包不会自动清除。`doctor` 必须报告 `backend: gpu`，完成一次真实 BF16 矩阵乘法，并列出实际 GPU、驱动和依赖版本。找不到 CUDA 时检查失败，不会改用 CPU。

WSL 下把项目、`.venv`、编译缓存、replay/checkpoint 放在 Linux 文件系统，例如 `~/DouGPU`；大批文件操作优先使用 Linux 路径。[S4] 已有 32 GB 内存时，可按需要给 WSL 分配约 24 GB；先查看实际 CPU 配额，不必硬编码 P/E 核编号。在 WSL 内不要安装 Linux NVIDIA 显示驱动。[S3]

若 shell 已设置 `LD_LIBRARY_PATH`，检查它是否覆盖了 pip 的 CUDA/cuDNN 库。`doctor` 会记录此情况。[S1] 本项目不自动修改宿主驱动或系统环境。

## 2. 建立独立本地实验

```bash
python run_local.py prepare \
  --config configs/rtx5070_balanced.json \
  --run-dir runs/5070

python run_local.py selftest --games 100
```

项目已包含固定提交 `718a5c920bf3361e34178a38f3b80458e176b351` 的 DouZero 规则源码及其许可证。`prepare` 校验这些规则字节，并为新训练代码建立来源锁；首次启动无需临时追踪 GitHub 主分支。`selftest` 只执行规则和编码检查，不初始化 JAX。

默认配置：

| 配置项 | 本地起点 | 说明 |
|---|---:|---|
| CPU workers | 8 | 为主进程、JAX 调度及系统保留余量 |
| 每 worker 牌局 | 8 | 共 64 局在途游戏 |
| actor groups | 2 | 每组 4 workers、32 局；保持有序合并 |
| inference batch | 32 | 与单组游戏数对齐 |
| action chunk | 2048 | 超出时继续分块，完整评分 |
| micro batch × accumulation | 32 × 8 | 有效 batch 固定 256 |
| history groups | 1 | 先减少静态形状组合；可单独测 2/4 组 |
| learner remat | inherit | 保留原 ModelConfig 中 remat=true |
| precision | BF16 主干 / FP32 参数与优化器 | 保留原混合精度策略 |
| replay | 65,536 条 | 保存回放及角色权重语义 |
| fresh samples / updates | 2048 / 4 | 名义学习样本复用比 0.5 |
| evaluation | 每 25 周期、每对手 256 对验证牌局 | 分角色晋升检查；不用于最终棋力结论 |
| checkpoint | 300 秒、保留 3 代 | 在下一安全点保存，可能被编译或正在执行的阶段延后 |

需要更少 CPU 竞争时，改用 `configs/rtx5070_conservative.json`：6 workers × 8 局。它同样是起点，不是已证明的最快配置。

`configs/rtx5070_throughput.json` 是 2026-10-01 在 LocalServer 上测量的可选配置：cuDNN attention、learner remat=false、64 × 4 micro/accumulation、2 个历史分组、每 worker 16 局及 inference batch=64。模型和训练目标不变。

三次短测从同一 checkpoint 开始，交叉执行两种配置，完整周期吞吐从 18.35 提高到 25.86 更新/秒，约 +41%。测试保留了当时的评估，两边保存间隔统一缩短为 10 秒，生产配置仍为 300 秒。测量窗口与源码版本见 [吞吐验证记录](docs/validation-notes.md#5-2026-10-01-localserver-吞吐优化)。后续晋升逻辑已有更新，这组吞吐结果仅适用于当时的版本和机器。

该配置通过了完整 512-token 容量与数值预检；其他机器上的最优配置、长期稳定性和棋力提升仍需另测。用它建立新实验时，先运行 `preflight`。

## 3. 选择从零训练，或继续 TPU 学习状态

### 从零训练

完成 `prepare` 后直接进入下一节。没有 checkpoint 的新实验会从初始化模型开始，不能接续 notebook 输出中显示的训练进度。

### 从 TPU 完整续训

上传的 notebook 包含源码和历史输出，**不包含参数、Adam、replay 的实际训练快照**。先把 Colab/Drive 实验的 `state` 目录复制到本机，其中必须有一组相匹配的：

- `ckpt_....zip`
- `ckpt_....ok.json`

然后向一个已 `prepare`、尚无 checkpoint 的新目录导入：

```bash
python run_local.py import-checkpoint \
  --source-state /absolute/path/to/copied_tpu_experiment/state \
  --run-dir runs/5070
```

如需复用原实验的规则缓存，可在 `prepare` 使用 `--upstream-cache /absolute/path/to/copied_tpu_experiment/source`。该目录须含 `source_lock.json` 和 `upstream_source.zip`。此选项只复用缓存，仍要求 notebook 固定的 DouZero 提交及相同文件哈希，不能用来切换规则版本。

导入时会核对完整归档及每个成员的哈希，并检查模型、完整参数结构、Adam step、规则源码和训练语义。参数、Adam m,v、冠军、replay、主 RNG、计数器和 sample credit 都会保留，来源记录写入 `performance_provenance.json`。源目录不会被改写。

`latest_policy.npz` 和 `best_policy.npz` 只用于策略部署与评估；它们缺少辅助头、优化器和 replay，不能用于完整续训。TPU→CUDA 与 actor 数量变化也不保证逐位相同的后续训练轨迹；未完成的 actor 牌局会重开。

## 4. 先预检，再做两周期 smoke，再长训

```bash
# 当前配置的模型、跨块动作评分及全 512 个非 PAD token 的合成反向检查。
python run_local.py preflight --run-dir runs/5070

# --cycles 表示本次会话额外运行的周期，不是全局周期上限。
python run_local.py train --run-dir runs/5070 --cycles 2 --hours 1

# 上一步正常后，从相同目录继续。
python run_local.py train --run-dir runs/5070 --hours 12
```

预检与训练分别拥有 GPU，顺序执行。`preflight` 使用新初始化的参数和合成数据，不加载或改写正式训练 checkpoint，也不等于真实自博弈测试。历史分组大于 1 时，只预热各组等宽的组合，混合长度组合仍可能在训练中首次编译。首次编译会花时间；缓存默认保存在 `.cache/jax`，可由 `JAX_COMPILATION_CACHE_DIR` 覆盖，后续是否命中取决于形状、代码、编译配置和设备。不要使用不可信用户可写入的编译缓存。[S5]

确认日志中的 `successful_steps > 0`，且 checkpoint 的 `optimizer.step`/`updates` 增长。两周期 smoke 只验证训练流程能否跑通，棋力和 GPU 性能需要另行评估。`Ctrl+C`/SIGTERM 会请求在安全点存盘，launcher 会等待 learner 完成保存。直接断电或强制终止只能恢复到最后一代已提交 checkpoint。

从零开始的两周期 smoke 不会到达默认第 25 周期的评估点，不能验证冠军晋升；导入 checkpoint 后是否触发评估取决于恢复的全局周期。常用结果位置：

| 路径 | 内容 |
|---|---|
| `runs/5070/metrics.jsonl` | 训练、阶段耗时、评估、保存与会话日志 |
| `runs/5070/checkpoints/` | 完整快照和 latest/best 策略导出 |
| `runs/5070/source/` | 固定规则归档、训练源码归档和来源锁 |
| `runs/5070/config.json` | 建立实验时的基线配置 |
| `runs/5070/session_config.json` | 本次实际执行配置 |
| `runs/5070/preflight.json` | 目标设备预检结果与有限合成基准 |

可选 `train --mirror-dir /path/to/backup` 为完整 checkpoint 建立第二份校验副本。每个位置按 `keep_checkpoints` 保留最近有效代；本地目录本身已经是持久保存位置。

## 5. 分阶段寻找这台机器的最快有效配置

先完成 baseline smoke/短训，正常停止源实验。每个候选都从同一份冻结快照开始，按顺序独占 GPU；不会改动源实验或替换你的正式训练参数。

```bash
# 先生成候选配置与预计训练预算。
python tune_local.py \
  --source-run runs/5070 \
  --config configs/rtx5070_balanced.json \
  --output benchmarks/workers \
  --stage workers

# 正式测量：默认每候选 3 次，每次训练预算 10 分钟，另有预检与启动时间。
python tune_local.py \
  --source-run runs/5070 \
  --config configs/rtx5070_balanced.json \
  --output benchmarks/workers \
  --stage workers --repeats 3 --minutes 10 --execute
```

基线相同的 worker 阶段通常有 4 个候选，约 120 分钟训练预算，加上预检、初始化、导入与最终存盘。可以先用 `--repeats 1` 筛选；报告会明确标为临时证据。目录只生成过计划时可继续执行；已经有 trial/snapshot 的目录不能再次覆盖。

按以下顺序逐阶段执行，把选中候选 JSON 作为下一阶段的 `--config`：

| `--stage` | 比较项 | 保持什么 |
|---|---|---|
| `workers` | 6 / 8 / 10 / 12 个 actor | 每 actor 8 局，并让 inference 容量匹配 |
| `envs` | 每 actor 4 / 8 / 16 / 32 局 | 固定选定 worker 数 |
| `inference` | history buckets、fused 各自开关 | 一次只改一个执行开关 |
| `chunks` | 1024 / 2048 个动作槽 | 所有合法动作仍完整处理 |
| `learner` | 16×16 / 32×8 / 64×4 / 128×2 | 有效 batch = 256 |
| `history` | 1 / 2 / 4 组 | 只重排同一批样本与裁去 PAD |
| `remat` | 重计算开/关 | 原模型配置与学习目标不变 |
| `attention` | manual / cuDNN；XLA 作为诊断项 | 因果性、有效历史、辅助目标不变 |

不必无条件跑完每个阶段：先看 collection、learner、编译和存盘耗时，针对占比最高的部分测量。只有实测胜出的配置才进入后续长期训练。

### 读懂测量结果

主指标为：

```text
sum(successful optimizer updates) / sum(full_cycle_seconds)
```

完整周期墙钟包含正常评估与周期内保存。进程启动、预检、导入和会话结束的最终存盘耗时单独列出，因此主指标不等于整个命令的吞吐。报告还提供样本行/s、采样决策/s、learner-only 吞吐、padding 和测量段首次出现的形状，比较时应区分各项口径。JAX 调用必须先预热，并在设备计算完成后计时。[S6]

默认忽略本会话前 25 周期，忽略导入日志中的旧会话。若测量段没有完整评估、周期保存、有效更新和真实新采样，或者发生非有限步/进程失败，trial 不进入有效排名。预算太短会显示 `unrankable`，应延长单次时间。sample credit 可能造成单个周期只学习而不采样，工具按整个窗口累计，避免这种周期虚增吞吐。

报告位于 `benchmarks/workers/summary.json`。三次重复可得到当前候选组的测量比较，不能证明全局最优或棋力提升。候选参数不自动写回正式实验。

### Colab L4 同参短测

用 `colab new -s dougpu-l4 --gpu L4` 创建独立会话，上传仓库快照、本地实际运行的完整配置和同一个完整 checkpoint（ZIP 与 `.ok.json`）。Colab 中使用独立 Python 3.12 环境安装现有 `requirements-gpu.txt`，不要使用 notebook 预装的不同版本 JAX，也不要套用另一份默认训练配置。

在上传后的仓库根目录执行：

```bash
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements-gpu.txt
.venv/bin/python scripts/colab_benchmark.py \
  --config input.config.json --source-state input-state \
  --run-dir benchmark --seconds 720
```

脚本复用原有导入、doctor、preflight、训练和统计流程；仅覆盖本次运行时长，断言其余配置及输入文件未变。`benchmark/summary.json` 排除前 25 个周期，保留真实采样、周期保存和配置原有的评估。预算必须足够覆盖原保存间隔，否则结果标记为不合格，不会缩短保存间隔凑数。下载日志与结果后用 `colab stop -s dougpu-l4` 释放会话。单次短测只比较该配置下的整机吞吐，不能视为纯 GPU 算力或多次重复基准。

## 6. 显存与注意力设置

默认保留 BF16 主矩阵乘、FP32 参数/Adam/归一化/softmax/Q 标量投影和 loss。不会启用 FP8/FP4 或减少完整历史。

JAX 默认会预分配 GPU 显存；本 launcher 起点设为总显存的 **70%**，为桌面及运行时留出空间。这是 allocator 的预分配比例，并非所有 CUDA 分配的硬上限。[S7] 可以在运行命令前显式覆盖，例如：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 python run_local.py preflight --run-dir runs/5070
```

cuDNN 是待测优化候选：使用明确的 `implementation='cudnn'` 与有效序列长度，避免显式 `[B,T,T]` 注意力矩阵；输入必须是有效非零前缀加右侧 PAD。主机侧会检查这个约定。不支持的 dtype/形状/设备会报错；不静默换回其他内核。[S8]

若在大微批时 OOM，先回到前一个已通过预检的执行配置。可以保留原 `32×8`，或测试 `16×16`；保持有效批量 256，不通过缩短历史/截断动作改变训练任务。`learner_remat` 的覆盖只影响执行，不改变旧快照的 ModelConfig。

## 7. 独立牌局与更强对手评估

```bash
# 示例 seed；正式测试须预先确定未参与任何模型或方案选择的新 seed。
# 1000 对牌局，共 2000 局。
python run_local.py evaluate --run-dir runs/5070 \
  --opponent rule --deals 1000 --seed 970001

# 已安装兼容的 CPU PyTorch，并准备可信的三份 DouZero 格式权重后：
python run_local.py evaluate --run-dir runs/5070 \
  --opponent douzero --weights /absolute/path/to/DouZero_WP \
  --deals 1000 --seed 970002
```

权重目录须含与固定 DouZero 模型定义兼容的 `landlord.ckpt`、`landlord_down.ckpt`、`landlord_up.ckpt`。项目不会自行下载权重，也不把缺失对手替换成随机策略；使用 `weights_only=True` 和严格 `state_dict` 加载。可选对手使用 CPU PyTorch，当前策略仍使用 JAX GPU。PyTorch 不在默认安装或 Docker 镜像依赖中；已有验证在临时评估容器中使用 CPU PyTorch 2.8.0。

2026-09-30 至 2026-10-01 的验证已使用服务器现有三份 DouZero 格式权重完成评估，哈希见 [历史对手报告](docs/historical-opponents-2026-10-01.zh-CN.md#复现与证据)。其发布来源未独立核验，因此不能称作已验证的官方冠军。加载成功和哈希固定也不证明来源或授权；请自行核对权重的可信来源与使用许可。

RTX 5070 配置使用内部 256 对验证牌局，记录地主、农民团队、均衡胜率及各自 95% 配对 bootstrap 区间。晋升要求对冻结冠军的均衡下界超过 50%，同时满足两个角色的非劣性条件：对冠军的角色胜率下界至少 45%，对固定规则对手的角色胜率变化下界至少 -5 个百分点。后者按候选与冠军在相同规则牌局上的结果计算，不能只比较两个总胜率。

门槛采用单侧 99% bootstrap 下界，对每次晋升的五项统计检查作 Bonferroni 调整，但不保证反复挑选模型后的总体误晋升率。`promotion_role_margin=0.05` 是预先设定的可接受回退幅度，允许范围内的退步。少于 128 对牌局时只反馈结果，不晋升。

`run_local.py evaluate` 默认评估 `best_policy.npz`，而非 latest。冠军首次晋升前，best 仍是初始策略；文件名不表示已通过强对手验收。

新导出会记录训练器已知的历史冠军选择 seed，导入时合并源实验与目标实验的记录。两个评估入口都会拒绝使用导出元数据中已记录的选择 seed。额外方案选择使用的 seed 须另行登记并纳入封存元数据；训练器不会自动发现它们。旧导出缺少元数据时，也须人工核对。

代码不会阻止复用已经看过的测试 seed。最终留出结果不能再用来选择 checkpoint 或修改门槛，示例 seed 也不能一直作为新留出集复用。正式棋力评估需要未参与模型选择的新牌局和固定强对手；吞吐提升本身不说明策略变强。

历史冻结对手实验可在 JSON 的 `train` 对象中指定 `historical_opponents` 策略 NPZ 路径列表，再用该配置建立新实验；空列表保持原自博弈。`historical_fraction` 默认 0.5，按完整牌局抽样。每局固定对手和当前策略阵营，两名农民共用各自阵营策略。历史对手不探索、不更新，其动作不进入 replay。`frames` 和探索衰减只统计当前策略的出牌决策，历史决策另记于 `historical_frames` 和 `opponent_usage`。

恢复时会核对冻结文件的 SHA256 与顺序。同一路径下不能替换权重；启停历史池或改变冻结权重须新建实验。目前该路径只支持有序 actors，不能与 ready-first 或研究型 selfplay KV cache 混用。

三 seed、相同成功更新次数的对照及独立牌局验证已完成。本次 50% 固定三对手历史池未达到采纳门槛，约需 1.9 倍训练墙钟，因此不推荐默认启用。结论仅适用于此次固定历史池，见 [实验报告](docs/historical-opponents-2026-10-01.zh-CN.md)。

## 8. Docker（可选）

宿主已安装 Docker 与 NVIDIA Container Toolkit 时，可用相同固定依赖构建。GPU reservation 配置采用 Docker 官方接口。[S9]

```bash
docker compose build

DOUGPU_UID="$(id -u)" DOUGPU_GID="$(id -g)" \
  docker compose run --rm trainer doctor --output reports/hardware-docker.json

DOUGPU_UID="$(id -u)" DOUGPU_GID="$(id -g)" \
  docker compose run --rm trainer prepare \
    --config configs/rtx5070_balanced.json --run-dir runs/5070-docker

DOUGPU_UID="$(id -u)" DOUGPU_GID="$(id -g)" \
  docker compose run --rm trainer preflight --run-dir runs/5070-docker

DOUGPU_UID="$(id -u)" DOUGPU_GID="$(id -g)" \
  docker compose run --rm trainer train --run-dir runs/5070-docker --hours 12
```

工程目录映射到 `/app`，数据和缓存留在宿主。原生与 Compose 使用同一项目 GPU 锁目录。基础镜像固定到官方 Python tag，未通过镜像 digest 锁定整个 OS。

2026-10-01 已在 LocalServer 的 RTX 5070 完成 CPU/GPU 回归修复、三个训练 seed 的留出评估和半小时连续运行。NVRTC 固定为 13.1.115；工程检查通过，但角色棋力并非单调提高。实测结果与边界见 [LocalServer 后续验证报告](docs/training-fixes-2026-10-01.zh-CN.md)。

后续实现了分角色置信区间、角色非劣性晋升门槛、历史选择 seed 隔离与默认 best 评估，见 [模型选择验证报告](docs/model-selection-2026-10-01.zh-CN.md)。接入历史冻结对手后，完成了 CPU/GPU 各 220 项回归、三个 seed 的匹配对照、一小时 PSS 与恢复检查，以及 40,000-update 扩展评估。历史组未达到采纳门槛；扩展阶段没有新冠军，追加预算未改善 best，见 [历史对手与扩展训练报告](docs/historical-opponents-2026-10-01.zh-CN.md)。

## 9. 开发验证

只有 CPU 的验证环境仍须为 Linux x86_64 或 WSL2，安装器的 `--cpu` 不会解除平台与 Python 3.12 限制。请在没有 CUDA JAX plugin 的独立项目副本中执行；若当前 `.venv` 已安装 GPU plugin，安装器会拒绝将其转换为 CPU-only 环境，不要为运行这段示例删除正式训练环境：

```bash
bash scripts/install.sh --cpu
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

PYTHONPATH="${PWD}:${PWD}/vendor" JAX_PLATFORMS=cpu \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests

python run_local.py prepare --config configs/cpu_smoke.json --run-dir runs/cpu-check
python run_local.py train --run-dir runs/cpu-check --cycles 2
```

CPU smoke 使用显式测试引擎和小模型，只验证代码流程。GPU 专用测试在无 CUDA 时跳过；不要把 CPU 测试耗时、合成预检吞吐或旧 TPU 日志作为 RTX 5070 训练实测数据。

2026-10-01 改名后验证：在 macOS arm64 的独立临时 Python 3.12.14 / JAX 0.7.2 CPU 环境中，当前 `tests/` 为 **231 passed、1 skipped**，跳过项为真实 CUDA 检查。另通过 100 局 DouZero 规则自检、新包两周期 smoke，以及改名前 `doutpu` 包生成的 CPU 完整 checkpoint 导入与续训；导入时参数、Adam、冠军、replay、主 RNG 和计数逐项不变，成功更新从 2 继续到 3，源归档未改写。这只验证改名后的代码与状态兼容性，不扩展安装器支持平台，也不是新的 GPU 性能或棋力验收。

## 10. 许可与来源

DouGPU 原创代码、文档及本地修改采用 [MIT License](LICENSE)，保留原 DouTPU-LM 版权声明。附带的 `vendor/douzero/` 与 `upstream_cache/upstream_source.zip` 使用上游 [Apache License 2.0](vendor/LICENSE)，不被本项目 MIT 重新授权。来源、固定提交与再分发要求见 [第三方声明](THIRD_PARTY_NOTICES.md)。第三方依赖及另行取得的权重须遵守各自许可，本项目许可证不授予它们的使用权。

项目名称统一为 **DouGPU**，硬件型号是配置目标，不属于项目名。上游 DouZero、来源 notebook 的 DouTPU 名称、原版权声明、历史实验路径和原始证据哈希保留；它们标识第三方或历史来源，不是当前项目名称。`docs/TPU_ORIGINAL_*` 与 `docs/notebook_source_manifest.json` 是原始来源记录，不随项目改名重写。

当前 Python 包为 `dougpu`，直接模块命令使用 `python -m dougpu.train`、`python -m dougpu.evaluate` 等；推荐工作流仍为 `run_local.py`。旧源码锁和 checkpoint 的 `doutpu_sha256` 字段名保留兼容，含义仍是训练源码哈希。改名改变了源码哈希，已有实验须用 `prepare` 建立新目录，再按第 3 节导入完整 checkpoint；不要改写旧实验的来源锁以绕过校验。Docker Compose 镜像名统一为 `dougpu:local`。

## 参考资料

- [S1 JAX 安装与平台支持](https://docs.jax.dev/en/latest/installation.html)
- [S2 CUDA 13.0 Update 2 组件和驱动](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html)
- [S3 NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [S4 Microsoft WSL 文件存储性能建议](https://learn.microsoft.com/en-us/windows/wsl/filesystems)
- [S5 JAX 持久编译缓存](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
- [S6 JAX 基准测试](https://docs.jax.dev/en/latest/benchmarking.html)
- [S7 JAX GPU 显存分配](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
- [S8 JAX dot_product_attention](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html)
- [S9 Docker Compose GPU](https://docs.docker.com/compose/how-tos/gpu-support/)

核对日期：2026-10-01。各验证记录仅适用于报告中的源码快照、配置与环境，不等于当前版本已重新完成全部实测。原 notebook 文档位于 `docs/TPU_ORIGINAL_*`，仅保留为历史来源，不能作为本 GPU 迁移的实机验证结果；历史文件哈希也不是当前工作区的完整性清单。
