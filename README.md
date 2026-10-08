# DouGPU

DouGPU 是从 `DouTPU_v6e1_Optimized_Final.ipynb` 迁移过来的 GPU 版本，目标机器为 Intel Core Ultra 5 245KF、桌面版 NVIDIA RTX 5070 12 GB 和 32 GB 内存。训练基于 JAX：GPU 由单个进程独占，DouZero 牌局模拟和特征编码则交给多个 CPU actor 完成。

迁移过程中，原模型结构、公开信息边界、DMC/NTP/belief 训练目标、完整合法动作集以及完整 checkpoint 格式都保持不变。默认采用 balanced 基线配置；另有一份在 LocalServer 的 RTX 5070 上实测调出的吞吐配置，可按需选用。架构设计、硬件依据、迁移中的取舍和验收范围写在[迁移报告](docs/gpu-migration.zh-CN.md)中，具体测试过程见[验证记录](docs/validation-notes.md)。

就目前的验证而言，模型具备基本的学习能力，在限定时长内运行稳定；但能否达到强 DouZero 基准的水平，现有结果还不足以下结论。训练默认采用自博弈，历史冻结对手池默认关闭。

## 1. 环境与安装

本项目可在原生 Linux x86_64 或 Windows 的 WSL2 Ubuntu 下运行。JAX 官方对 WSL2 上 NVIDIA GPU 的支持目前仍标注为 experimental；原生 Windows Python 则无法运行本项目的 JAX CUDA 训练。[S1]

安装脚本不负责安装 Python，需事先自行装好 Python 3.12。如需指定解释器，可执行 `bash scripts/install.sh --python /path/to/python3.12`。

依赖版本方面，JAX/JAXLIB/plugin/PJRT 固定为 0.7.2，cuDNN 固定为 9.18.1.3。CUDA 组件大体沿用 13.0 Update 2，只有 NVRTC 单独固定在 13.1.115，用来解决本机 cuDNN attention 运行时编译失败的问题，所以这套依赖与原版 CUDA 13.0 Update 2 并不完全一致。22 个核心依赖的版本和 wheel 记录见[依赖清单](docs/dependency_resolution.json)。安装约束以 `requirements-*.txt` 和 `constraints-cuda13.txt` 为准，实际装上的版本记录在 `.venv/resolved-gpu.txt`。CUDA 库通过 pip wheel 获取，系统中不必安装完整的 CUDA Toolkit。

原生 Linux 下，驱动需要同时支持 RTX 5070 和 CUDA 13。按照 JAX 文档，CUDA 13 须搭配 580 系列或更新的驱动。CUDA 13.0 Update 2 配套的 Linux 驱动是 580.95.05，但本项目混用了不同版本的组件，也没有逐一测试全部内核，因此这个版本号不能视为实测的最低要求。我们做 GPU 验证时用过 595.71.05 和 595.91.07；在你自己的环境中，仍应先通过 `doctor` 和 `preflight` 两项检查。WSL 直接使用 Windows 宿主机的 NVIDIA 驱动。[S1][S2][S3]

将项目解压到 WSL/Linux 的本地磁盘后执行：

```bash
# 替换为实际项目目录；旧检出目录 DouZero 无须为改名而移动。
cd /absolute/path/to/DouGPU
bash scripts/install.sh
source .venv/bin/activate

python run_local.py doctor --output reports/hardware.json
```

安装器会在项目目录下新建或复用一个 Python 3.12 的 `.venv`，并生成 `.venv/resolved-gpu.txt`，环境中原有的其他包不会被自动清理。`doctor` 的输出中必须出现 `backend: gpu`；它会实际执行一次 BF16 矩阵乘法，并列出检测到的 GPU、驱动和依赖版本。如果找不到 CUDA，检查直接判为失败，而不会退回 CPU 运行。

在 WSL 下，项目、`.venv`、编译缓存以及 replay/checkpoint 都应放在 Linux 文件系统中（例如 `~/DouGPU`），涉及大批文件读写时尽量使用 Linux 路径。[S4] 机器有 32 GB 内存时，可视需要给 WSL 分配 24 GB 左右；CPU 方面先查看实际分到的配额即可，不必在配置中写死 P 核、E 核编号。另外，请勿在 WSL 内部安装 Linux 版 NVIDIA 显示驱动。[S3]

如果 shell 中已经设置了 `LD_LIBRARY_PATH`，请确认它没有覆盖 pip 安装的 CUDA/cuDNN 库，`doctor` 也会记录这一情况。[S1] 本项目不会自动改动宿主机驱动或系统环境。

## 2. 建立独立本地实验

```bash
python run_local.py prepare \
  --config configs/rtx5070_balanced.json \
  --run-dir runs/5070

python run_local.py selftest --games 100
```

仓库中已附带 DouZero 规则源码（固定在提交 `718a5c920bf3361e34178a38f3b80458e176b351`）及其许可证。`prepare` 会校验这些规则文件的字节内容，并为新的训练代码生成来源锁，所以首次启动时不需要临时追踪 GitHub 主分支。`selftest` 只检查规则和编码，不会初始化 JAX。

默认配置如下：

| 配置项 | 本地起点 | 说明 |
|---|---:|---|
| CPU workers | 8 | 给主进程、JAX 调度和系统留出余量 |
| 每 worker 牌局 | 8 | 同时进行的牌局共 64 局 |
| actor groups | 2 | 每组 4 workers、32 局，合并时保持顺序 |
| inference batch | 32 | 与单组牌局数一致 |
| action chunk | 2048 | 超出时继续分块，所有动作都会评分 |
| micro batch × accumulation | 32 × 8 | 有效 batch 固定为 256 |
| history groups | 1 | 先减少静态形状组合，2/4 组可单独测试 |
| learner remat | inherit | 沿用原 ModelConfig 中的 remat=true |
| precision | BF16 主干 / FP32 参数与优化器 | 沿用原混合精度策略 |
| replay | 65,536 条 | 保留回放及角色权重的语义 |
| fresh samples / updates | 2048 / 4 | 名义学习样本复用比为 0.5 |
| evaluation | 每 25 周期、每对手 256 对验证牌局 | 用于分角色晋升检查，不作为最终棋力结论 |
| checkpoint | 300 秒、保留 3 代 | 在下一个安全点保存，可能因编译或正在执行的阶段而推迟 |

如果希望减少 CPU 争用，可改用 `configs/rtx5070_conservative.json`（6 workers × 8 局）。这份配置同样只是一个起点，并未证明是最快的。

`configs/rtx5070_throughput.json` 是 2026-10-01 在 LocalServer 上实测得到的可选配置，改动包括：启用 cuDNN attention、learner remat=false、micro batch × accumulation 设为 64 × 4、2 个历史分组、每 worker 16 局，以及 inference batch=64。模型和训练目标保持不变。

我们从同一个 checkpoint 出发做了三次短测，两种配置交替运行，完整周期吞吐由 18.35 提升到 25.86 更新/秒，增幅约 41%。测试期间评估照常进行，两边的保存间隔都临时缩短为 10 秒（生产配置仍为 300 秒）。测量窗口和对应的源码版本见[吞吐验证记录](docs/validation-notes.md#5-2026-10-01-localserver-吞吐优化)。此后晋升逻辑又有修改，因此这组数字只对应当时的代码版本和机器。

该配置已通过完整 512-token 容量检查和数值预检。换到其他机器上哪种配置最优、长时间运行是否稳定、棋力能否提升，都还需要另行测试。用它建立新实验时，请先运行 `preflight`。

## 3. 选择从零训练，或继续 TPU 学习状态

### 从零训练

`prepare` 完成后可直接跳到下一节。没有 checkpoint 的新实验会从初始化模型开始训练，无法接续 notebook 输出中显示的训练进度。

### 从 TPU 完整续训

上传的 notebook 里只有源码和历史输出，参数、Adam 状态和 replay 的实际训练快照都不在其中。因此需要先把 Colab/Drive 上实验的 `state` 目录复制到本机，目录中必须有一组相互匹配的文件：

- `ckpt_....zip`
- `ckpt_....ok.json`

然后将其导入一个已执行过 `prepare`、但还没有任何 checkpoint 的新目录：

```bash
python run_local.py import-checkpoint \
  --source-state /absolute/path/to/copied_tpu_experiment/state \
  --run-dir runs/5070
```

若想复用原实验的规则缓存，可以在 `prepare` 时加上 `--upstream-cache /absolute/path/to/copied_tpu_experiment/source`，该目录需包含 `source_lock.json` 和 `upstream_source.zip`。这个选项只用于复用缓存：DouZero 提交仍须与 notebook 固定的版本一致，文件哈希也必须相同，所以无法借此切换规则版本。

导入时，程序会核对整个归档及其中每个成员的哈希，并检查模型、完整参数结构、Adam step、规则源码和训练语义是否一致。参数、Adam 的 m 和 v、冠军、replay、主 RNG、各项计数器以及 sample credit 都会原样保留，来源信息写入 `performance_provenance.json`，源目录不做任何改动。

`latest_policy.npz` 和 `best_policy.npz` 只适合用于策略部署和评估。它们缺少辅助头、优化器状态和 replay，无法用来完整续训。此外，从 TPU 换到 CUDA 或改变 actor 数量后，后续训练轨迹不保证与原来逐位一致；尚未打完的 actor 牌局会重新开局。

## 4. 先预检，再做两周期 smoke，再长训

```bash
# 当前配置的模型、跨块动作评分及全 512 个非 PAD token 的合成反向检查。
python run_local.py preflight --run-dir runs/5070

# --cycles 表示本次会话额外运行的周期，不是全局周期上限。
python run_local.py train --run-dir runs/5070 --cycles 2 --hours 1

# 上一步正常后，从相同目录继续。
python run_local.py train --run-dir runs/5070 --hours 12
```

预检和训练各自独占 GPU，须先后执行。`preflight` 使用新初始化的参数和合成数据，既不加载也不改写正式训练的 checkpoint，因此不能代替真实的自博弈测试。历史分组数大于 1 时，预热只覆盖各组等宽的组合，长度混合的组合仍可能在训练中首次编译。首次编译需要一段时间。编译缓存默认存放在 `.cache/jax`，可通过 `JAX_COMPILATION_CACHE_DIR` 修改位置；之后能否命中缓存，取决于形状、代码、编译配置和设备。请勿使用不可信用户有写权限的编译缓存。[S5]

运行后请确认日志中 `successful_steps > 0`，并且 checkpoint 里的 `optimizer.step`/`updates` 在增长。两周期 smoke 只能说明训练流程可以跑通，棋力和 GPU 性能需另行评估。按下 `Ctrl+C` 或发送 SIGTERM 后，程序会请求在安全点存盘，launcher 会等 learner 保存完毕。若直接断电或强制终止进程，只能恢复到最后一次已提交的 checkpoint。

从零开始的两周期 smoke 到不了默认的第 25 周期评估点，所以验证不了冠军晋升；导入 checkpoint 后是否会触发评估，要看恢复后的全局周期数。常用的输出位置如下：

| 路径 | 内容 |
|---|---|
| `runs/5070/metrics.jsonl` | 训练、各阶段耗时、评估、保存与会话日志 |
| `runs/5070/checkpoints/` | 完整快照和 latest/best 策略导出 |
| `runs/5070/source/` | 固定规则归档、训练源码归档和来源锁 |
| `runs/5070/config.json` | 建立实验时的基线配置 |
| `runs/5070/session_config.json` | 本次会话实际使用的配置 |
| `runs/5070/preflight.json` | 目标设备预检结果与有限的合成基准 |

如需额外备份，可在 `train` 时加上 `--mirror-dir /path/to/backup`，为完整 checkpoint 再保存一份带校验的副本。每个位置都按 `keep_checkpoints` 保留最近的有效版本。本地目录本身就是持久存储。

## 5. 分阶段寻找这台机器的最快有效配置

调参前先完成 baseline 的 smoke 或短训，并正常停止该源实验。每个候选配置都从同一份冻结快照出发，依次独占 GPU 运行，源实验不会被改动，正式训练参数也不会被替换。

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

在基线相同的情况下，worker 阶段一般有 4 个候选，训练预算合计约 120 分钟，此外还要加上预检、初始化、导入和最终存盘的时间。可以先用 `--repeats 1` 粗筛，这时报告会明确标注为临时证据。只生成过计划的目录可以接着执行；已经有 trial/snapshot 的目录则不允许再次覆盖。

各阶段按下表顺序进行，上一阶段选出的候选 JSON 作为下一阶段的 `--config`：

| `--stage` | 比较项 | 保持不变的部分 |
|---|---|---|
| `workers` | 6 / 8 / 10 / 12 个 actor | 每 actor 8 局，inference 容量随之匹配 |
| `envs` | 每 actor 4 / 8 / 16 / 32 局 | 沿用已选定的 worker 数 |
| `inference` | history buckets、fused 分别开关 | 每次只改一个执行开关 |
| `chunks` | 1024 / 2048 个动作槽 | 所有合法动作仍完整处理 |
| `learner` | 16×16 / 32×8 / 64×4 / 128×2 | 有效 batch = 256 |
| `history` | 1 / 2 / 4 组 | 只对同一批样本重新分组并裁去 PAD |
| `remat` | 重计算开/关 | 原模型配置与学习目标不变 |
| `attention` | manual / cuDNN；XLA 仅作诊断 | 因果性、有效历史和辅助目标不变 |

并非每个阶段都要跑完。可以先看 collection、learner、编译和存盘各自的耗时，挑占比最高的环节去测。只有实测胜出的配置才用于后续的长期训练。

### 读懂测量结果

主指标定义为：

```text
sum(successful optimizer updates) / sum(full_cycle_seconds)
```

完整周期的墙钟时间包含正常评估和周期内的保存；进程启动、预检、导入以及会话结束时的最终存盘另行统计，所以主指标并不等于整条命令的吞吐。报告中还给出样本行/s、采样决策/s、learner-only 吞吐、padding，以及测量段内首次出现的形状，比较时需注意各项的统计口径并不相同。JAX 调用必须先预热，并在设备端计算完成之后再计时。[S6]

工具默认忽略本会话的前 25 个周期，导入日志中的旧会话也不计入。若测量段内缺少完整评估、周期保存、有效更新或真实的新采样，或者出现非有限步、进程失败，该 trial 不参与有效排名。预算过短时结果会显示为 `unrankable`，此时应延长单次运行时间。由于 sample credit 的存在，个别周期可能只学习不采样；工具按整个窗口累计计算，以免这类周期把吞吐算高。

报告写入 `benchmarks/workers/summary.json`。三次重复只能在当前这组候选之间做比较，既不能证明全局最优，也不能说明棋力有所提升。候选参数不会自动写回正式实验。

### Colab L4 同参短测

先用 `colab new -s dougpu-l4 --gpu L4` 创建一个独立会话，然后上传仓库快照、本地实际运行时的完整配置，以及同一个完整 checkpoint（ZIP 和 `.ok.json`）。在 Colab 中需另建 Python 3.12 环境，用仓库现有的 `requirements-gpu.txt` 安装依赖；不要使用 notebook 预装的其他版本 JAX，也不要换用另一份默认训练配置。

在上传后的仓库根目录下执行：

```bash
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements-gpu.txt
.venv/bin/python scripts/colab_benchmark.py \
  --config input.config.json --source-state input-state \
  --run-dir benchmark --seconds 720
```

该脚本沿用已有的导入、doctor、preflight、训练和统计流程，只覆盖本次运行时长，并断言其余配置和输入文件均未改变。`benchmark/summary.json` 同样排除前 25 个周期，保留真实采样、周期保存以及配置中原有的评估。预算必须足以覆盖原定的保存间隔，否则结果会被标记为不合格；脚本不会为了凑出结果去缩短保存间隔。下载日志和结果后，用 `colab stop -s dougpu-l4` 释放会话。单次短测反映的只是该配置下的整机吞吐，既不等同于纯 GPU 算力，也不能当作多次重复的基准。

## 6. 显存与注意力设置

默认情况下，主矩阵乘法使用 BF16，参数、Adam、归一化、softmax、Q 标量投影和 loss 则保持 FP32。项目不会启用 FP8/FP4，也不会缩减完整历史。

JAX 默认会预先分配 GPU 显存。本 launcher 将初始比例设为总显存的 70%，给桌面和运行时留出余量。这一数值是 allocator 的预分配比例，并非对所有 CUDA 分配的硬性上限。[S7] 运行命令前可以显式覆盖，例如：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 python run_local.py preflight --run-dir runs/5070
```

cuDNN 目前仍是有待测试的优化选项。它显式指定 `implementation='cudnn'` 并传入有效序列长度，从而避免构造显式的 `[B,T,T]` 注意力矩阵。输入必须由有效的非零前缀加右侧 PAD 组成，主机端会检查这一约定。遇到不支持的 dtype、形状或设备时程序会直接报错，不会悄悄换回其他内核。[S8]

若在较大的微批下出现 OOM，先退回上一个已通过预检的执行配置。可以继续使用原来的 `32×8`，或者试试 `16×16`；有效批量应保持在 256，不要靠缩短历史或截断动作来改变训练任务。覆盖 `learner_remat` 只影响执行方式，旧快照中的 ModelConfig 不会因此改变。

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

权重目录中需要有 `landlord.ckpt`、`landlord_down.ckpt` 和 `landlord_up.ckpt`，且须与固定版本的 DouZero 模型定义兼容。项目不会自行下载权重，缺少对手时也不会用随机策略顶替；加载时使用 `weights_only=True` 和严格的 `state_dict` 匹配。可选对手运行在 CPU PyTorch 上，被评估的当前策略仍在 JAX GPU 上运行。默认安装和 Docker 镜像都不包含 PyTorch，此前的验证是在临时评估容器中用 CPU PyTorch 2.8.0 完成的。

2026-09-30 至 2026-10-01 的验证中，我们使用服务器上已有的三份 DouZero 格式权重完成了评估，权重哈希列于[历史对手报告](docs/historical-opponents-2026-10-01.zh-CN.md#复现与证据)。由于其发布来源没有经过独立核实，不能称之为已验证的官方冠军。能够成功加载、哈希也已固定，并不代表来源可靠或已获授权，权重的出处和使用许可请自行核对。

RTX 5070 配置在内部使用 256 对验证牌局，分别记录地主胜率、农民团队胜率和均衡胜率，以及各自的 95% 配对 bootstrap 区间。晋升需要同时满足两类条件：一是对冻结冠军的均衡胜率下界高于 50%；二是两个角色都满足非劣性要求，即对冠军的角色胜率下界不低于 45%，且对固定规则对手的角色胜率变化下界不低于 -5 个百分点。后一项依据候选模型与冠军在同一批规则牌局上的结果计算，不能只比较两者的总胜率。

门槛取单侧 99% bootstrap 下界，并对每次晋升涉及的五项统计检验做 Bonferroni 校正；但如果反复挑选模型，总体误晋升率仍无法保证。`promotion_role_margin=0.05` 是事先设定的可接受回退幅度，此范围内的退步是允许的。验证牌局少于 128 对时只报告结果，不做晋升。

`run_local.py evaluate` 默认评估的是 `best_policy.npz`，而不是 latest。冠军首次晋升之前，best 仍是初始策略，所以这个文件名并不意味着已经通过了强对手验收。

新的导出会记录训练器已知的历史冠军选择 seed，导入时则合并源实验与目标实验的记录。两个评估入口都会拒绝使用导出元数据中已登记的选择 seed。若在其他方案选择中用过额外的 seed，需另行登记并写入封存元数据，训练器无法自动发现这些 seed。缺少相应元数据的旧版导出也需要人工核对。

需要说明的是，代码并不会阻止复用已经看过的测试 seed。最终留出集的结果不能再用来挑选 checkpoint 或调整门槛，示例 seed 也不能一直充当新的留出集。正式的棋力评估要求使用未参与模型选择的新牌局和固定的强对手；吞吐提高本身并不说明策略变强了。

若要做历史冻结对手实验，可在配置 JSON 的 `train` 对象中用 `historical_opponents` 指定一组策略 NPZ 路径，再用这份配置建立新实验；列表为空时即为原来的自博弈。`historical_fraction` 默认为 0.5，以整局为单位抽样。每局开始时确定历史对手和当前策略各自所在的阵营，整局不变，两名农民共用本阵营的策略。历史对手既不探索也不更新，其动作不会写入 replay。`frames` 和探索衰减只统计当前策略的出牌决策，历史对手的决策另记在 `historical_frames` 和 `opponent_usage` 中。

恢复训练时会核对冻结文件的 SHA256 及其顺序。同一路径下不允许替换权重；启用或停用历史池、更换冻结权重，都必须新建实验。目前该路径只支持有序 actors，不能与 ready-first 或研究用的 selfplay KV cache 同时使用。

我们已完成三个 seed、成功更新次数相同条件下的对照实验，以及独立牌局验证。本次采用 50% 比例、固定三个对手的历史池没有达到采纳门槛，训练墙钟时间还增加到约 1.9 倍，因此不建议默认开启。这一结论仅针对本次的固定历史池，详见[实验报告](docs/historical-opponents-2026-10-01.zh-CN.md)。

## 8. Docker（可选）

如果宿主机已安装 Docker 和 NVIDIA Container Toolkit，可以用同一套固定依赖构建镜像，GPU reservation 的配置遵循 Docker 官方接口。[S9]

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

工程目录挂载到 `/app`，数据和缓存都保留在宿主机上。原生运行与 Compose 共用项目的同一个 GPU 锁目录。基础镜像固定到官方 Python tag，但没有用镜像 digest 锁定整个操作系统。

2026-10-01，我们在 LocalServer 的 RTX 5070 上完成了 CPU/GPU 回归问题的修复、三个训练 seed 的留出评估和半小时连续运行，NVRTC 固定为 13.1.115。工程检查全部通过，不过各角色的棋力并非单调上升。实测结果及其适用范围见[LocalServer 后续验证报告](docs/training-fixes-2026-10-01.zh-CN.md)。

此后又加入了分角色置信区间、角色非劣性晋升门槛和历史选择 seed 隔离，并把默认评估对象改为 best，见[模型选择验证报告](docs/model-selection-2026-10-01.zh-CN.md)。接入历史冻结对手之后，我们完成了 CPU、GPU 各 220 项回归测试，三个 seed 的匹配对照，一小时的 PSS 与恢复检查，以及 40,000-update 的扩展评估。历史对手组没有达到采纳门槛；扩展阶段也没有产生新冠军，追加的训练预算并未改善 best，见[历史对手与扩展训练报告](docs/historical-opponents-2026-10-01.zh-CN.md)。

## 9. 开发验证

即使只做 CPU 验证，环境也必须是 Linux x86_64 或 WSL2，安装器的 `--cpu` 选项不会放宽平台和 Python 3.12 的限制。请在一个没有安装 CUDA JAX plugin 的独立项目副本中操作。如果当前 `.venv` 已装有 GPU plugin，安装器会拒绝把它转换成 CPU-only 环境；请不要为了运行下面的示例而删除正式训练环境：

```bash
bash scripts/install.sh --cpu
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

PYTHONPATH="${PWD}:${PWD}/vendor" JAX_PLATFORMS=cpu \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests

python run_local.py prepare --config configs/cpu_smoke.json --run-dir runs/cpu-check
python run_local.py train --run-dir runs/cpu-check --cycles 2
```

CPU smoke 使用显式指定的测试引擎和小模型，只用于验证代码流程。没有 CUDA 时，GPU 专用测试会被跳过。CPU 测试耗时、合成预检的吞吐以及旧的 TPU 日志，都不应当作 RTX 5070 的训练实测数据。

2026-10-01 改名之后，我们在 macOS arm64 上另建了一个临时的 Python 3.12.14 / JAX 0.7.2 CPU 环境进行验证，当前 `tests/` 的结果为 231 passed、1 skipped，跳过的一项是真实 CUDA 检查。此外还通过了 100 局 DouZero 规则自检和新包的两周期 smoke，并测试了改名前 `doutpu` 包生成的 CPU 完整 checkpoint 的导入与续训：导入后参数、Adam、冠军、replay、主 RNG 和计数器逐项一致，成功更新次数从 2 继续增加到 3，源归档未被改写。这些测试只能说明改名后的代码与旧状态兼容，安装器支持的平台并未因此扩大，也不构成新的 GPU 性能或棋力验收。

## 10. 许可与来源

DouGPU 的原创代码、文档及本地修改采用 [MIT License](LICENSE)，并保留原 DouTPU-LM 的版权声明。随附的 `vendor/douzero/` 和 `upstream_cache/upstream_source.zip` 沿用上游的 [Apache License 2.0](vendor/LICENSE)，本项目不会将其重新授权为 MIT。来源、固定提交和再分发要求见[第三方声明](THIRD_PARTY_NOTICES.md)。第三方依赖和另行获取的权重须遵守各自的许可，本项目的许可证并不授予它们的使用权。

项目名称统一为 DouGPU，硬件型号只是配置目标，不属于项目名。上游 DouZero 的名称、来源 notebook 中的 DouTPU 名称、原版权声明、历史实验路径和原始证据哈希都予以保留，它们指代第三方或历史来源，而非当前项目。`docs/TPU_ORIGINAL_*` 和 `docs/notebook_source_manifest.json` 属于原始来源记录，不随项目改名而改写。

当前 Python 包名为 `dougpu`，推荐使用 `run_local.py`。新实验的来源锁只记录规则引擎、编码版本和固定上游来源，不记录本地训练源码哈希或 Git commit。旧实验的锁（包括历史 `doutpu_sha256`）原样保留；普通源码清理不再阻止续训，checkpoint 与实验锁仍须完整相等。每次训练的启动日志及 checkpoint `versions` 尽量记录 `git_commit` 和 `git_dirty`；Git 不可用时记录 `unknown`，不阻断启动，也不用于完整性证明。`trainer_source.zip` 仅代表 prepare 时的源码快照，后续会话以各自的版本记录为准。Docker Compose 的镜像名为 `dougpu:local`。

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

核对日期：2026-10-01。各份验证记录只对应报告中所列的源码快照、配置和环境，并不表示当前版本已重新完成全部实测。原 notebook 的文档存放在 `docs/TPU_ORIGINAL_*`，仅作为历史来源保留，不能当作本次 GPU 迁移的实机验证结果；历史文件的哈希也不是当前工作区的完整性清单。
