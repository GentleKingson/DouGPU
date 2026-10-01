# DouTPU-LM: 单 TPU v6e-1 / Colab 训练执行层优化

> **第九轮稳定发布，仅纳入日志压缩：** 使用 `DouTPU_v6e1_Optimized.ipynb`。发布树 105 项本地测试通过；新压缩快照在真实 v6e-1 上跨进程重启并续训至 Adam step 70,744，补齐第八轮缺项。第八轮同一源快照 ZIP 体积减少 37.05%，Colab 本地模拟保存耗时增加 2.13%；这是存储取舍，不是训练提速。首块 selector 候选通过数值与恢复检查，完整周期吞吐观测 +1.45%，未达预设 2% 门槛，不在本发布源码中。评估数组路径和 scan 展开也不启用。第七轮回退 notebook 单独保留。

> **本次修订的验收边界：** 以下原版 FullStage 数字是历史背景，不是本次改动的实测结果。
> 新 notebook 为 `DouTPU_v6e1_Optimized.ipynb`；本次结果见随附 `Optimization_Review.md`。
> 新的推理分桶、单块融合和 learner remat 覆盖都是独立开关。第一轮真实权重 TPU 快速测试中，分桶出现动作分歧，推理候选均变慢，no-remat 未见收益，全部维持关闭。
> `padding_cycle` 是全采样周期的加权计数，原 `padding` 字段仍保持最后一波的含义。
> 第二轮复用单个学习阶段内的 Replay 角色索引，不跨周期缓存；新增 `cycle_end.full_cycle_seconds`，包含周期内评估与保存。旧计时字段保留，不能相加。
> 第二轮主机采样/分组耗时 -58.5%，端到端短窗 -0.66%、较长窗口 +0.84%，尚无稳定提速证据。59 项回归及同终点 TPU 状态一致性验证通过；详见随附报告。
> 第四轮让默认有序 actor 的 packed arrays 直接用于推理准备，避免逐条拆包再重组；恢复成功时跳过无用的随机参数初始化。模型、有效 batch、Adam 更新次数、评估和保存规则不变。当轮结果见 `Optimization_Review.md` 的第四轮章节，不能沿用旧轮次的 TPU 结论。
> 第五轮只在 actor 已完成样本的传输包中省去未使用的 token 尾部列；Replay 写入时补回零，存储及训练张量仍保持原格式。当轮结果见报告第五轮章节；样本载荷减少不等于整轮训练同比提速。
> 第六轮手牌计数复用现有有界只读计数缓存，减少重复 Counter 构造，输出数组仍独立，不向公开输入加入 oracle 信息。缓存上限仍是 32,768 项，但实际驻留与内存可能增加。当轮实测见报告第六轮章节。
> 第七轮验证 action_chunk=1024 候选，102 项本地测试和 14 项 v6e-1 环境测试通过，真实快照状态对照与恢复一致。完整周期吞吐观测 +1.93%，未达到事先设定的 2% 默认启用门槛，故默认仍为 2048，不改变第六轮训练运行时和 v7 实验目录。新增分块边界回归；最新结果见报告第七轮章节。诊断 trace 含编译且区间覆盖不完整，不代表稳态利用率。
> 第九轮测试主体 680.15 秒，共 4,360 次成功 TPU 测试更新；21 项针对性 TPU 测试通过，不等于全套 105/109 项均在 TPU 运行。缓存诊断中同 VM 重启出现 5 次持久缓存命中，仅耗时约 0.24 秒的 selector 因默认 1 秒阈值没有写缓存；冷/热三周期进程为 108.05/52.51 秒。该诊断不是新增缓存功能或稳态吞吐收益，默认缓存阈值不改。完整证据、模型一致性和限制见报告第九轮章节。

**更新日期：2026-09-29。** 此代码是独立实现，不是三个原项目的官方移植，也不兼容其模型权重。采用 DouZero 的 CPU 规则引擎、DMC 终局价值回归、DanLM 式公开历史因果序列建模，以及受 FableDan 启发的辅助任务与集中批量推理。

**验证边界：** 已在 Colab 单卡 TPU v6 lite、JAX 0.7.2 上做真实快照续训、数值筛选和性能诊断；最终数字、Drive 保存验收及限制以随附 `V6e_FullStage_Report.md` 为准。本地 `reference` 引擎只用于软件测试，不声称与 DouZero 全规则等价。吞吐、有限性和短期配对评测不是长期棋力保证。

## 1. 在 Colab 使用

本版使用 `DouTPU_v6e1_Optimized.ipynb`，选择 TPU v6e-1 运行时。第一次需要本人完成 Google Drive 授权。Notebook 已内嵌源代码，无须另上传 ZIP。默认读取 `doutpu_lm_v1` 的校验快照，复制到新的 `doutpu_lm_v6_optimized_v9`，不覆盖原实验。已在旧优化版继续训练时，把 `SOURCE_EXPERIMENT` 改为实际源实验（例如 `doutpu_lm_v6_optimized_v7` 或候选 v8），仍迁移到独立 v9 目录。本轮测试未挂载生产 Drive，VM 本地模拟保存的结果不代表真实 Drive I/O 性能。

默认 `RUN_MODE="train"`，训练进程单次主动停止预算为 5.5 小时；预检和安装不计入此预算。`RUN_MODE="smoke"` 使用相同网络和 TPU 数学路径，只运行少量采样与更新，用于首次验收。换回 train、保留同一实验名即可继续。

Notebook 顺序：解包代码 → 挂载 Drive → 在独立子进程检测 JAX/TPU → 首次下载并锁定 DouZero → CPU 规则自测/单元测试 → TPU 预检 → 自动恢复/训练 → 分代保存、导出模型 → 展示日志图表。

`EXPERIMENT` 是持久化实验标识。**重连时保持相同名称**，再次运行整个 Notebook。修改网络、目标权重、学习率等训练语义时使用新实验名；代码会拒绝不兼容恢复，避免混合实验。每次运行会保存配置、代码哈希、上游 SHA、实际 Python/JAX/NumPy 版本。

Colab 分配的硬件与时长不保证固定；代码不能自动分配 TPU、代替 Drive 授权、在虚拟机销毁后自行重启，也不包含保活脚本。中断保护只能尽力执行，不能拦截 SIGKILL 或平台直接回收。

## 2. 范围与规则

本版是**已经确定地主之后的出牌阶段**：20/17/17 张牌、公开三张底牌、地主→下家→上家循环，使用 DouZero 合法动作生成器。优化胜负 WP，不训练叫牌、抢地主、加倍、春天计分、商业平台协议或完整比赛积分。

地主获胜时各角色回报为 `[+1,-1,-1]`，任一农民获胜时为 `[-1,+1,+1]`。两个农民共享团队目标，不能把农民队友先出完牌标成另一个农民输。训练仅入库完整牌局；没有 bootstrap、目标网络、PPO、MCTS、TD-λ 或联盟训练。

`reference.py` 是离线单元测试工具，不应拿它训练的权重宣称通过官方规则验证。正式 Notebook 不会静默回退到它；快照也禁止跨引擎恢复。

## 3. 默认网络与目标

- 公开历史词表 32、最大长度 512；3 层 causal Transformer，宽度 128、2 个 64 维注意力头、SwiGLU FFN 512，RMSNorm、RoPE、QK 归一化。
- 自家牌 15 维计数、三方余牌数 3 维、角色 3 维，共 21 维状态；候选动作 15 维计数加 PASS，共 16 维。
- 历史编码一次，融合自家状态后进入小型候选动作 MLP；共用骨干，最终有地主、下家、上家三个线性标量头。不是三个独立大模型。
- 原始参数总量 **922,401**。主干矩阵运算可用 BF16，权重、AdamW 状态、规范化计算、最终 Q 投影与损失保持 FP32。
- 损失：`MSE(Q, terminal_return) + 0.02 * NTP + 0.05 * belief_MSE`。NTP 右移标签、因果遮罩并忽略 PAD；辅助损失先按每条序列归一化，再按角色重加权。
- belief 输出另外两家各 15 个牌点的归一化数量。真实手牌仅作训练标签，不进入 policy API。推理导出移除 NTP/belief 参数；belief 不是满足所有约束的联合发牌后验。

历史格式为 `[BOS, BOTTOM, 三张底牌, END]`；每个出牌动作 `[角色, 逐张牌, END]`；PASS 为 `[角色,PASS,END]`。完整牌局中非 PASS 动作不超过 54 次，PASS 不超过 106 次，故保守上界 `6+54+2*54+3*106=486`，512 足够。依赖于上述出牌阶段规则；扩展叫牌等事件后需重新审计。超过上限直接报错，不截断。

当前自家手牌不注入过去每个历史 token 的 NTP 分支，以免“用现在的信息预测过去”造成不当的辅助任务泄漏。状态和动作仍有明确编码设计，因此不称为绝对“零领域知识”。

## 4. CPU / TPU 调度

类的兼容默认仍为 2 个 spawn worker × 32 局。v6e 推荐配置使用 32 × 8 局、2 个有序 actor 组、推理 batch 128、单批 learner 预取。仅主进程持有 TPU；CPU worker 不初始化 JAX。CPU 配额不足时 notebook 联合降低 worker、总局数与推理 batch，不声称低配预设也经过完整 TPU 搜索。

推理：历史长度只允许 64/128/256/512 四个桶；候选动作扁平化，每块 2048 条，并带 owner 索引和有效遮罩。所有块都参与设备端 argmax，一次回传动作索引与有限性检查。超过一块继续计算，**没有最大合法动作数截断**。同一状态的 Transformer 不随动作数反复执行。

有序 actor 按原 worker 顺序合并数组，连续分组仍为 packed views；进入 JIT 前使用相同 dtype、形状、padding 与合法动作顺序。不会提前复写仍被引用的数组，不引入共享内存。KV/ready-first 仍使用原请求列表，研究性 history 重排使用原准备路径。

学习：微批 32 × 累积 8，有效 batch 256；`lax.scan` 执行累积，保留 remat。`history_groups=4` 对已经抽出的同一批样本按长度排序，再分组裁去 PAD；不重新采样，保留全局角色权重、逐序列 NTP 归一化和一次 Adam。每周期仍独立调用四次 AdamW，`lr=1e-4, decay=1e-5, clip=1`。分组改变浮点归约顺序，不承诺 BF16 逐位相同，且增加首次编译成本。

Notebook 在 smoke、训练预算少于 30 分钟或低 CPU 配额时保守选择 `history_groups=1`；32 worker 的长会话使用 4。30 分钟是基于编译摊销的保守规则，不是自动调参得到的全局最优阈值。

回放 65,536 条，紧凑数组约 **41.375 MiB** 主机内存，不含环境和编译缓存。按三角色平衡采样并矫正整除余数。开启 `sample_credit` 时，每条完成样本增加额度，每次成功更新扣除 512；终局超额结转，不丢弃长局，长期学习比率约为 0.5。旧版不结转的实测约为 0.466，二者不能直接按 frames/s 宣称同强度加速。`replay_version_updates` 使用成功更新计数记录行为版本，年龄 64 周期对应 256 次更新。旧快照转换必须有完整历史日志证明每周期成功更新数恒定，否则明确拒绝。探索策略和 replay 采样算法不变。

`ready_first`、`actor_learner_overlap`、`selfplay_kv_cache`、`eval_kv_cache` 是保留的研究对照，推荐配置全部关闭。ready-first 已测无收益；训练权重的 BF16 KV 在 TPU 上出现动作分歧，不能视为严格等价的优化。不要仅因选项存在而启用。

权重在一轮采样阶段保持固定，跨周期尚未结束的牌局可能混合版本。旧数据的 Monte Carlo 目标不是对“最新策略 Q”严格无偏的估计；小回放、年龄限制、受控更新次数只是工程缓解，并非理论修复。

## 5. 保存、恢复与导出

Drive 默认目录：

```text
MyDrive/DouTPU/<EXPERIMENT>/
  source/source_lock.json       # 上游 commit + 逐文件哈希 + 本项目源码哈希
  source/upstream_source.zip    # 小型规则引擎归档，含上游许可证
  code/doutpu_source.zip        # 本次运行源代码
  configs/config_<time>.json
  preflight.json
  state/
    ckpt_<cycle>_<step>_<time>.zip
    ckpt_<cycle>_<step>_<time>.ok.json
    latest_policy.npz
    best_policy.npz
    metrics.jsonl
```

快照包含参数、AdamW 一二阶矩与步数、冠军参数、回放和写指针、主 RNG 状态、计数器、配置、版本、源码标识及日志。使用 NPZ/JSON，不载入 Python pickle。保存先在本地打包，再复制到 Drive 并校验 SHA-256，最后写提交标记。只从校验通过且具有提交标记的世代恢复；最新损坏时回退。保留最近三代有效快照。

第八轮候选仅对 ZIP 内的 `metrics.jsonl` 使用 DEFLATE level 1；已压缩的 NPZ 和其他条目仍使用原来的 STORE。日志内容、manifest、外层 SHA 与提交规则不变，不删日志、不改变保存间隔、不引入异步写入。体积下降不等于真实 Drive 写入或完整训练同比提速。

默认每 300 秒在安全点检查保存，长编译、环境调用或远端写入会延迟，**不是严格的最多丢失五分钟保证**。会话开始、正常结束、可捕获错误时也保存。Drive 同步失败会明确告警，最后一次同步失败会让任务以失败状态结束，不能把本地成功误报成云端成功。

恢复不保存仍在进行的 actor 牌局或其逐步 RNG 状态。它们被重新发牌；已完成的回放与优化器继续使用。因此不是逐 bit 可复现续跑，也不会保证更换 JAX/TPU 后数值完全一致。

`latest_policy.npz` 是最后状态；`best_policy.npz` 是通过内部对局筛选的冻结冠军，未发生晋升前仍可能是初始模型，不能把文件名当作模型强度保证。

编译缓存仅存 `/content` 本地；没有把大量小缓存文件同步 Drive。换 VM 后仍可能重新编译。只运行自己信任的源码、模型和缓存。

## 6. 评测

内部每 25 周期用 32 个固定牌局，各打两次：当前模型当地主对冻结冠军双农民，再反过来。报告地主胜率、农民团队胜率、二者均值，并按**牌局对**而非独立单场 bootstrap。还单独测简单规则对手。小样本指标用于工程监控，不是强度认证或正式统计检验；重复用同一组牌挑冠军也会过拟合。

只有样本数至少 32、配对 bootstrap 的下界大于 0.5 时才晋升。真实比较应使用未用于选模型的新种子、更多固定牌局、多个独立训练种子，并与有明确版本的强基线对战。不能用自博弈地主胜率变化、loss 下降或战胜随机策略证明战胜 DouZero。

可选官方 DouZero 对手接口已提供，但本地没有官方预训练权重，**此接口尚未用真实权重端到端验证**。放入可信的三个 state_dict：`landlord.ckpt`、`landlord_down.ckpt`、`landlord_up.ckpt`；额外需要 PyTorch。它只用于单独评测，不是训练依赖，也不会自动下载不明权重。

```bash
PYTHONPATH=.:vendor python -m doutpu.evaluate \
  --policy /path/to/latest_policy.npz --engine douzero \
  --opponent douzero --weights /path/to/douzero_WP \
  --deals 1000 --seed 910001 --output evaluation_holdout.json
```

该对手调用上游公开观测编码和原始模型，以 CPU 推理；使用 `torch.load(..., weights_only=True)`，不回退到任意 pickle 反序列化。结果记录两方权重哈希。没有投放到线上平台的协议实现。

## 7. 普通 Python / 离线验证

```bash
# CPU 测试，不代表实际 TPU 部署：
python -m pip install -r requirements.txt pytest
JAX_PLATFORMS=cpu python -m pytest -q tests
python -m doutpu.selftest --engine reference --games 200
JAX_PLATFORMS=cpu python -m doutpu.train \
  --config configs/cpu_smoke.json --workdir /tmp/doutpu_local \
  --savedir /tmp/doutpu_persistent

# 真实规则引擎首次下载与验证：
python bootstrap.py --savedir /path/to/persistent/source
PYTHONPATH=.:vendor python -m doutpu.selftest --engine douzero --games 100

# 单 TPU，准备正确 JAX/libtpu 运行时后：
PYTHONPATH=.:vendor python -m doutpu.train \
  --config configs/tpu_v6e_fullstage.json --workdir /tmp/doutpu_local \
  --savedir /path/to/persistent/state \
  --source-lock /path/to/persistent/source/source_lock.json
```

保留可用的 Colab JAX/TPU 组合，不盲目升级。仅在栈缺失、初始化失败或没有发现 TPU 时，Notebook 尝试一次官方 `pip install -U "jax[tpu]==0.7.2"` 并重新开启子进程检测；可用栈版本不同则警告本轮结果不能直接套用。不会同时混装 torch_xla。Notebook 内核不初始化 TPU，避免它和训练子进程争用。

## 8. 调优和局限

优先查看日志 `inference_seconds / actor_seconds / learner_seconds`，以及 `action_padding_fraction / history_padding_fraction / state_padding_fraction`，而不是只看 TPU 峰值算力或设备利用率。日志的 padding 统计来自周期最后一次推理波次，不是全周期加权平均。

环境慢：增加有限的 CPU worker 或优化规则引擎，但不要假设 Colab 有完整 TPU VM 的 CPU/RAM。推理慢：先增加并发牌局改善批量，再尝试更小动作块减少 padding；所有改动都要实测。学习显存不足：先减小 micro_batch、相应增加 accumulation 保持总批量；不要把梯度累积误当多芯片并行。

本轮额外测试了有界 ready-first、CPU/learner 重叠、版本化 KV 和 TPU Splash 原型，只有通过数值、真实入口收益与资源门槛的路径才进入推荐配置。共享内存、后台 Drive 保存和全规则引擎重写仍按 profile 决定是否值得投入，不把它们当作已实现功能。当前没有 pmap、8 芯片分布式、逐步 actor 现场恢复、Botzone 提交器或叫牌策略。

## 9. 源码导览

`environment.py`/`encoding.py`：规则适配与公开信息边界。`actors.py`：CPU ring 与完整局样本。`model.py`：原生 JAX 模型/目标/AdamW。`inference.py`：有限形状、动作分块。`replay.py`：紧凑角色平衡回放。`checkpoint.py`：校验世代与导出。`evaluation.py`/`evaluate.py`：配对评测、可选官方基线。`train.py`：单 TPU 生命周期。`bootstrap.py`：来源锁定。`preflight.py`：实际设备张量编译与合成吞吐基准。

本地验证记录在 `validation/`。其中极少量自博弈、对局胜率和时间只说明软件链路跑通，不能外推到 TPU 性能或正式规则下的实力。

## 10. 研究来源与许可

检索日期 2026-09-27，细节以各仓库检索时公开文件为准。DanLM 公开说明已经包括 DouLM，但未取得足够完整的训练源码来核验所有训练超参数，不能将 FableDan 默认配置反推为 DanLM 配置。DanLM 与 FableDan 的 LICENSE 含额外非商业限制；本包未复制其源码或权重。运行时下载 DouZero 并保存其原始 LICENSE/NOTICE。用于商业场景前需另行核对所有依赖和权重的授权。

- DanLM README: https://github.com/dashidhy/DanLM
- FableDan: https://github.com/lrx0716/FableDan
- FableDan model: https://github.com/lrx0716/FableDan/blob/main/fabledan/model_torch.py
- FableDan trainer: https://github.com/lrx0716/FableDan/blob/main/fabledan/train_fast.py
- FableDan encoder: https://github.com/lrx0716/FableDan/blob/main/fabledan/encode.py
- FableDan roadmap: https://github.com/lrx0716/FableDan/blob/main/DESIGN.md
- DouZero paper: https://proceedings.mlr.press/v139/zha21a.html
- DouZero: https://github.com/kwai/DouZero
- TPU v5e: https://docs.cloud.google.com/tpu/docs/v5e
- JAX installation: https://docs.jax.dev/en/latest/installation.html
- JAX JIT: https://docs.jax.dev/en/latest/jit-compilation.html
- Colab FAQ: https://research.google.com/colaboratory/faq.html
