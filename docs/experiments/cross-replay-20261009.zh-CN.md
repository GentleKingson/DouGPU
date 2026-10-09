# 冻结 Replay 跨端点 Q 误差比较

日期：2026-10-09，Asia/Hong_Kong。源码基线：`fcd1276f4d2a99b89b65489f09540cb4921c7827`。

结论：**R0_PASS；R1 数值执行 PASS；R2=RESEARCH_SCREEN_INCONCLUSIVE / STOP。G_NOT_ENTERED，TRAINING_PAUSED。** 全部结果标记 **EXPLORATORY_NUMERICAL_ONLY**。

两个参数快照均在各自端点 Replay 上具有较低 Q 误差，三个角色方向一致；跨 Replay 的差值符号相反。该交叉形态是本次观察结果，但不满足执行前冻结的“六个差值跨 Replay 同向”筛选规则。因此停止，不事后改变规则、不补抽、不进入 G。

## 资格与冻结

已先审阅角色机制、batch512、E/F 闭环报告。没有重复角色梯度冲突、batch512 梯度/Adam 探针或历史 NTP 研究；没有寻找声明已删除的旧实验目录。

两份原始阶段 ZIP 在 Mac 实际存在，并与 LocalServer 既有归档重算的外层 SHA 一致。复用旧协议读取器和 Store 验证外层 manifest、完整 checkpoint/marker、内层 manifest、参数形状/有限性、Adam.step、Replay.restore、主 RNG 可恢复性和来源快照；两端 Adam.step 分别为 2,000 / 20,000。旧协议门禁仍 FAIL。

| 原件 | 外层 ZIP SHA256 | 完整 checkpoint SHA256 |
|---|---|---|
| 2k | `fcf71d8cb9c249353c3df2bc3de6511dc6901a95e7108accf25fa5304902b99d` | `c653d73a98dd412f1df2e95088ff957d616b8bcd7e4575074d1f9545f42e5830` |
| 20k | `1702ae91c4aca6dceff455f4d96bc14b7f6b112e942905d3402f56bfdfc6a1ad` | `7ad239ebf88a1f7324a1f85d384d02ccea9a716ce02d52f4e1b8c597b87e8238` |

ModelConfig 相同：width=128、layers=3、heads=2、ffn=512、q_hidden=256、max_seq=512、bf16=true、remat=true。规则来源锁和编码 schema=1 相同；两个归档 source.zip / trainer_source.zip 中的 model.py、encoding.py 与本次冻结源码逐字节一致。params/replay/optimizer/meta 的独立 SHA 和逐数组状态摘要见 eligibility.json。

这些校验只确认数值和归档源码可读，不补足历史实际 worker seeds、恢复输入摘要或当时加载代码的执行凭证。历史 Baseline 仍 **INVALID_PROTOCOL**，历史 NTP 仍 **CLOSED / INVALID_PROTOCOL**。

两端 Replay 各 65,536 行。没有持久 sample ID 或独立牌局簇 ID，是否共享原始样本标记 **UNKNOWN**；物理索引相同不代表同一历史样本。

## 事前方案

在打开任何本次模型输出前，冻结了完整索引、脚本哈希、源文件哈希、以下口径和预算：

- 独立 PCG64 诊断 seed=93172641；不使用主 RNG、模型选优或最终留出种子。按 R₂ₖ、R₂₀ₖ、角色 0/1/2 的固定顺序，从 Replay.strata 当前合格行各无放回抽取 64 行。共 384 行，每行分别用两个模型前向，共 768 次主前向。
- CPU BF16/manual，两模型完全相同；Q 输出 FP32，MSE 在 NumPy float64 中聚合。固定 microbatch=8、完整 512-token 张量，无额外截断。训练原路径为 GPU/cuDNN，不能把不同硬件/attention 路径的数字当成逐位等价。
- 复用 encode_state、q_values；不调用 sample_losses 的辅助目标、自动微分、make_train_step、优化器、Actor、评测或训练入口。没有改任何生产文件和训练配置。
- 唯一主比较为每个 Replay、每个角色的 Δ=MSE(θ₂₀ₖ,R)−MSE(θ₂ₖ,R)。辅助记录正/负标签、序列长度、版本差及标签条件 MSE；不做 bootstrap、置信区间或棋力解释。
- 筛选规则：六个 Δ 同号且绝对值均至少 0.01，所有正/负标签条件 Δ 也同向，才记探索性信号；否则 INCONCLUSIVE / STOP。0.01 为事前选定的描述性筛选下限，不是统计显著性或未来实验采用门槛。**该保守规则不会接纳本次观察到的双向交叉形态；本报告保留该规则的这一局限，不作事后修订。**
- 同一冻结行用 θ₂ₖ 单独前向两次，额外 2 次前向；Q 和误差复算容差 rtol=1e−6、atol=1e−7。修改期望输入 SHA 必须拒绝。
- 4 CPU、4 GiB、总墙钟上限 1,800 秒、网络关闭、无 GPU 挂载；超时或任何检查失败即停止，不追加规模。

计划 SHA：`f52a26b001041513e344b478efd48c025aa7a2053f31dc9102d8135e129440c9`。
一次性脚本 SHA：`7439704f0e1de089ed0fc7b81a9541ab9a43e0b6d287526d273428317ada0ba2`。

## 四格误差矩阵，按角色分解

每个角色对应两 Replay × 两参数快照的四格矩阵，每格 64 行。同一 Replay 的两模型使用完全相同的索引和 MC 标签。

| 角色 | 固定 Replay | MSE θ₂ₖ | MSE θ₂₀ₖ | Δ（20k−2k） |
|---|---|---:|---:|---:|
| 地主 | R₂ₖ | 0.591396 | 0.989132 | +0.397735 |
| 地主 | R₂₀ₖ | 1.077272 | 0.507920 | −0.569351 |
| 下家农民 | R₂ₖ | 0.615333 | 1.233021 | +0.617687 |
| 下家农民 | R₂₀ₖ | 1.289307 | 0.535833 | −0.753474 |
| 上家农民 | R₂ₖ | 0.489368 | 1.264237 | +0.774869 |
| 上家农民 | R₂₀ₖ | 1.232604 | 0.461567 | −0.771038 |

负值只表示 θ₂₀ₖ 对这些固定 MC 标签的误差较低；正值反之。各自 Replay 是训练状态的一部分，不是独立测试集。不能把对自身 Replay 的较低误差称为泛化能力或棋力优势。

## 样本组成与解释边界

下表仅为固定抽中的 64 行；所有合格行的同类统计也保存在 cross-replay.json。

| Replay / 角色 | 正 / 负标签 | 正标签比例 | 序列长度 P50 / P90 / P99 | 版本差 P50 / P90 / P99 |
|---|---:|---:|---:|---:|
| R₂ₖ 地主 | 42 / 22 | 65.625% | 79 / 150 / 191.07 | 64 / 122.8 / 132 |
| R₂ₖ 下家 | 30 / 34 | 46.875% | 70 / 143.5 / 157 | 64 / 108 / 129.48 |
| R₂ₖ 上家 | 31 / 33 | 48.4375% | 96 / 159.9 / 181.7 | 66 / 120 / 133.48 |
| R₂₀ₖ 地主 | 41 / 23 | 64.0625% | 60 / 116 / 155.4 | 72 / 124 / 132 |
| R₂₀ₖ 下家 | 28 / 36 | 43.75% | 67.5 / 137.4 / 169.77 | 68 / 124 / 129.48 |
| R₂₀ₖ 上家 | 21 / 43 | 32.8125% | 68 / 125.4 / 155 | 70 / 122.8 / 136 |

标签全部为 ±1。两端 replay_version_updates=true，版本差单位是成功更新数，资格窗口 256；不是秒数或策略 KL。行数和正标签比例不是独立完整牌局数或胜率。

正/负标签分层的 Δ 均保留对应 Replay 总体 Δ 的方向：R₂ₖ 为正，R₂₀ₖ 为负。这个固定样本结果并非仅由正负标签混合比例翻转；但分层后状态、动作、序列长度、后续行为策略仍未配对，没有反事实标签，无法单独识别覆盖变化、标签迁移、遗忘或策略滞后的因果贡献。

| 分级 | 结论 |
|---|---|
| OBSERVED | 三角色均出现各自端点 Replay 误差较低的交叉形态，标签分层方向保持 |
| NOT_MEASURED | 独立牌局簇、原始样本跨端点身份、反事实回报、策略 KL、跨 seed 稳定性、棋力 |
| INCONCLUSIVE | 单一可干预机制、因果归因、未来受控实验资格；冻结筛选规则未通过 |

## 执行与可复算性

既有镜像 `sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e`。
Python 3.12.14 / JAX 0.7.2 / NumPy 2.5.3，实际 backend=cpu。单次脚本耗时 **6.742 秒**，退出码 0，未触及预算。标准错误含既有 CUDA 插件无设备探测信息；没有 GPU 设备挂载或 GPU 计算。

首次源码传输发生 SSH 中断，收到空 tar，解包即失败，容器和推理均未启动。重新传输并核对源码包 SHA 后才执行唯一一次推理；没有重抽或重跑模型输出。

同一行重复前向和误差检查通过；假输入 SHA 被拒绝；采样可从 seed 独立重建，角色计数和无放回约束通过。模型、Replay、Adam、元数据和输入文件前后哈希一致；原始两份外层 ZIP SHA 不变。Mac 和 LocalServer 仅用 NumPy 重建索引、读取冻结预测并复算全部汇总，结果一致；没有第二次完整模型推理。

完整产物在 `reports/cross-replay-20261009/`：eligibility.json、plan.json / plan.sha256、analyze.py、只读 cross-replay.json、summary.json、完整端点 ZIP/marker、来源快照、基线源码包和执行/读回记录。cross-replay.json 包括每行预测、索引、完整误差矩阵、标签分层和输入身份，SHA 为 `feefbbf436011908fe8a8d94897448f4abcc06adbb930baf654f64d759808424`。

不进行新增前向即可复算：

```sh
# 把 reader-source.tar.gz 解包到新的 source 目录，保留原产物不动。
PYTHONPATH=/absolute/source python /absolute/evidence/analyze.py verify /absolute/evidence
```

`run` 仅为同一冻结输入的复算入口；本轮不再执行它。原始 R0 外层校验所需两阶段 ZIP 继续复用既有 Baseline 双域归档，不重写。恢复步骤和所有文件哈希见证据包 RESTORE.txt / archive-manifest.json。

到此停止。不调整 Replay age、NTP/belief/LR/batch，不扩大样本，不绘制事后相关性，不调用最终留出，不进行选优或新训练。**G_NOT_ENTERED / TRAINING_PAUSED。**

最终证据包 `reports/cross-replay-20261009.zip`，SHA256 `cfb79e08093d06bc615649bf92219c90dfe90b7eb5a605db1a892914640f0391`。Mac 与 LocalServer 各自从该 ZIP 解包，核验 23 个文件并仅用 NumPy 读回复算通过；回执为 `reports/cross-replay-20261009-archive-verification.json`。LocalServer 副本为 `/var/tmp/cross-replay-20261009.zip`。原始外层 ZIP 未改动，本轮唯一仓库新增为此结案报告；一次性代码与大体积证据保留在 Git 忽略的 reports 归档中。
