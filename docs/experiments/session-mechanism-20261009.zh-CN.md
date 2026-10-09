# 会话终点闭环与离线机制诊断（2026-10-09）

结论：**E_PASS；F_PASS（仅观察）；G 未进入。保持 MECHANISM RESEARCH / TRAINING_PAUSED。**

本地 HEAD 与 `git ls-remote origin refs/heads/main` 均为 `cf396b77df403e713ffb82a52a040513f1cfd52a`。以下是该基线上的最小协议改动及验证，不是新实验授权。GPU 使用量为 0；只运行 CPU 协议验收、CPU 回归及冻结输入的离线梯度计算，没有棋力评估。

本次实际取得并读取本机历史 Baseline 三个原始阶段 ZIP，逐项核验 manifest 并用 Store 装载。历史门禁仍 FAIL；没有补写历史事件、修改旧 checkpoint 或重新执行历史 NTP。此处没有独立重新解包整个历史 NTP 证据 ZIP，不能替它扩大结论。

## E：逐会话完整终点

改动文件：

- `dougpu/train.py`：正常终止先写 `session_end`，再保存最终 checkpoint，使恢复后的累计日志保留先前会话末尾。终止事件只说明运行停止，不说明保存成功。Actor 关闭异常也进入现有 error checkpoint 路径。target_updates 的 COMPLETE 判定、RNG、Actor seeds、模型及更新计数逻辑不变。
- `scripts/protocol_gate.py`：每段均要求唯一且有序的 actor_start/start/session_end、非空连续成功更新边界、准确 target、COMPLETE 及 session_end reason。读取 `endpoints/<实际 SHA256>/` 内独立 ZIP 与 marker，使用原 Store、check_array_state、Replay 和主 RNG 恢复检查；按前段真实端点 SHA 串联 resume_input，并从输入 RNG 复算 Actor seeds。逐字段比较预先冻结的 Python/JAX/NumPy/backend。Git unknown 不参与来源证明。
- `scripts/check_session_chain.py`：一个可运行验收入口，创建真实 reference 引擎 CPU 0→1→2 链，另有 NumPy 离线读回模式。每段退出后、下一段开始前，用现有 copy_verified 冻结 checkpoint 与 marker；keep_checkpoints=1，证明独立端点不依赖滚动目录幸存。
- `tests/test_protocol_gate.py`、`tests/test_train_accounting.py`、`tests/test_batch_dose_experiment.py`：补足中间终止事件及失败路径。保存失败即使外部日志已有 COMPLETE session_end，也只能读到非 session_end 的旧 checkpoint；信号、时间/周期上限及 Actor 关闭异常不能通过门禁。

受保护的各段日志必须以 session_end 结束。外部 JSONL 在最终 session_end 后允许至多一条实际保存后的 checkpoint 耗时回执，以兼容现有运行指标；该回执不参与完成证明，恢复时也不要求保留。门禁拒绝其余尾随事件。session_end 的 runtime_seconds 现在截止于最终保存前；最终保存耗时仍单独记录。

`Store.load_latest()` 没有修改：损坏新一代仍可回退。审计再把实际装载 SHA 与冻结端点 SHA 比较，回退到错误代际会 FAIL。缺失中间端点原件会 FAIL，不从日志推测补齐。

### 实际验收

LocalServer 使用既有镜像 `sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e`，无 GPU 挂载、`--network none --cpus 4 --memory 4g`，`JAX_PLATFORMS=cpu`、`CUDA_VISIBLE_DEVICES=`。Python 3.12.14 / JAX 0.7.2 / NumPy 2.5.3。Mac 使用既有 NumPy，未安装或运行 JAX。

首先完成 E 链及两地解包后才执行 F。删除累计字段后又对最终代码重跑真实两会话链。最终端点为：

| 更新边界 | checkpoint SHA256 |
|---|---|
| 0→1 | `3fdff3fdaf290b359061c986e6cb31569c2a6ce0697fcc99c7da26adcb70a5c4` |
| 1→2 | `ddd7000b7fdef26eae6fb572671fd393e45d80a78a4b1cf485f699e381c95e7c` |

冻结意向 SHA：`f174b282bc1f9190e50d14e962c420f2568c8983ce915a139d7184bdab63fc5e`。
执行回执计划 SHA：`70949f6fb935a57a979ccfdaff0edaa7012e042c4d68094ab89d3e804ce2d1c3`。

意向在执行前冻结配置、版本和边界；实际 UUID/ZIP SHA 无法预知，执行回执仅在保存成功后追加这些值，并引用意向 SHA。不能把事后回执称为事前生成的 checkpoint 哈希。

两地主体 JSON 完全一致：正例 PASS；九项负例均 FAIL，包括删除中间 session_end、中间 COMPLETE 改 INCOMPLETE、伪造前段 SHA、删除端点 ZIP、更新不连续、损坏 marker、同时篡改 Actor 事件与元数据中的 seed、改源码、改冻结 JAX 版本。中间日志/状态篡改重写了真实 ZIP/manifest/marker 并更新相应计划 SHA，仍被语义门禁拒绝。

历史 0→8→2,000→20,000 三段被实际读取，仍因缺少独立 Actor seeds、来源证明和恢复输入摘要而 FAIL。历史 NTP 保持 **CLOSED / INVALID_PROTOCOL**。

## F：测量修正与观察

`cumulative_sample_update_ratio` 没有生产读者，已从新日志删除。旧报告及 fork 中的失效说明不改写。逐周期 `sample_update_ratio` 保留：本周期成功更新使用的样本次数 ÷ max(本周期新完成样本数, 1)，不是累计 Replay ratio。

`scripts/mechanism_diagnostics.py` 仅离线计算成功样本消耗，逐段使用 start 的 effective_batch 与 successful_steps；失败更新不计入。缺 batch 或更新不连续则 UNKNOWN。`tests/test_mechanism_diagnostics.py` 验证两个不同 batch 的合成会话和有解析解的梯度方差。

输入是已核验的 Baseline 2,000-update 完整状态，SHA 为 `c653d73a98dd412f1df2e95088ff957d616b8bcd7e4575074d1f9545f42e5830`。历史协议无效不妨碍读取其数值状态，但这些观察不能追认历史实验。

| 角色 | Replay 样本数／可用数 | 版本差 P50 / P90 / P99 | 过期排除率 |
|---|---:|---:|---:|
| 地主 | 22,516 / 22,516 | 72 / 124 / 136 | 0 |
| 下家农民 | 21,676 / 21,676 | 72 / 124 / 136 | 0 |
| 上家农民 | 21,344 / 21,344 | 72 / 120 / 136 | 0 |

采样 clock=2,000，replay_version_updates=true；单位为成功更新数，阈值为 64×4=256。脚本对普通版本使用 cycle 及原始 replay_max_age，严格区分单位。这是端点库存的版本差，不是秒数、策略 KL 或训练全过程的采样滞后分布。

成功样本消耗 512,000；累计完成样本 1,024,088；成功消耗/完成样本=0.4999570349；每个成功更新对应完成样本=512.044。该输入各段 batch 相同，但计算仍按实际有序会话求和。生产实际重复抽样率及分角色历史采集量 **NOT_MEASURED**：旧日志未记录所需索引/计数。

### 梯度与数值路径

CPU 探针固定 seed=20261009、4 个 batch、全局 batch=3、每角色 batch=1，复用 Replay.sample、冻结模型及 sample_losses/JAX 自动微分。保持冻结 BF16，使用 manual attention；没有调用 make_train_step 或优化器。预先限额 240 秒、4 CPU、4 GiB，退出码 0。主训练 RNG 未消费。精确输入张量和物理 Replay 索引见 `gradient-inputs.npz`。

抽样有放回，实测 12 次抽样无重复，三个角色各 4 次也均无重复。全局与角色梯度使用同一组输入，因此不能把这些报告当作互相独立的实验。

定义：对四个 minibatch 梯度求坐标样本方差（ddof=1）之和，再除以样本平均梯度的平方模长。未进行噪声偏差校正，也没有将该比值称为精确的 gradient-noise scale。

| 分层 | 方差迹 / 平均梯度平方模长 |
|---|---:|
| 全局 | 4.1941 |
| 地主 | 4.1540 |
| 下家农民 | 4.3577 |
| 上家农民 | 3.9874 |

仅四个小批次，没有可靠置信区间、跨 batch 对照或最优 batch 推断。各次 loss、梯度模长、方差迹及均值模长均保存在 JSON。

冻结日志记录 nonfinite 跳步 0，成功周期的平均 grad norm 范围 0.28834–33.78870。代码复核确认：norm 与 metrics 有限时才选择新参数/Adam.step；无效步不递增成功计数；连续三次无效触发停止。日志只是成功周期均值，异常发生在周期日志写出前的失败尝试可能缺失，因此不能把“记录为 0”写成无条件数值稳定证明。

BF16/cuDNN 与 FP32/manual 的固定输入比较 **NOT_MEASURED**，没有可比 GPU 测量；不外推 FP16 loss-scaling 结论。模型与生产精度路径均未修改。

## 验证、归档与下一步

可复跑命令（训练入口只在上述无 GPU 容器中执行）：

```sh
python scripts/check_session_chain.py --train NEW_CPU_EVIDENCE_DIR
python scripts/check_session_chain.py RESTORED_EVIDENCE_DIR
python scripts/protocol_gate.py --execution RESTORED_EVIDENCE_DIR/run RESTORED_EVIDENCE_DIR/execution-plan.json
python -m pytest -q -p no:cacheprovider tests/test_train_accounting.py tests/test_batch_dose_experiment.py tests/test_protocol_gate.py tests/test_checkpoint_logs.py tests/test_local_metrics.py tests/test_mechanism_diagnostics.py
python -m pytest -q -p no:cacheprovider tests/test_core.py::test_optimizer_updates_and_skips_nonfinite tests/test_history_groups.py
```

结果：主要回归 **70 passed**；数值路径回归 **3 passed**；Mac 门禁单测 7 passed、诊断单测 2 passed。没有运行全仓测试、GPU 性能或棋力评估。

最终可恢复归档：`reports/session-mechanism-20261009.zip`，SHA256 `d81399f4dbecfd9d7cb441414fd95020b6cfcae8a776bce68d08689590bcffea`。包括真实两段端点/marker、累计日志、冻结意向、回执计划、源 ZIP、冻结诊断 checkpoint、输入张量、诊断 JSON、测试日志、清单及 RESTORE.txt。Mac 与 LocalServer 各独立解包并校验 35 个文件，链审计一致；Replay、消耗和 nonfinite 的 NumPy 复算也一致。Mac 没有重算 JAX 梯度。

两地回执：`reports/session-mechanism-20261009-{mac,server}-readback.json`；汇总回执：`reports/session-mechanism-20261009-archive-verification.json`。LocalServer 归档路径 `/var/tmp/dougpu-session-closure-20261009/session-mechanism-20261009.zip`。哈希校验不是 WORM 或第三方签名，容器镜像记录身份但未整镜像导出。

没有发现可复算的滞后—不稳定关联；梯度探针不足以支持 batch 假设；无已记录的 nonfinite 可复现缺陷。故 **G 未进入**，不虚填独立 seeds、paired holdout、采用阈值或 GPU 预算。后续若形成明确单机制假设，仍须先冻结完整协议并另行明确批准预算，才能另立训练执行任务。


## 入库审阅与研究结论分级

本次审阅覆盖全部 tracked diff、新增脚本、测试和本报告；追踪 train.main 的恢复、正常退出、异常、信号及最终保存路径，以及 Store、Replay、protocol_gate 和 metrics_summary 的调用关系。按单个实验目录只有一个写入者、少量串行会话的既有用法评审，没有为未计划的大规模审计增加框架。

What this change does: Preserve each session's termination event in its checkpoint and verify every archived endpoint. Remove the misleading cumulative field and keep diagnostics offline.

审阅发现并修复一项必须处理的问题：check_session_chain 的冻结意向校验及 CPU 约束使用 assert，Python -O 会移除这些校验。现复用 protocol_gate.require，保持优化模式下失败拒绝；新增测试覆盖意向哈希、段数、配置被改动时，普通及 -O 两种模式均明确拒绝。它是验收器的修复，不改训练算法或旧归档。

没有发现其余阻断项。生产改动只涉及日志、结束路径错误处理与指标删除；model.py、checkpoint.py、Replay、Actor、训练配置默认值及主 RNG 路径均保持基线字节。没有新增依赖、服务或实验选项。Verdict: Ship，条件是本次相关 CPU 回归和提交后的干净工作树读回通过。

| 证据类别 | OBSERVED | NOT_MEASURED | INCONCLUSIVE |
|---|---|---|---|
| Replay | 单个 2,000-update 端点三角色 P99=136，阈值=256，过期排除率=0 | 历史实际重复抽样率、逐次采样滞后及策略 KL | 策略滞后是否影响学习；不能据此调整 age |
| 梯度 | 四个固定小批次，全局方差比值 4.1941 | 跨 batch 对照、可靠置信区间、跨训练 seed 稳定性 | 最优 batch、扩大 batch 的收益及 batch512 重启依据 |
| Nonfinite | 冻结日志记录跳步=0，已记录成功周期均值可复算 | 所有失败尝试的逐步梯度、可比 BF16/cuDNN 与 FP32/manual 输入对照 | 全面数值稳定或单一数值缺陷的因果解释 |

过期排除为零只描述该端点在当前阈值下的过滤结果，不证明策略滞后无影响。历史 Baseline 的数值可读取不改变其 INVALID_PROTOCOL 标记。停止增加诊断数量和实验基础设施；G=NOT_ENTERED，TRAINING_PAUSED。

入库前补跑：同一无 GPU、无网络、4 CPU / 4 GiB 容器中，原 73 项加优化模式校验回归，共 **74 passed in 16.69s**；Mac 门禁单测 **8 passed**。`git diff --check` 通过。`python -O scripts/check_session_chain.py` 对原归档解包目录的输出与原回执完全一致，原 ZIP SHA 不变。此次是同一 Agent 的差异复审，隔离进程/干净检出验收不等于第三方代码审计。


### 已提交源码的可恢复性验收

E/F 实现、测试及本报告已作为 `c849e78a1cbf82654731acac80b738e3cb76b43e` 推送 GitHub main。随后从 GitHub 全新浅克隆该提交，确认工作树干净，以该克隆的读取器重新解包原 ZIP 到临时目录；LocalServer 使用同一干净提交的 git archive，源码与原证据 ZIP 只读挂载。未用归档内的旧读取器替代已提交版本。

两地均校验 35 个文件，0→1→2 正例 PASS、9 项篡改负例拒绝；普通与 `python -O` 读取器一致，`protocol_gate --execution` PASS。Replay、消耗、nonfinite 三类 NumPy 结果与旧结果相同；完整读回回执两地完全一致，均未导入 JAX。历史三段原件通过干净源码重新读取仍 FAIL。原归档 SHA 保持 `d81399f4dbecfd9d7cb441414fd95020b6cfcae8a776bce68d08689590bcffea`，没有覆盖或补写。

机器可读闭环回执：[ef-publication-20261009.json](ef-publication-20261009.json)。原始新回执、CPU 测试日志、精确提交源码和复核脚本位于 `reports/ef-publication-20261009/`；恢复补充包 `reports/ef-publication-20261009.zip` 的 SHA256 为 `11fa9321859bd5d522cf20a5e9005d46073142a0cdd51abda26b252bac3e3073`，与原证据 ZIP 一起保存。LocalServer 副本位于 `/var/tmp/dougpu-ef-review-20261009/ef-publication-20261009.zip`。

本节及 JSON 为仅文档的后续提交，未改变已验收的 Python 源码。E/F 标记为 **ENGINEERING_CLOSED**；G=NOT_ENTERED，TRAINING_PAUSED。没有扩大梯度探针、调整 Replay age、恢复 batch512/NTP 或追加训练 seed。
