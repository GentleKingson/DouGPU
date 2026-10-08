# 实验报告与证据获取

更新日期：2026-10-09，Asia/Hong_Kong。

[执行证据闭环](experiments/execution-evidence-20261009.zh-CN.md)已完成：INFRASTRUCTURE_READY / TRAINING_PAUSED；新记录可跨机器离线复核，历史缺失证据未补齐，准入仍 FAIL。

**证据现状：旧实验原始数据已删除，新 Baseline 已独立归档。** 2026-10-01 至 2026-10-04 历史实验的原始目录、checkpoint、诊断文件和归档不可用，相关路径与 SHA256 仅作历史身份记录。2026-10-08 新 Baseline 的证据不受此历史缺失影响。

## 新 Baseline 独立评估

2,000/20,000 更新完整状态、三份候选、WP/ADP 权重、冻结协议、24,000 局逐副 outcomes、统计及复核脚本已保存至独立归档，并完成 Drive 读回、SHA256 校验和恢复复算。主比较为 **POSITIVE_SIGNAL**；这是单训练 seed 的探索性证据，不自动晋升或续训。结果、限制、下载回执和复核命令统一见[新 Baseline 强对手评估报告](experiments/baseline-strength-20261008.zh-CN.md)。

## 后续选优、机制分析与 NTP 结案

此前的 [Checkpoint Selection](experiments/checkpoint-selection-20261008.zh-CN.md) 已按地主非劣门槛停止（NO_DECISION）；后续[角色机制分析](experiments/role-mechanism-20261008.zh-CN.md) 为 INCONCLUSIVE / STOP。两轮证据均已在 Mac 与 LocalServer 归档并恢复核验，入口分别为 `reports/checkpoint-selection-20261008-archive-verification.json` 和 `reports/role-mechanism-20261008-archive-verification.json`。这些结论不构成新训练的授权或地主学习缺陷的因果证明。

另行获批的 [NTP 消融](experiments/ntp-ablation-20261008.zh-CN.md)因恢复边界漏核而中止于 500 更新，复核结案为 **CLOSED / INVALID_PROTOCOL；Training：PAUSED**，未执行棋力比较。完整中止状态及偏差记录已双域恢复验证，回执为 `reports/ntp-ablation-20261008-archive-verification.json`；不能将恢复成功或 CPU/GPU 检查通过解释为消融实验有效。下一阶段仅强化以下协议准入要求，不自动重试。

## 历史 Batch 决定

Effective batch512 rejected：seed43/44/45 全部 FAIL，跨 seed REJECT，WP / ADP Balanced 等权点差均值分别为 -1.4514pp / -1.5722pp。不进入 final holdout，不加样本或第四 seed，不自动 rescue，生产默认 batch256 不变。拒绝的是本轮 batch512 方案，不是 accumulation=8。

既有日志已确认 fork 后阶段 sample-update ratio（replay sampling intensity）为 baseline 约 2×，不是按更新次数定义的 UTD 翻倍；frozen-replay 梯度与 Adam 单步诊断已完成，未确定退化因果来源。报告及诊断限制见 [batch512 报告](experiments/batch512-20261004.zh-CN.md)。原始目录为 `reports/batch512-experiment-20261004/`，含冻结 `input/protocol.json`、`evaluation-output/statistics.json`、`evaluation-final-verification.json`、`diagnose_logs.py`、`log-diagnostics.json`、`frozen_probe.py`、`frozen-replay-output/` 与 `diagnostics-verification.json`；这些文件未随仓库发布，现已不可用。评测输出归档 SHA256 为 `d1342b9e2d3cdbcd49e98e29885c6e3e198c92e961a759aa667e3fb69b44e34d`，仅包含本轮评测输出，不是含训练 checkpoint 的完整归档。

## 历史预算决定

6h→8h 独立训练 seed 的最终结果为 seed44 PASS（5/5）、seed43 INCONCLUSIVE（4/5）、seed45 INCONCLUSIVE（4/5）。按 seed45 执行前冻结的跨 seed 规则，停止本轮预算扩张，保留 6h，不进入 12h，不追加第四 seed 或补抽留出。6h 是实验预算决定，不代表已修改单次会话的 `max_hours`。6h / batch256 是历史推荐配置与预算方向，不代表仍有可继续训练的 6h 权重。

这些结果未证明 8h 普遍无效或显著退化，但不足以确立可靠的跨 seed 边际收益。角色方差初查尚未确定因果来源。

## 阅读历史报告

各阶段报告中的“当前”“下一步”“未启动”和测试数量均指该阶段结束时的状态，不是现行执行指令。保留原始统计、门槛与当时决定，不用后续结果追溯改写：

- [六小时 seed42 报告](experiments/six-hours-seed42-20261002.zh-CN.md)是 4h→6h，地主与农民非劣界均为 -3pp。
- 后续 6h→8h 协议的地主非劣界为 -1pp，农民仍为 -3pp，不能混用两轮门槛。
- seed43/44 的长跑均从零独立初始化，但在 4h 后完整状态恢复到 6h，不是不中断的六小时进程。早期短跑报告的“连续运行”只描述其当轮运行。

发布版报告见[实验索引](experiments/README.md)，预算扩张最终结论见[seed45 最终报告](experiments/eight-hours-seed45-20261003.zh-CN.md)。原始报告曾位于仓库相对路径 `reports/eight-hours-seed45-20261003/final-report.md`。统计与跨 seed 决定分别在其 `bundle/evidence/statistics.json`、`bundle/evidence/cross-seed-summary.json`；初查在 `bundle/evidence/role-variance-initial-audit.json`。

## Coach 结案

2026-10-08，Coach 方案在 Stage 0 停止，维护者同意 **STOP / CLOSED_NO_DATA**。冻结源码与规则哈希核验、100 副官方规则初始牌编码和 replay 往返自检通过；真实 checkpoint 与诊断证据缺失。Stage 1 拟合、快速 GPU 训练及正式棋力 A/B 均未执行，生产模型、配置和训练算法未修改。

本地审计文件为 `reports/coach-audit-20261008/{coach-data-audit.json,coach-fit.json,selfcheck.py}`；该目录被 Git 忽略，未随仓库发布。本节记录结案，不将本地审计转述冒充可公开重算的原始训练证据。无须为推进本轮重新采集数据。

今后若另立 Coach 实验，须先规定数据采集、证据持久化、独立验证与预算，并明确以下准入口径：恢复不包含 actor 在途牌局及内部 RNG 的逐位恢复；每候选随机放行概率与最终接受样本的随机通道占比不同；AUC 的并列预测按 0.5 计分，事前固定 ECE 和分组校准规则。本轮不为这些要求增加生产实现。

## 已删除证据的处理

`reports/` 被 Git 忽略，原始证据、checkpoint、驱动和归档不会随 clone 获取。对已删除的旧实验，原先“仅保存在实验工作区、可向维护者申请”的说明已失效，目前没有已确认可用的原始备份。不以新 Baseline、重训、参考模拟或新增牌局替代已删除的原始证据。

如果将来找回独立备份，先核对该轮归档 SHA256，再按其 `evidence-manifest.json` 验证文件，执行原来的复核脚本；不能只凭报告中的哈希宣称恢复成功。seed45 最终归档历史 SHA256 为 `72550e6a32b6d7fb7dfa5281e25c57177d6f49f7575fe953fe010ef507562430`，不能用于其他轮次。旧服务器临时目录不是获取地址。

## 新实验的最小保留与恢复流程

复用 `Store`、`--mirror-dir` 和 SHA256，不另建存储系统。下列流程是下一轮实验的前置要求，不授权启动训练。

1. **事先确定两处持久存储。** 训练工作目录与镜像必须在不同故障域（例如不同主机或独立备份服务），镜像放在工作目录树之外，并为每个实验分配独立目录。同盘两个目录只能验证程序恢复逻辑，不能抵抗整盘丢失。确认挂载实际可用、空间与权限正常，避免远端掉线后误写本地挂载点。
2. **训练时使用现有镜像。** 获得预算批准后，运行现有 `train` 命令并指定 `--mirror-dir <独立存储上的实验镜像目录>`。保持 `save_replay=true`，检查每次保存的 `remote_ok=True`。Store 会复制完整 ZIP、最后提交的 `.ok.json`、latest/best 策略及指标日志；ZIP 内包含参数、Adam、champion、replay、配置、主 RNG、计数与来源记录。镜像失败会保留本地状态并报警；后续保存失败不保证自动停止训练，不得把仅本地成功视为备份完成。
3. **冻结关键节点，不依赖滚动保留。** 本地和镜像都按 `keep_checkpoints` 清理旧代，latest/best 导出会被覆盖。实验安全停止后，将需要保留的起点和端点 ZIP 与对应 `.ok.json` 复制到不受 Store 清理的独立归档目录。不要让不同实验共用镜像。保留最终策略、完整指标日志、`config.json`、`session_config.json`、`source/`、`run_info.json`、preflight 结果、冻结协议、评估 JSON/逐牌 outcomes、发牌 seeds、对手身份与复算入口。外部评估结果及整个 run 目录**不会**被 `--mirror-dir` 自动备份，必须随结案归档显式复制。
4. **固定实际源码与清单。** checkpoint 的 `versions` 尽力记录 commit/dirty，但 `unknown` 或 dirty 状态不能单独证明源码身份；`source/trainer_source.zip` 也只代表 prepare 时的源码。每个会话保存实际 commit 对应源码（有本地修改时保存实际源码快照）、依赖记录和配置。对所有归档文件生成相对路径 SHA256 清单，并在第二份备份上重算；Linux 可用 `sha256sum`，macOS 可用 `shasum -a 256`。清单不包含自身，不删除或回写原始输入。
5. **从备份实际恢复，再宣布闭环。** 先确认两份归档完整，不删除唯一真实副本。在隔离临时工作目录恢复配置与 `source/`，用现有 `Store(新本地目录, 镜像目录).load_latest()` 只读加载，并运行 `check_array_state`；核对实际加载 ZIP 的 SHA256，避免将回退旧代误认为恢复了指定端点。逐项比较参数、Adam m/v/step、champion、replay（含 ring 位置）、主 RNG、配置、日志和计数，确认 `Adam.step == updates`。也可使用现有 `import-checkpoint` 导入到新的 prepared run，但导入会记录 fork 来源，不应要求 fork 元数据整体不变。此步无须启动 learner 或 GPU。
6. **独立核验外围产物。** 按外层清单核验最终策略、评估结果、源码及配置；Store 的 ZIP manifest 只覆盖 ZIP 内文件。记录恢复所用归档/端点哈希、验证命令和结果。恢复范围仍为 learner、Adam、replay 和主 RNG，actor 在途牌局会重启。只有恢复记录与两份持久归档都存在，才能批准清理工作目录。

## 执行前会话历史一致性门禁

2026-10-09 已执行[一次性离线联合审计](experiments/protocol-hardening-20261009.zh-CN.md)，入口为 `scripts/protocol_gate.py`。边界和完整状态核验通过，但独立执行证据不足，真实准入 FAIL；下列要求不因此放宽。

适用于后续复用历史训练 Baseline 的实验；必须在任何新 GPU 预检、smoke 或训练之前完成。此要求不授权启动新实验，也不修改生产训练器。复用已归档的 `reports/ntp-ablation-20261008/audit-restarts.py` 的日志边界提取逻辑；该脚本本身是本轮失败诊断，尚不具备以下完整联合核验能力。

1. **还原实际会话。** 从 `metrics.jsonl` 的 `session_id`、`successful_steps`、`updates` 还原有序会话的开始与结束更新数。核对会话内及会话间连续性、累计成功更新数，拒绝日志缺口、重复/交错会话及仅凭最终配置推断历史。checkpoint 日志可能不包含保存之后的 `session_end`，缺失部分必须由执行记录和对应状态交叉说明，不能自动忽略。
2. **联合确认恢复来源和配置。** 对每次会话关联实际命令、退出记录、配置快照、输入/输出 checkpoint SHA256 与 metadata；核对 `Adam.step == updates`、恢复源确为上一会话指定状态、源码/规则/依赖身份一致，并验证 worker seeds 及 actor 重启的执行语义。比较每次实际解析后的 ModelConfig/TrainConfig；复用 `check_training_semantics` 检查学习字段，同时检查它未覆盖但可能影响数据轨迹的 actor、采样和执行配置。缺少某会话配置或恢复来源证据时不得用当前默认值填补。
3. **对照事前 B 计划。** 逐会话比较成功更新边界、停止/恢复方式及配置；唯一计划学习目标差异为已声明的 NTP 权重。所有其他差异须在新协议中明确分类、给出可比性依据并冻结，不得默认为无影响。本次历史 A 的边界是 `0→8→2,000→20,000`；B 若另获授权，须从零开始匹配，不能复用已中止的 500-update 状态。
4. **形成启动前回执。** 保存逐会话对照表、输入哈希、差异清单和明确 PASS/FAIL 结论，与新协议在两个故障域归档。任何缺失、矛盾或未经声明的学习条件变化均为 **INVALID_PROTOCOL / STOP**，禁止启动 GPU。日志提取脚本退出码 0、端点状态恢复成功或边界数字相同，均不能单独视为联合门禁通过。

即使门禁通过，也只排除已核验的执行差异，不保证不同目标下自博弈轨迹逐位一致，不替代跨 seed 复现。将来的执行者应在新实验的一次性准入检查中落实这些要求，不为本轮结案新增训练功能或通用框架。

## 既有恢复能力测试

现有测试覆盖镜像读取、较新镜像优先、损坏代回退、非法 marker 拒绝和 full-state 导入。此前在 `tests/test_local_runtime.py` 补充的完整状态恢复测试会：保存带 replay、主 RNG、配置、来源记录和日志的镜像，删除测试本地主目录及原始状态目录，再从镜像恢复逐项比较；分别破坏 ZIP 和仅破坏 ZIP 内条目（重算外层哈希），确认两层校验均拒绝加载。

CPU 回归命令（使用项目现有开发环境）：

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/test_checkpoint_logs.py tests/test_source_and_recovery_guards.py tests/test_local_runtime.py tests/test_files.py
```

该测试使用临时目录中的合成训练状态，不执行训练，不证明跨主机故障恢复或实际备份服务可用。Store 的保存/恢复能力无需为此修改；后续新 Baseline 已另行完成真实完整状态归档、不同故障域读回及恢复演练，详见上方报告。新 Baseline 不代表重建了过去的 6h 棋力；本轮评估结束后不自动续训。
