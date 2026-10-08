# 实验报告与证据获取

更新日期：2026-10-08，Asia/Hong_Kong。

**证据现状：维护者已确认原始实验已删除。** 本页及历史报告保留当时的统计和决策，但所列原始目录、checkpoint、诊断文件和归档目前不可用，无法重新复核或用于续训。下文的路径与 SHA256 仅作历史身份记录，不是可下载地址或现存备份证明。

## 当前 Batch 决定

Effective batch512 rejected：seed43/44/45 全部 FAIL，跨 seed REJECT，WP / ADP Balanced 等权点差均值分别为 -1.4514pp / -1.5722pp。不进入 final holdout，不加样本或第四 seed，不自动 rescue，生产默认 batch256 不变。拒绝的是本轮 batch512 方案，不是 accumulation=8。

既有日志已确认 fork 后阶段 sample-update ratio（replay sampling intensity）为 baseline 约 2×，不是按更新次数定义的 UTD 翻倍；frozen-replay 梯度与 Adam 单步诊断已完成，未确定退化因果来源。报告及诊断限制见 [batch512 报告](experiments/batch512-20261004.zh-CN.md)。原始目录为 `reports/batch512-experiment-20261004/`，含冻结 `input/protocol.json`、`evaluation-output/statistics.json`、`evaluation-final-verification.json`、`diagnose_logs.py`、`log-diagnostics.json`、`frozen_probe.py`、`frozen-replay-output/` 与 `diagnostics-verification.json`；这些文件未随仓库发布，现已不可用。评测输出归档 SHA256 为 `d1342b9e2d3cdbcd49e98e29885c6e3e198c92e961a759aa667e3fb69b44e34d`，仅包含本轮评测输出，不是含训练 checkpoint 的完整归档。

## 当前预算决定

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

`reports/` 被 Git 忽略，历史报告提到的原始证据、checkpoint、驱动和归档不会随 clone 获取。原先“仅保存在实验工作区、可向维护者申请”的说明已失效；目前没有已确认可用的原始备份。不以重训、参考模拟或新增牌局替代已删除的原始证据。

如果将来找回独立备份，先核对该轮归档 SHA256，再按其 `evidence-manifest.json` 验证文件，执行原来的复核脚本；不能只凭报告中的哈希宣称恢复成功。seed45 最终归档历史 SHA256 为 `72550e6a32b6d7fb7dfa5281e25c57177d6f49f7575fe953fe010ef507562430`，不能用于其他轮次。旧服务器临时目录不是获取地址。

## 新实验的最小保留与恢复流程

复用 `Store`、`--mirror-dir` 和 SHA256，不另建存储系统。下列流程是下一轮实验的前置要求，不授权启动训练。

1. **事先确定两处持久存储。** 训练工作目录与镜像必须在不同故障域（例如不同主机或独立备份服务），镜像放在工作目录树之外，并为每个实验分配独立目录。同盘两个目录只能验证程序恢复逻辑，不能抵抗整盘丢失。确认挂载实际可用、空间与权限正常，避免远端掉线后误写本地挂载点。
2. **训练时使用现有镜像。** 获得预算批准后，运行现有 `train` 命令并指定 `--mirror-dir <独立存储上的实验镜像目录>`。保持 `save_replay=true`，检查每次保存的 `remote_ok=True`。Store 会复制完整 ZIP、最后提交的 `.ok.json`、latest/best 策略及指标日志；ZIP 内包含参数、Adam、champion、replay、配置、主 RNG、计数与来源记录。镜像失败会保留本地状态并报警；后续保存失败不保证自动停止训练，不得把仅本地成功视为备份完成。
3. **冻结关键节点，不依赖滚动保留。** 本地和镜像都按 `keep_checkpoints` 清理旧代，latest/best 导出会被覆盖。实验安全停止后，将需要保留的起点和端点 ZIP 与对应 `.ok.json` 复制到不受 Store 清理的独立归档目录。不要让不同实验共用镜像。保留最终策略、完整指标日志、`config.json`、`session_config.json`、`source/`、`run_info.json`、preflight 结果、冻结协议、评估 JSON/逐牌 outcomes、发牌 seeds、对手身份与复算入口。外部评估结果及整个 run 目录**不会**被 `--mirror-dir` 自动备份，必须随结案归档显式复制。
4. **固定实际源码与清单。** checkpoint 的 `versions` 尽力记录 commit/dirty，但 `unknown` 或 dirty 状态不能单独证明源码身份；`source/trainer_source.zip` 也只代表 prepare 时的源码。每个会话保存实际 commit 对应源码（有本地修改时保存实际源码快照）、依赖记录和配置。对所有归档文件生成相对路径 SHA256 清单，并在第二份备份上重算；Linux 可用 `sha256sum`，macOS 可用 `shasum -a 256`。清单不包含自身，不删除或回写原始输入。
5. **从备份实际恢复，再宣布闭环。** 先确认两份归档完整，不删除唯一真实副本。在隔离临时工作目录恢复配置与 `source/`，用现有 `Store(新本地目录, 镜像目录).load_latest()` 只读加载，并运行 `check_array_state`；核对实际加载 ZIP 的 SHA256，避免将回退旧代误认为恢复了指定端点。逐项比较参数、Adam m/v/step、champion、replay（含 ring 位置）、主 RNG、配置、日志和计数，确认 `Adam.step == updates`。也可使用现有 `import-checkpoint` 导入到新的 prepared run，但导入会记录 fork 来源，不应要求 fork 元数据整体不变。此步无须启动 learner 或 GPU。
6. **独立核验外围产物。** 按外层清单核验最终策略、评估结果、源码及配置；Store 的 ZIP manifest 只覆盖 ZIP 内文件。记录恢复所用归档/端点哈希、验证命令和结果。恢复范围仍为 learner、Adam、replay 和主 RNG，actor 在途牌局会重启。只有恢复记录与两份持久归档都存在，才能批准清理工作目录。

## 本轮代码级验证与启动门槛

现有测试覆盖镜像读取、较新镜像优先、损坏代回退、非法 marker 拒绝和 full-state 导入。本轮在 `tests/test_local_runtime.py` 补充一个完整状态恢复测试：保存带 replay、主 RNG、配置、来源记录和日志的镜像，删除测试本地主目录及原始状态目录，再从镜像恢复逐项比较；分别破坏 ZIP 和仅破坏 ZIP 内条目（重算外层哈希），确认两层校验均拒绝加载。

CPU 回归命令（使用项目现有开发环境）：

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/test_checkpoint_logs.py tests/test_source_and_recovery_guards.py tests/test_local_runtime.py tests/test_files.py
```

该测试使用临时目录中的合成训练状态，不执行训练，不证明跨主机故障恢复或实际备份服务可用。Store 的保存/恢复能力无需为此修改；实际不同故障域的存储选定、全产物归档和该存储上的恢复演练仍待下一轮实验前完成。新 Baseline 的 GPU 预算尚未批准，本轮结束后不启动 preflight、训练或评估，不宣称重建了过去的 6h 棋力。
