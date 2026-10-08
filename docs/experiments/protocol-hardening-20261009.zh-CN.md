# 实验协议可靠性强化

日期：2026-10-09（Asia/Hong_Kong）。**离线审计完成；准入 FAIL / PROTOCOL_NOT_READY；Training：PAUSED。** 未获得或执行训练授权，GPU 消耗为 0。NTP、Coach、Checkpoint Selection 和角色机制实验维持原结案状态。

新增一次性 [离线门禁](../../scripts/protocol_gate.py) 与 [标准库自检](../../tests/test_protocol_gate.py)。生产训练器、模型、Actor、checkpoint 和恢复语义均未改动，无新增依赖。门禁直接使用 `Store.load_latest()`、`check_array_state()`、Replay 恢复和 `check_training_semantics()`；只有临时目录中的 checkpoint 副本被加载，历史归档保持只读。

## 本次实际核验

| 检查 | 结果 |
| --- | --- |
| Gate 0：文件完整性 | Mac、LocalServer 重算 NTP ZIP SHA256 及 51 项外层清单一致；三个 Baseline ZIP 与原 cloud-verification 回执一致 |
| Gate 1：成功更新历史 | 还原下表三段；完整参数、Adam、champion、replay、主 RNG、metadata、日志及解析配置通过核验 |
| Gate 1：执行证据完整性 | FAIL：未独立记录实际 worker seeds；源码快照不能单独证明运行时源码身份；恢复日志只有输入文件名，没有加载时输入 SHA256 |
| Gate 2：A/B 配置比较 | 冻结离线候选匹配三段实际配置，仅 `ntp_weight: 0.02→0`；证据缺口仍阻断联合准入 |
| Gate 3：错误阻断 | 两地 6 项自检通过；Mac 的 Python `-O` 亦通过；真实历史回放稳定拒绝漏掉 8-update 边界、缺配置及恢复方式矛盾 |
| Gate 4：归档 | 两地完整审计 JSON 逐项相同、都退出 1；本轮证据小包另行双域读回核验。归档成功不等于准入通过 |

| 会话 ID | 成功更新范围 | Adam.step | 恢复来源 |
| --- | --- | --- | --- |
| `7ff9bfd112b448b1988c56d0c4a59c55` | 0→8 | 8 | 从零初始化；首段无 RESUME 日志 |
| `8eac621dcca245ef938f51ff807d2f36` | 8→2,000 | 2,000 | stdout 文件名对应上一段归档 checkpoint |
| `86a7ab239ce64536a590786afc4babbb` | 2,000→20,000 | 20,000 | stdout 文件名对应上一段归档 checkpoint |

恢复源文件名和上一段文件哈希能关联预期输入，但不能独立证明当时加载的字节。门禁保留这一区别，不将关联推断写成实际输入哈希记录。三段启动日志的 Git commit/dirty 均为 `unknown`；冻结源码 ZIP 一致仍不足以单独消除执行身份缺口。worker seeds 只可按代码推导预期生成方式，本轮没有把推导值当成实际记录，也没有尝试恢复 Actor 在途牌局。

三个端点 checkpoint 内均缺少本会话最后的 `session_end`，这符合源码先保存、后写结束日志的顺序。对应外部 metrics、stdout 的 start/end、checkpoint 事件、正常退出记录、完整状态及端点 metadata 均相符；因此该缺尾本身不是本轮失败原因。完整累计日志通过连续性检查，拒绝重复、交错会话和更新缺口。

## 准入口径与复算

离线候选计划逐段保留实际 ModelConfig/TrainConfig，包括当时动态计算出的 `max_hours`；它不是已批准的新训练方案。比较器先要求完整字段，禁止用默认值补齐；复用学习语义检查，并逐字段检查其未覆盖的参数。差异按学习、数据轨迹、执行环境、记录字段分类；本次只接受已声明的 NTP 权重差异，其他字段即使属于记录项也不默认放行。

此脚本专用于上述三份历史归档，不是通用实验平台。这些归档缺少独立执行证明，因此其读取器不会产生真实 PASS；没有接受人工 `verified=true` 或豁免缺失项的入口。合成完整证据测试可以 PASS，仅说明判定逻辑正确，不替代历史审计。若另有独立历史证据，须先实现并审查对应证据读取与交叉验证，再重新审计；不能通过修改状态字符串放行。

```bash
# 标准库正反例，不导入 NumPy、JAX，也不启动 Actor。
python3 -m unittest discover -s tests -p test_protocol_gate.py -v

# 实际完整状态核验需要已有 NumPy；预期 exit=1，JSON status=FAIL。
python3 scripts/protocol_gate.py \
  reports/baseline-20261008-seed20261009 \
  reports/protocol-hardening-20261009/candidate-plan.json
```

Mac 使用既有 NumPy 2.3.5；LocalServer 通过既有 Docker 镜像 `sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e` 中的 NumPy 2.5.3 独立复算。容器未挂 GPU、未联网，输入只读，临时 checkpoint 在 tmpfs 中加载。未运行 JAX、Actor 或训练进程。

证据目录：`reports/protocol-hardening-20261009/`（Git 忽略）。入口为 `completion.json`、`mac-audit.json`、`server-audit.json`、`candidate-plan.json`、`real-evidence-regressions.json` 和两地自检日志。新包复用原始归档，保留输入哈希与可复算源码，不重复打包训练状态；原始输入位置与恢复步骤见包内 `RESTORE.txt`。归档为 `reports/protocol-hardening-20261009.zip`，SHA256 为 `8327a2e25ad0e43855865bee08ced1a0e62301634e550e86d3f24f9a9cbb36da`；两地读回核验 12 个清单文件及 10 个源码文件，回执为 `reports/protocol-hardening-20261009-archive-verification.json`。旧 `audit-restarts.py` 未修改，重新运行仍是退出 0、输出 INVALID_PROTOCOL_STOP；该结果仅用于核对边界提取一致，不能当作准入成功。

本轮未重查论文、未跑生产全套测试或棋力评测。结论限于协议和证据完整性，不说明 NTP 的棋力收益。任何后续算法实验都需要另行批准；即使将来边界一致，也不保证自博弈轨迹逐位相同，单训练 seed 仍不能替代跨 seed 不确定性分析。
