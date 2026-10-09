# B2 独立强对手评估：预算预测门禁停止

日期：2026-10-10，Asia/Hong_Kong。结案：**TECHNICAL_FAIL / STOP；棋力 NOT_EVALUATED；TRAINING_PAUSED / G_NOT_ENTERED**。192 局真实 GPU 预检全部完成且重复结果一致，**正式对局为 0**。冻结的耗时预测超过剩余评估窗口，按预注册规则停止；没有调整门槛、缩小样本、重启、晋升或续训。

本次用户在[零 GPU 输入预审](baseline-replication-b2-preaudit-20261010.zh-CN.md)后明确回复“批准”，独立授权 **B2 最多 1,800 秒**，包括 192 局预检、48,000 局正式评估、统计封存和双位置读回。未借用 B1 剩余预算。完整身份、原始执行 argv、计时和归档入口见[机器回执](baseline-replication-b2-20261010.json)。

## 冻结身份与实际执行

- 协议 SHA256：`fdcbf9d555aa6f666fe1abe431a79058f93cd08874742a8fe62d2f587fd360c5`；输入/控制 seal SHA256：`2f008af1a404f68f449c998349a6089d7bf721f7fe0cf6e583b31af233d65c33`，启动前在 Mac 与 LocalServer 再核验完全一致。冻结文件和控制脚本未修改；授权单独追加在 `authorization.json`，不把预审中的零授权历史改成事前已经批准。
- 评估源码：`d7e6e124229a056f936596d26e4b452f45aff064`；六份 2k/20k latest 对应 B1 独立封存端点，来源、政策 SHA、checkpoint/参数关系沿用本轮预审，并在真实运行前再次核验。B1 仍 COMPLETE / EXECUTION_QUALIFIED；历史实验结论保持原状。
- 规则锁 SHA256：`40f47b2cd94a76fe7112140d41b85af0b7462b6c35197241fe71fc506348e1ac`，DouZero `718a5c920bf3361e34178a38f3b80458e176b351`、encoding=1。
- 镜像 `sha256:464175f20cb5460e34e42cd295bf0c9f6629796600b32d9f514243384a55621e`；RTX 5070、UUID `GPU-46dce763-982c-cb97-2004-b7770ccdd7eb`、驱动 595.91.07；JAX 实际 backend=gpu，设备 `cuda:0`。NumPy 2.5.3、JAX/JAXLIB 0.7.2、CPU Torch 2.8.0+cpu，依赖集合/字节检查通过。
- 固定 CUDA、BF16/manual attention、batch=64、action_chunk=2048、workers=0、KV=false；CPU WP/ADP 六角色安全装载，权重和政策有限。无新依赖、配置或生产源码修改。

通过已冻结入口 `python3 reports/b2-preaudit-20261010/control/controller.py` 启动 LocalServer 独立容器。完整 GPU docker argv 位于 `output/evaluation-exit.json`；固定 runc、network=none、24g RAM、2g shm、单 device=0、只读输入入口，GNU timeout 的 TERM→90 秒强杀宽限扣在评估窗口内。源/政策/对手/依赖身份先核验，拒绝 CPU 回退。

三条训练 seed `2026101101/1102/1103`，预检 seed `2026101301/1302/1303`，正式 seed `2026101401/1402/1403` 均保持预审原值。每链 4 副预检牌、两端点×两对手×两阵营重复两次：**24 份逐副文件、192 局**。调用现有 `paired_evaluate`，逐次核对实际有序牌、初始三角色手牌/底牌、地主开局及两阵营映射。每组合两次 outcomes 与完整配对 identity 相同；候选和对手参数逐数组摘要前后不变。Mac/LocalServer 随后独立重算全部预检结果一致。

预检仅用于身份、合法性、重复一致性及成本门禁，**不展示或使用预检胜率作棋力筛选**。未产生 `formal-*.json`、`run-complete.json` 或正式统计；预定 48,000 局没有启动，不从这些预检产生方向性结论。

## 预算门禁和退出事实

冻结公式：最慢 warmed 8-game 组合的每局墙钟 × 48,000 × 1.5 + 60 秒；另在 1,800 秒内预留 300 秒结案窗口。实际最慢 warmed 组合为 **0.026020 秒/局**。

| 项目 | 实测 / 按冻结公式计算 |
|---|---:|
| 正式评估预测 | 1933.451 秒 |
| 门禁当时剩余评估窗口 | 1486.803 秒 |
| 预测超出窗口 | 446.648 秒 |
| GPU 评估容器进程墙钟 | 13.560 秒 |
| 容器退出码 / timed_out | 1 / false |
| 原授权记录至两地普通及优化读回完成 | 228.626 秒（3.81 分钟） |

门禁 `OVER_BUDGET` 触发 `Forecast exceeds evaluation window; STOP without smaller formal sample`，全块按技术规则结案 **TECHNICAL_FAIL / STOP**。实际没有达到硬超时；13.56 秒是 GPU 容器进程包络，包含 CPU/编译等待，不是 CUDA 内核忙碌时间。完整封存/读回仍在原 1,800 秒内，未重新授权或追加额度。

成本预测来自很小的 8-game warmed 预检并加 1.5×安全系数，是保守的事前门禁。**本轮没有测量 48,000 局的实际耗时，不能据门禁失败断言正式规模必然需要超过 1,800 秒。** B0 的历史线性估算也不能代替当前冻结门禁。两者分开披露，未在停止后调低系数或重启来取得结果。

Docker inspect 显示 Running=false、State.Pid=0；监督器 finally 删除命名容器，结束后 `docker ps` 和 GPU compute PID 清单为空。LocalServer `STOP.json` 和 Mac `local-STOP.json` 保留原始失败；后续只做 CPU 证据验证。新增训练更新 0、模型晋升 false、G_NOT_ENTERED。

## 失败证据封存和可复算入口

归档：`reports/b2-preaudit-20261010/failure-evidence.zip`，**22,569,267 字节**，SHA256 **`380968e70fadb1dd36a3cd43535a37bf8a1e84f88580efa3bed78b96587d5291`**。manifest 覆盖 **206 文件**，包含原冻结 source/control/input、独立授权、budget、24 份预检逐副结果、完整 GPU 日志/inspect/退出、门禁预测、STOP 和 CPU 失败复核脚本。两地最终读回回执与结案回执在 ZIP 外，不改写封存包。

Mac 和 LocalServer 分别在全新临时目录解压，验证外层 SHA、manifest、原 protocol/seal 及全部原冻结输入/控制；普通和 `-O` 模式均重新核验 24 份结果的配对身份、角色/牌序、参数不变标志、binary outcomes 与共享统计重算，并从实际第二次预检耗时复算原预测。**四次结果完全一致，FAILED_ATTEMPT_READBACK_PASS**。这是失败证据资格通过，实验状态继续 TECHNICAL_FAIL / STOP。

已有 NumPy 的 Python 可零 GPU 重算：

```sh
python reports/b2-preaudit-20261010/failure-readback.py \
  reports/b2-preaudit-20261010/failure-evidence.zip \
  380968e70fadb1dd36a3cd43535a37bf8a1e84f88580efa3bed78b96587d5291 \
  /tmp/b2-failure-readback.json
```

加 `-O` 复算优化解释器。脚本只读冻结材料、重算 CPU 统计，不导入 JAX、不新建牌局。用户授权、原 GPU argv、原 STOP 和异常均可在归档中审阅；不能把预检文件拿去通过六组正式 Balanced 门槛。

证据 ZIP 保留在本机和 `LocalServer:/var/tmp/dougpu-b2-preaudit-20261010/failure-evidence.zip`；reports 被 Git 忽略，没有新的公开下载地址。Git 发布范围为报告、机器 JSON 与实验索引；原预审封存、归档及 B1 历史不覆盖。

## 结案边界

棋力为 **NOT_EVALUATED**；六组 Balanced、12 项角色点差及训练 seed 总体结论均不可从本轮预检得出，不能报告 DIRECTIONAL_SIGNAL_ONLY。用户提供 WP/ADP 的官方发布身份仍 NOT_VERIFIED，历史隔离仍只覆盖可取得材料。

若以后继续，需要在结果未启用的准备阶段另行审阅成本预测或独立预算并明确批准新执行协议。**本轮 STOP 已执行；没有自动修门槛、重启、加 seed/牌局、续训或晋升。**
