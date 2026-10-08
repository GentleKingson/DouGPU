# 实验执行证据闭环

日期：2026-10-09（Asia/Hong_Kong）。阶段状态：**INFRASTRUCTURE_READY / TRAINING_PAUSED**。历史 Baseline 准入继续 FAIL；历史 NTP 消融继续 CLOSED / INVALID_PROTOCOL。仅运行合成 CPU 测试及离线恢复验证，未启动研究训练，GPU 消耗为 0。

## 实施结果

- A：离线历史门禁、六项原自检及原报告纳入版本管理；与既有归档源码哈希核对一致。历史读取器的拒绝准入语义不变。
- B：`Store.load_latest()` 返回已核对 marker 的 SHA256，保留损坏新版本回退。训练进程把实际恢复路径、SHA256、更新数、cycle 与本次 session ID 写入 start 日志及 checkpoint metadata；启动器的预检查不充当恢复凭证。
- C：ActorPool / ReadyActorPool 成功创建后，记录实际 worker seeds、顺序、模式、packed 布局及更新边界。`session_start` checkpoint 仍早于 seeds 生成；后续 checkpoint 包含 Actor 事件。新增记录不调用主 RNG，不改变 Actor 初始化方式。
- 源码：启动时核对当前根目录 Python 文件及 `dougpu/*.py` 与既有 `trainer_source.zip` 的文件集合和 SHA256，并记录加载模块的路径映射；继续复用 DouZero source lock 核验。快照存在但不一致时，在恢复/保存前拒绝执行。直接运行缺少快照时记录空 snapshot hash，离线准入拒绝；Git `unknown` 不再充当证明。
- D：扩展同一离线门禁的 `--execution` 读取模式，读取实际 checkpoint、累计 JSONL、冻结源码 ZIP 和端点计划；复用完整数组、Replay 和 RNG 恢复检查。逐会话比较配置、source lock、恢复输入哈希和边界，交叉检查日志与 checkpoint。PASS 只表示执行证据完整，不授权实验。

## 验证与独立复算

| 验证 | 结果 |
| --- | --- |
| LocalServer 既有 Docker 环境，CPU 回归 | 111 项通过；无 GPU 挂载、无网络、源码只读 |
| Mac 既有 NumPy 环境及复用的 pytest 文件 | 85 项通过；未安装依赖 |
| 两地完整合成归档恢复 | PASS 回执逐项相同；检查参数、Adam、champion、Replay、主 RNG 和日志 |
| 两地真实历史 Baseline 重验 | FAIL 回执逐项相同；原始三个归档 SHA256 符合冻结计划 |
| 合成负例 | 缺 seed、错误恢复哈希、源码不一致、缺重启边界、伪造状态和 Actor metadata 不一致均拒绝 |
| RNG 与 Actor 参数 | ordered / ready-first / overlap 四种路径核对实际构造参数；已结束端点不消耗 RNG；继续恢复生成预期下一组 seeds |
| 源码篡改 | 修改快照中的训练源码后启动核验拒绝 |

Mac 没有 JAX，训练入口集成回归在 LocalServer CPU 上执行；Mac 独立执行完整 checkpoint 的 NumPy 离线恢复与门禁。早期隔离目录缺少 upstream cache / vendor 导致的测试输入错误已补齐并重跑通过。没有开展 GPU 性能、棋力或论文再审。

## 使用与归档

历史入口保持不变。新证据使用：

```bash
python scripts/protocol_gate.py --execution RUN_DIRECTORY APPROVED_PLAN.json
```

运行目录需包含 `checkpoints/`、外部 `metrics.jsonl`、`source/trainer_source.zip`。计划顶层为终点 `checkpoint_sha256` 与有序 `sessions`；每段包含完整 `model`、`train`、`source_lock`、`start_updates`、`end_updates`、`source_sha256`、`input_checkpoint_sha256`（从零开始为 null）。恢复端点、配置、源码及边界应在运行前冻结；终点输出哈希在验收归档时固定。测试中的 `approved-plan.json` 是合成夹具，不是研究训练批准文件。

此次读取器有意只支持同一冻结源码、从零起始的完整会话链，并要求每个有效训练会话存在 Actor 启动记录和最终 target_updates 完整端点。失败会话、无 Actor 的空会话、缺历史证据均不豁免。启动时文件核验不等同于持续监控或抗恶意篡改的硬件证明。

本轮证据目录为 `reports/execution-evidence-20261009/`（Git 忽略），含两地测试日志、合成完整恢复包、两地 PASS / 历史 FAIL 回执、源码包、哈希清单及恢复步骤。最终 ZIP 与独立读回回执分别为 `reports/execution-evidence-20261009.zip` 和 `reports/execution-evidence-20261009-archive-verification.json`，同时保存在 LocalServer `/root/` 对应路径。

旧 prepare 快照不会被覆盖。新的代码与旧快照不一致时须使用另行审查的新实验目录，不能通过重写历史快照让旧实验通过。新训练继续 PAUSED；本阶段不调整 NTP 权重、模型、RNG 流或预算。
