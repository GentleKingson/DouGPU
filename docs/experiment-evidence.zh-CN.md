# 实验报告与证据获取

更新日期：2026-10-03，Asia/Hong_Kong。

## 当前预算决定

6h→8h 独立训练 seed 的最终结果为 seed44 PASS（5/5）、seed43 INCONCLUSIVE（4/5）、seed45 INCONCLUSIVE（4/5）。按 seed45 执行前冻结的跨 seed 规则，停止本轮预算扩张，保留 6h，不进入 12h，不追加第四 seed 或补抽留出。6h 是实验预算决定，不代表已修改单次会话的 `max_hours`。

这些结果未证明 8h 普遍无效或显著退化，但不足以确立可靠的跨 seed 边际收益。角色方差初查尚未确定因果来源。

## 阅读历史报告

各阶段报告中的“当前”“下一步”“未启动”和测试数量均指该阶段结束时的状态，不是现行执行指令。保留原始统计、门槛与当时决定，不用后续结果追溯改写：

- [六小时 seed42 报告](experiments/six-hours-seed42-20261002.zh-CN.md)是 4h→6h，地主与农民非劣界均为 -3pp。
- 后续 6h→8h 协议的地主非劣界为 -1pp，农民仍为 -3pp，不能混用两轮门槛。
- seed43/44 的长跑均从零独立初始化，但在 4h 后完整状态恢复到 6h，不是不中断的六小时进程。早期短跑报告的“连续运行”只描述其当轮运行。

发布版报告见[实验索引](experiments/README.md)，最新结论见[seed45 最终报告](experiments/eight-hours-seed45-20261003.zh-CN.md)。原始报告在仓库相对路径 `reports/eight-hours-seed45-20261003/final-report.md`。统计与跨 seed 决定分别在其 `bundle/evidence/statistics.json`、`bundle/evidence/cross-seed-summary.json`；初查在 `bundle/evidence/role-variance-initial-audit.json`。

## 获取与复核

`reports/` 被 Git 忽略，报告提到的原始证据、checkpoint、驱动和归档不会随 clone 获取。目前仅保存在实验工作区，尚未发布公共下载地址。需要复核时，请向仓库维护者申请对应实验目录的证据归档、校验清单及复核脚本；本页不表示归档已上传或保证公开分发第三方权重。

取得归档后，先核对对应报告中的归档 SHA256，再按 `evidence-manifest.json` 验证文件。复核脚本和依赖恢复方法以该轮报告为准；八小时实验的 `verify_delivery.py` 位于各实验目录。不要将审计入口替换成训练入口，也不要对既有 checkpoint 盲目重跑训练。

seed45 最终归档 SHA256：`72550e6a32b6d7fb7dfa5281e25c57177d6f49f7575fe953fe010ef507562430`。其他轮次使用各自报告的哈希，不能互换。服务器临时目录是历史运行记录，清理后不能作为获取地址。
