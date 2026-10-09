# Ataraxos 斗地主可行性：零 GPU 文献与源码核查

日期：2026-10-09，Asia/Hong_Kong。DouGPU 基线：`057695699681d69c573449b3291c0358bfdc711d`。开始时本地 HEAD、origin/main 和实时 `git ls-remote origin refs/heads/main` 一致，工作区干净；最新提交为 `docs: close frozen cross-replay research screen`。

**结论：本次静态研究与文档闭环 GO；接入、运行复现、迁移算法和新实验 NO_GO / STOP。公开成套斗地主权重及其可复算性均为 NOT_VERIFIED。G_NOT_ENTERED / TRAINING_PAUSED。** 源码可取得并核验，不等于论文模型、训练轨迹或结果可以复现。

## 既有结案不变

本次先阅读下列报告，引用其已冻结结论，没有重新运行原脚本或打开模型做数值诊断。

| 材料 | 保留的状态及边界 |
|---|---|
| [E/F 会话机制与已提交源码恢复验收](session-mechanism-20261009.zh-CN.md)、[发布回执](ef-publication-20261009.json) | E_PASS；F_PASS 仅观察；ENGINEERING_CLOSED。历史 Baseline 门禁仍 FAIL / INVALID_PROTOCOL；历史 NTP 仍 CLOSED / INVALID_PROTOCOL；G 未进入 |
| [R0–R2 冻结 Replay](cross-replay-20261009.zh-CN.md) | R0_PASS；R1 数值执行 PASS；R2=RESEARCH_SCREEN_INCONCLUSIVE / STOP。自身 Replay 误差较低的交叉形态没有通过原冻结规则，不能解释为棋力或单一因果机制 |
| [角色机制](role-mechanism-20261008.zh-CN.md) | COMPLETE / INCONCLUSIVE / STOP；局部负梯度 cosine 不证明持续角色冲突；三角色已有独立 Q 输出列、等权分层采样 |
| [batch512](batch512-20261004.zh-CN.md) | 三 seed 全 FAIL，REJECT；不进 final holdout、不加第四 seed，默认 batch256 不变。梯度/Adam 与 replay sampling intensity 同时变化，未完成因果归因 |

Ataraxos 的论文结果不补足这些历史协议缺口，也不构成重新开启 NTP、batch512、角色改造或 G 的理由。

## 第一门槛：公开权重是否成套且可复算

核查对象是 **Dou Dizhu** 的 MARVEL policy 与配套 Yomi belief；若使用当前默认 `local-subgame`，policy checkpoint 还必须含兼容的 CentralCritic Q 权重。不能用 Stratego 权重、第三方对手权重或随机初始化替代。

2026-10-09 实际读取官方 GitHub API 与固定提交文件，结果如下。这里的“未找到”限定于所列公开入口，不声称穷尽互联网或作者私有存储。

| 入口 | 实际证据 | 判定 |
|---|---|---|
| [官方提交](https://github.com/AtaraxosAI/doudizhu/commit/acdcf199cf422e2b683cb22142b8e7cb33b14460)、[递归文件树 API](https://api.github.com/repos/AtaraxosAI/doudizhu/git/trees/acdcf199cf422e2b683cb22142b8e7cb33b14460?recursive=1) | `truncated=false`；186 个 blob。已下载源码文本并逐个重算 Git blob SHA1，与树中身份一致。没有 `.pt/.pth/.ckpt/.safetensors/.onnx/.npz/.zip` 文件 | SOURCE_VERIFIED；不是模型验证 |
| [Releases API](https://api.github.com/repos/AtaraxosAI/doudizhu/releases)、[Tags API](https://api.github.com/repos/AtaraxosAI/doudizhu/tags) | 均返回空数组；未发现 release 权重附件 | 权重 NOT_VERIFIED |
| [README][readme]、[运行说明][notes]、[CheckpointStore][store] | 仅说明本地路径或 `wandb:<entity>/<project>/<run_id>[:<iteration>]` 的解析方法；未给可定位的公开成套模型。测试中的 `run123/run456` 等为 mock 输入 | locator 语法不是下载来源；NOT_VERIFIED |
| [权重询问 issue #1](https://github.com/AtaraxosAI/doudizhu/issues/1) | 2026-10-06 建立，截至核查为 OPEN、0 comments。提问者也未找到 policy/belief；没有作者答复可作为发布证明 | 旁证，不是“作者确认不存在” |
| [Nature 论文][paper]的 Code/Data availability、[官网](https://ataraxosai.github.io/) | 论文指向官方代码组织；官网展示 Stratego 对局档案；本次未取得斗地主权重定位符、配套哈希和复现回执 | 论文有结果不等于权重已交付 |

未取得权重字节，因此没有权重 SHA256、配对元数据、模型装载或输出复算结果；这些全部 **NOT_VERIFIED**，不能写为 PASS。未登录或枚举私有 W&B，未向作者发消息，未下载其他游戏权重充数，未自训补齐。即使后续出现下载地址，本轮禁止推理的约束仍然有效。

## 论文证据与代码默认值要分开

[Sokota 等，Scalable decision-making for games of imperfect information][paper]，Nature 658，55–59，发表于 **2026-09-30**，DOI `10.1038/s41586-026-11036-y`。其思路是受抑制的自博弈策略更新与隐信息下的测试时搜索。斗地主结果按地主对两位独立农民、交换阵营的 duplicated deals 报告：Ataraxos 对无搜索 policy 的角色平均得分为 `0.107 ± 0.012`，对 PerfectDou 为 `0.199 ± 0.015`（标准误）。这是作者报告，本任务未复算；不是 DouGPU 的 WP/ADP 提升。

[补充材料 S7、S9][supp]给出斗地主的自回归隐牌归属 NLL 学习、分解动作前缀和 rollout 搜索：每个合法续接采样 `ceil(200/|A|)` 隐手，rollout 到终局，再以步长 5 做抑制策略改进。S7 所列 policy 与 belief 训练资源仅是论文方法背景，不形成本项目预算；本报告不提出复训申请。

当前源码 [engine][engine]、[agent 装配][worker]与 [notes][notes]却以 `local-subgame` 为默认：Yomi 隐状态粒子 → 局部子博弈 → CentralCritic 叶节点估值 → 带策略先验的 PCFR。另一条 `update-equiv` 使用 rollout 估值和 `pi_new(a) ∝ pi_old(a) * exp(beta * Q(a))`。因此默认入口不能直接称为论文 S7.4 的复现；即使选择 `update-equiv`，仍需权重、参数、动作分解与评测输入的对应证据，不能仅凭同名宣称数值等价。

## 源码、依赖、许可与权重来源

上游快照为 `acdcf199cf422e2b683cb22142b8e7cb33b14460`；API 返回提交日期 `2026-05-26T05:02:21Z`。README 称其从 `nanogamekits` 单提交拆出；该声明不证明完整研发历史或与论文最终运行版本一致。本次仅将公开源码放在 `/tmp/ataraxos-research/source` 阅读，未纳入 DouGPU、未导入执行、未安装。固定提交链接是持久来源；临时目录不是正式证据归档。

| 对象 | 静态读取到的来源与约束 |
|---|---|
| 上游代码许可 | [LICENSE][license] 实为 MIT，Copyright (c) 2026 AtaraxosAI；复制或分发实质代码时应保留版权与许可文本，包含无担保条款。许可文件 SHA256：`16f6008fb22189e67a25ed66a2e4b8a3a79959dd9153c3f1d6db597d0bdca080` |
| Python 栈 | [pyproject.toml][deps]：Python ≥3.10、NumPy ≥2、torch、tqdm、tyro；dev 加 pytest、可选 wandb。[requirements][requirements] 的 dev 列表另有 transformers。多数依赖未精确锁版本，不能视为论文环境锁 |
| 本地原生模块 | 仓库自带 `nanopokerlib` C++/pybind 源码；[构建声明][native-build]要求 C++20；[build-system][native-project]列 CMake ≥3.26.0、setuptools、wheel。构建优先使用现有/本地依赖，否则 FetchContent 拉取 [pybind11 v2.11.1](https://github.com/pybind/pybind11/tree/v2.11.1) 与 [PokerHandEvaluator v0.5.3](https://github.com/HenryRLee/PokerHandEvaluator/tree/v0.5.3)。本次未触发构建或下载这些依赖 |
| 外部评测对手 | [PerfectDou 安装脚本][perfectdou]指向 `Netease-Games-AI-Lab-Guangzhou/PerfectDou`，使用 Python/ONNX 相关环境；[FPDou 脚本][fpdou]默认 Google Drive 文件 ID `12NyPl7aOivItoPMnNei2A-F0KMHVsa-U`。它们是可选对手来源，不是 MARVEL/Yomi 预训练包；未执行脚本、未验证这些权重 |
| 权重与授权 | MARVEL/Yomi 的公开文件、配对身份、哈希和具体授权均 NOT_VERIFIED。仓库 MIT 不能自动替未取得的外部权重或所有第三方依赖证明许可；本轮没有引入或分发其代码/权重 |
| DouGPU 现状 | [CPU 依赖](../../requirements-cpu.txt)固定 JAX/JAXLIB 0.7.2、NumPy 2.5.3 等；[GPU 依赖](../../requirements-gpu.txt)另有 CUDA13 路径，本轮未调用。现有 `upstream_cache/source_lock.json` 记录规则来源 `kwai/DouZero@718a5c920bf3361e34178a38f3b80458e176b351`、encoding_schema=1；本次仅读取，未 bootstrap 或改锁 |

## 实际接口比较

以下基于固定提交源码静态追踪，未用模型输出确认运行兼容性。

| 边界 | DouGPU `0576956` | Ataraxos 斗地主快照 | 可行性含义 |
|---|---|---|---|
| 学习与输出 | [model.py](../../dougpu/model.py)：`encode_state` → `q_values`，完整合法动作逐项标量 Q；三角色共享主干但有独立输出列，MC 目标 MSE | [VRPO][vrpo]：PyTorch Agent 策略 logits、clipped policy-gradient 与独立 CentralCritic Q 目标；另有 IPPO 注册入口 | 不是把 JAX 换成 Torch；Q 标量也不等于上游策略分布或集中式 critic |
| Belief | `sample_losses` 的 sigmoid 30 维手牌计数回归（两位其他玩家各 15 ranks，计数归一化）+ MSE；是辅助目标 | [Yomi loss][belief-loss]：合法候选 token 上 masked NLL；[Yomi connector][belief]自回归采样并重建 determinized histories | DouGPU 边际预测未定义合法联合隐手分布，不能直接提供 Yomi 粒子或 likelihood |
| 推理与搜索 | [inference.py](../../dougpu/inference.py)：按合法动作 Q 选取，训练可 ε 探索；没有对应的隐状态搜索接口 | [Kagami interface][interface]：`infer_batch(infoset_strs, pending_actions)` 返回动作字符串、概率及 `is_commit`；`act_batch` 迭代前缀至提交 | 即使对 Q 做 softmax，也没有恢复训练策略、belief 或前缀概率语义；本轮不写适配器 |
| 状态编码 | [encoding.py](../../dougpu/encoding.py)：PublicState 的当前手牌、原始底牌、绝对角色历史和完整合法动作；state/action/belief 维度 21/16/30，历史上限 512 tokens | [infoset.py][infoset]：`L/P1/P2` 字符串，观察者**初始私牌**（地主扣除公开底牌）+ 底牌 + 动作历史；C++ 转 tokens/channels | 需要显式恢复自己的已出牌、映射底牌/PASS/角色与 rank，不能直接复制 token IDs；不能传入真实对手手牌 |
| 原生规则与动作 | [environment.py](../../dougpu/environment.py)复用 DouZero 规则，`step(index)` 提交完整动作；给定 deal 的数组切片发牌 | [C++ 游戏说明][cpp-game]与 [belief endpoint][cpp-belief]：chance 发牌、动作前缀、多次 step 后 commit、infoset 重建及合法隐牌扩展 | 同 seed 不保证同 deal；“PerfectDou-compatible” 是上游声明，现有一致性测试未在本轮运行，规则等价 NOT_VERIFIED |
| 模型存储 | [checkpoint.py](../../dougpu/checkpoint.py)：完整 ZIP 状态与 NPZ policy；policy 导出排除 NTP/belief 参数 | [CheckpointStore][store]解析本地/W&B；[VRPO][vrpo]保存 `agent_state_dict`、`critic_state_dict` 等；Yomi 要求 model/config/iteration | NPZ 不能直接喂给 `.pt` loader；现有 DouGPU policy 导出甚至不含 belief。上游 `local-subgame` 还需 critic，单个 policy 文件名不足以证明齐套 |
| 评测契约 | 本地既有 paired WP/ADP、角色与历史协议身份 | [worker][worker]走 socket worker，Kagami action string 返回 host；论文另有 duplicated-deal / role-averaged-score 口径 | 不能挪用上游均分代替 DouGPU 分角色门槛，不能凭论文农民侧收益解释本地角色机制 |

源码可复用的主要价值是接口和方法参照。现有 Q+MSE-belief 与上游 policy+生成式 belief+搜索存在实质契约差异；在权重门槛尚未通过时实施迁移，无法区分接口错误、模型缺失与算法收益，故不进入实现。

## 单项必要条件与 STOP

**继续研究的唯一必要条件：官方提供可公开取得、可固定身份且可核对配套关系的斗地主复现包。** 一个包应同时交付 MARVEL policy、配套 Yomi belief（选择 `local-subgame` 时含兼容 critic）、各文件哈希及授权、对应源码/配置/依赖身份，以及指定搜索路径的冻结评测输入、输出与复算说明。它是同一个“可审计复现包”条件，不用自训、第三方替代权重或仅发布下载计划来满足。

该条件现在未满足；它只是重新做静态资格审查的必要条件，不是充分条件，更不是自动批准运行。即使满足，本任务仍在文档闭环后 STOP；任何后续运行须另行明确范围，不在此申请 GPU 预算。

不训练、不推理（包括 CPU forward）、不做梯度/优化器探针、不造新牌或启动评测；不安装上游软件、不引入依赖、不调整 belief/NTP/LR/batch/Replay；不重开旧结案，不提交实验协议或资源申请。**NO_GO / NOT_VERIFIED / STOP；G_NOT_ENTERED / TRAINING_PAUSED。**

## 本次最小检查

仅做 Git 身份/差异、UTF-8 文档及本地链接检查；上游 186 个源码 blob 身份校验通过。未运行训练、推理、上游测试或全仓测试；GPU 使用为 0。Nature 普通页面曾被 cookie 重定向阻断，后读取其官方静态页面及补充 PDF；没有以第三方报道代替论文。

实际检查结果：`git diff --check` 通过；标准库文档检查退出码 0，31 个本地链接、26 处引用式链接及 18 个固定提交源码文件链接检查通过（源码链接按已下载文件核对，未声称逐一 HTTP 检测）。改动集合严格为本报告与 README；README 原有行全部保留，其他 tracked 文件与基线一致。新报告尚未跟踪，普通 `git diff --stat` 只显示 README 的 3 行新增，不能据此漏计本报告。

[paper]: https://www.nature.com/articles/s41586-026-11036-y?code=static&error=cookies_not_supported
[supp]: https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf
[readme]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/README.md
[notes]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/docs/doudizhu.md
[license]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/LICENSE
[deps]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/pyproject.toml
[requirements]: https://github.com/AtaraxosAI/doudizhu/tree/acdcf199cf422e2b683cb22142b8e7cb33b14460/requirements
[native-build]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/nanopokerlib/CMakeLists.txt
[native-project]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/nanopokerlib/pyproject.toml
[perfectdou]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/scripts/install-perfectdou.sh
[fpdou]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/scripts/install-fpdou.sh
[store]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/marvel/runtime/checkpoints.py
[vrpo]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/marvel/training/vrpo.py
[belief-loss]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/yomi/training/loss.py
[belief]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/kagami/connectors/yomi.py
[interface]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/kagami/core/interfaces.py
[engine]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/kagami/games/doudizhu/engine.py
[worker]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/evaluations/doudizhu/agents/marvel.py
[infoset]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/evaluations/doudizhu/runtime/infoset.py
[cpp-game]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/nanopokerlib/cpp/games/doudizhu/README.md
[cpp-belief]: https://github.com/AtaraxosAI/doudizhu/blob/acdcf199cf422e2b683cb22142b8e7cb33b14460/nanopokerlib/cpp/games/doudizhu/endpoint/doudizhu_endpoint_methods_belief.hpp
