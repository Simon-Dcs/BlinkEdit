# 完整指令编辑的增量效率实验

从仓库根目录执行；复用 README 的 `embodiedgen` 环境，以及上一轮已下载的官方资产。只新增实验编排，不改生产几何算法、GPT 认证或模型。

```bash
source scripts/efficiency/env.sh  # GPU 7, native Codex backend, gpt-6-astra
codex login status              # 应显示 Logged in using ChatGPT
python scripts/efficiency_optimization/test_protocol.py
python scripts/efficiency_optimization/benchmark.py setup
python scripts/efficiency_optimization/benchmark.py pilot
```

`setup` 创建新目录并写入 `outputs/efficiency_optimization/latest_run.txt`；默认拒绝覆盖已有任务。首次实施留下了 3 次 v1 试跑和 3 次修订试跑，修订试跑用 `pilot_finish`；后续复跑稳定候选无需重做历史故障。

确认试跑产物后，运行预先固定的 60 轮正式对照：

```bash
python scripts/efficiency_optimization/benchmark.py formal
python scripts/efficiency_optimization/analyze.py
```

可用 `--root <实验根目录>` 指定某次实验。程序串行调用模型，冷流程每轮启动 Python，M2 通过标准输入复用同一 worker 和两个 SAPIEN 场景；每轮仍在计时范围内复制输入、重新 collect 原生状态并移除旧模拟对象。常驻执行器不常驻 Codex 模型会话。

## 如何读结果

- `manifest.json`：冻结的指令、顺序、配置、种子和阈值。
- `metrics/tasks.csv`：所有单次值；`phase` 区分试跑与正式实验。
- `metrics/requests.jsonl`：逐请求真实 usage；原始 prompt、JSON 事件和响应保存在各任务的 `request_*` 中。
- `metrics/sessions.csv`：四轮会话完整成本，包括首轮及重复删除。
- `metrics/comparisons.csv`：同期同任务、同基准种子的配对结果。
- `comparisons/`：性能图、场景前后图和全量结果联系表；第一重复 T4 的视频位于对应任务目录。

主指标 `verified_s` 从控制器收到指令/启动冷进程开始，连续计至验证和发布完成。包含输入复制、模型 CLI 往返、原生编辑、前后各两视图和适用的 2 秒释放诊断；视频编码在终点之后单列。`edit_s` 截止原生编辑及 floorplan 写出，尚未完成验收。

`input_tokens` 已含 cached input；reasoning 也是 output 的子项。总 token 只相加 input 与 output。当前监督聊天用量不在 CLI usage 中。模型往返是 CLI、网络、远端等待和计算的合计。

`sessions.csv` 的 `verified_s` 为四轮验收区间之和，`session_wall_s` 为首轮启动至第四轮验收的连续实测墙钟，包含轮间退出及调度开销。原始 worker 记录中的 `bootstrap_s` 在后续常驻轮次实际表示进程启动后距本轮接收的时间，不能当成每轮导入成本；逐轮启动/接收开销使用控制器记录的 `process_overhead_s`。

`*_consumption.json` 中 `requests` 和 `known_tokens` 是截至该阶段结束的实验累计值；`wall_s` 是该阶段的墙钟。分阶段用量应按 `metrics/tasks.csv` 的 `phase` 汇总。

先看质量，再看预设阈值：P1 的完整会话总 token 降幅至少 10%；P2 的 T3/T4 计入首次初始化的平均验收时间降幅至少 15%。只有两者均有价值才运行 `combination` 的额外 12 轮，不为凑正结果继续改 prompt。

控制器按全实验累计请求数/已知 tokens 限制下一轮启动；正式运行含组合累计 2 小时。每次请求最长 180 秒，每任务最长 600 秒。达到 GPU 冲突、订阅认证故障或连续三次模型失败等停止条件时，保留目录及日志，先查明原因再启动新实验。
