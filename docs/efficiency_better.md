# EmbodiedGen 3D Editing：效率优化实验结果

**2026-11-14 · 已完成 · 物理 GPU 7 · `gpt-6-astra / medium`**

按 [`efficiency_plan.md`](efficiency_plan.md) 实现并完成 **60 轮正式实验 + 6 次工程试跑**，共 **66 次真实模型请求、1,009,390 个已知 tokens**。通过项目原生 Codex 后端使用当前 ChatGPT 订阅，未使用 API Key。

**结论：常驻进程确实降低本地开销，但本轮两个候选都没有达到预设的完整流程采用门槛。** 上下文精简的完整会话总 token 仅减少 **0.93%**；常驻进程的本地部分平均减少 **41.72%**，但计入全部模型往返后，验收时间均值反而增加 **2.28%**。因此保留“一次规划 + 原生批量执行”的 M0 为默认，M1/M2 作为可复跑的实验实现保留；按计划不追加 M3 组合。

正式实验中，**48 次可行编辑全部完成**，另有 **8 次正确的无操作返回、2 次正确拒绝、2 次澄清**。这些非编辑结果单独统计，没有合并成“60 次编辑成功”。

![完整指令任务、会话 token 与本地/模型往返时间](outputs/efficiency_optimization/20261114_204906/comparisons/performance.png)

黑点表示全部单次值。M2 的长尾样本完整保留；右图的模型往返包含 CLI、网络、远端等待和计算。

## 1. 实际场景：删掉旧台灯，再放四个物体

下面是预先指定的第一重复 T4：移除办公室地面台灯，在办公室桌上放台灯、花瓶、书和杯子，餐厅不变。三组使用同一相机和分辨率，均调用 EmbodiedGen 原生放置与写回接口。

![T4 输入及 M0、M1、M2 实际输出](outputs/efficiency_optimization/20261114_204906/comparisons/scene_comparison.jpg)

- 2 秒释放诊断视频：[M0](outputs/efficiency_optimization/20261114_204906/runs/M0/formal_M0_T4_r1/release.mp4)、[M1](outputs/efficiency_optimization/20261114_204906/runs/M1/formal_M1_T4_r1/release.mp4)、[M2](outputs/efficiency_optimization/20261114_204906/runs/M2/formal_M2_T4_r1/release.mp4)。
- 全部 60 轮输出的联系表已逐张检查：[1](outputs/efficiency_optimization/20261114_204906/comparisons/contact_01.jpg)、[2](outputs/efficiency_optimization/20261114_204906/comparisons/contact_02.jpg)、[3](outputs/efficiency_optimization/20261114_204906/comparisons/contact_03.jpg)、[4](outputs/efficiency_optimization/20261114_204906/comparisons/contact_04.jpg)、[5](outputs/efficiency_optimization/20261114_204906/comparisons/contact_05.jpg)。

## 2. 实现了什么，怎样接入订阅

新代码位于 [`scripts/efficiency_optimization/`](scripts/efficiency_optimization/)，生产代码及原有入口保持不变。

```text
完整中文指令 + 实际场景清单 + 已确认历史
  → GPT_CLIENT.query → 原生 _query_codex → codex exec
  → 共用 JSON 协议与合法性检查
  → 独立候选场景
  → 原生 delete_object / batch_insert_objects / visualize
  → 几何与资产检查 → SAPIEN 加载、前后渲染、适用的释放诊断
  → 通过后写 published.json；后续轮次只接续已通过的场景
```

| 组别 | 业务上下文 | 本地执行方式 |
|---|---|---|
| M0 主基线 | E1 风格场景清单、完整新指令、必要历史及已确认结果 | 每轮启动新 Python 进程 |
| M1 上下文候选 | 相同事实的紧凑 JSON；复用已确认对象 ID、支撑物和删除记录 | 与 M0 相同 |
| M2 常驻候选 | 与 M0 相同的实际 prompt | 复用 Python、原生执行器、SAPIEN 场景及相机；每次独立任务重新加载输入并清理旧对象 |

M1 的压缩由确定性代码完成，每轮仍调用 GPT；历史删除记录保留，用来识别第四轮的过期指代。每轮重新读取实际场景，计划中的 ID 必须仍然合法。M2 只常驻本地进程，模型请求仍使用原生临时 `codex exec`。六组 M0/M2 复杂任务配对的实际业务 prompt 逐对完全相同。

共用协议区分 `execute / noop / clarify / impossible`。模型只生成受限动作，不生成坐标、缩放或网格编辑代码。验收预期只在独立验收器中，未传入规划 prompt。合法性错误最多允许一次模型纠正；**正式 60 轮均只用一次模型请求，没有额外摘要或纠错调用**。

### 环境与认证

- Conda：`/home/dcs/anaconda3/envs/embodiedgen`；Python 3.10.13，Torch 2.8.0+cu126，NumPy 1.26.4，trimesh 5.1.1，SAPIEN 3.0.0b1；`pip check` 通过。
- 仓库：`f0124197888c2b733e4eaa65acd81ad9cfda3b79`，v2.1.0；Codex CLI 0.167.0。
- `codex login status` 实际返回 `Logged in using ChatGPT`；使用 `GPT_PROVIDER=codex`、`MODEL_NAME=gpt-6-astra`、`GPT_TIMEOUT=180`，沿用项目官方 CLI 登录机制，无需项目 API Key，也未提取登录凭据。
- `CUDA_VISIBLE_DEVICES=7`，SAPIEN 显式使用可见的 `cuda:0`。每轮保存 PID→GPU UUID 证据，均为 `GPU-63a185b7-8463-446a-808b-10425a0e9800`。原生网格处理及放置主要在 CPU，GPT 在远端；GPU 7 承担实际渲染。
- OMP/OpenBLAS/MKL 各 4 线程；原生网格解析最多 8 workers。机器非独占，模型请求和实验渲染串行执行。

完整配置、包清单、任务及执行顺序见 [manifest](outputs/efficiency_optimization/20261114_204906/manifest.json) 与 [setup](outputs/efficiency_optimization/20261114_204906/setup/)。

## 3. 固定条件与计量口径

复用上一轮官方 table、lamp、vase、book、mug，以及仓库 sofa；双房间夹具包含两张桌子、两盏地面台灯及地面。素材出处和版本沿用[上一轮 manifest](outputs/efficiency/20261112_214800/manifest.json)，数据集 revision 为 `58258b50a0fc95034f2f3cc03b332ec6f72b91fd`。本轮测场景级实例编辑，未增加图生 3D、纹理生成、整屋生成或 USD 导出。

所有组统一使用固定资产尺寸、旋转 `(1.5708, 0, 0)`、`top` 策略、10,000 点采样和 2,000 次搜索上限。独立案例基准种子为 42、43、44；原生 trimesh 采样仍独立引入熵，相同种子不代表候选点逐位相同。M0/M1/M2 按预先固定的交错顺序运行，没有根据结果挑任务或重跑慢样本。

**主指标 `T_verified`** 从控制器启动冷进程/向常驻进程派发任务前开始，连续计至验证及结果发布完成，包含导入、输入副本与重置、上下文构造、模型往返、原生编辑、资产和几何检查、SAPIEN 加载、前后各两视图（720×1080）及适用的 2 秒释放诊断。`T_edit` 截止 URDF 和 floorplan 就绪，两者逐任务同时保存在 CSV 中。

视频编码在验收终点之后单列。控制器完整等待时间另存 `controller_wall_s`；它还包含编码、进程退出等收尾。四轮会话同时报告各轮验收区间之和，以及从第一轮启动至第四轮验收的连续墙钟，后者包含轮间退出与调度。初始化素材复制单独记录，每任务的输入复制和常驻重置则计入主指标。

Token 使用后端真实 usage：`input + output`，input 已含 cached input，reasoning 是 output 子项，均不重复相加。当前监督聊天用量不在 CLI usage 中；这里统计的是项目实际发起的模型请求。

## 4. 单轮完整任务：全部单次值保存在 CSV

T1：删除办公室台灯；T2：桌上插花瓶；T3：桌上插入四物体；T4：删除旧台灯并插入四物体。每个表格行均为 3 次重复，时间使用验收终点。

| 任务 | 组别 | 均值 / 中位数（秒） | 最小—最大（秒） | 平均总 tokens | 编辑通过 |
|---|---|---:|---:|---:|---:|
| T1 | M0 | 30.54 / 28.21 | 27.36–36.04 | 15,486.3 | 3/3 |
| T1 | M1 | 26.11 / 25.79 | 19.40–33.14 | 15,163.0 | 3/3 |
| T2 | M0 | 26.18 / 25.21 | 24.61–28.73 | 14,904.7 | 3/3 |
| T2 | M1 | 27.79 / 26.43 | 25.47–31.47 | 14,887.0 | 3/3 |
| T3 | M0 | 31.31 / 31.69 | 28.36–33.88 | 15,359.7 | 3/3 |
| T3 | M1 | 32.50 / 32.50 | 28.92–36.08 | 15,343.3 | 3/3 |
| T3 | M2 | 24.63 / 23.84 | 21.14–28.90 | 15,669.0 | 3/3 |
| T4 | M0 | 29.92 / 29.82 | 26.51–33.45 | 15,390.3 | 3/3 |
| T4 | M1 | 35.22 / 36.37 | 31.76–37.52 | 16,004.0 | 3/3 |
| T4 | M2 | 38.01 / 27.04 | 22.58–64.40 | 15,089.0 | 3/3 |

按全部尝试计算，M0 的 12 次单轮主任务共 353.87 秒、183,423 tokens，即每次成功 29.49 秒、15,285.25 tokens；M1 同样 12 次共 364.85 秒、184,192 tokens，即每次成功 30.40 秒、15,349.33 tokens。M2 只测 T3/T4，不能直接用其六次总体均值对比 M0 的全部四类任务。

完整单次值、两个时间终点、逐请求和配对变化分别见 [tasks.csv](outputs/efficiency_optimization/20261114_204906/metrics/tasks.csv)、[requests.jsonl](outputs/efficiency_optimization/20261114_204906/metrics/requests.jsonl)、[comparisons.csv](outputs/efficiency_optimization/20261114_204906/metrics/comparisons.csv)。

## 5. P1：指代正确，但完整会话 token 仅减少 0.93%

双方都从空历史开始，完成“放杯子 → 在刚才桌上放书、保留杯子 → 删除刚才杯子、书不动 → 再删刚才杯子”。全部 6 个四轮会话均完成前三轮编辑，第四轮正确返回对象已不存在，书和其他对象保持不变。

| 重复 / 种子 | M0 总 tokens | M1 总 tokens | M0 连续会话墙钟（秒） | M1 连续会话墙钟（秒） |
|---|---:|---:|---:|---:|
| 1 / 42 | 60,907 | 61,687 | 99.24 | 129.80 |
| 2 / 43 | 62,784 | 60,503 | 103.04 | 142.67 |
| 3 / 44 | 60,922 | 60,704 | 107.78 | 104.88 |
| 合计 | **184,613** | **182,894** | **310.05** | **377.35** |

M1 总 token 少 1,719，降幅 **0.93%**，低于预设 **10%**。连续会话墙钟均值 M0 为 103.35 秒，M1 为 125.78 秒；中位数分别 103.04、129.80 秒，范围分别 99.24–107.78、104.88–142.67 秒。各轮验收区间之和的平均值分别为 98.45、120.81 秒，见 [sessions.csv](outputs/efficiency_optimization/20261114_204906/metrics/sessions.csv)。

精简只作用于业务上下文，不能消除 CLI 的其他输入开销。另一个实测现象是：M0 三次 T3 提交的业务 prompt 完全相同、均为 1,485 字符，但真实 input usage 分别为 14,846、15,778、14,844。CLI 未暴露完整内部输入构成，因此不能把这些波动都归因于压缩或服务端缓存。

**判断：本轮 P1 不采用，不再继续微调 prompt 凑收益。** 缓存正确性通过了当前四轮测试；外部改名、跨房间移动或支撑物变化没有作为新增扰动任务测量。

## 6. P2：本地平均少 5.21 秒，完整流程均值仍未达标

以下只比较同期、同任务 T3/T4 的六对测量，M2 计入首次进程与渲染器初始化，并计入每次输入加载和重置。

| 指标 | M0 冷进程 | M2 常驻 |
|---|---:|---:|
| 平均 `T_verified` | **30.62 秒** | **31.32 秒** |
| 中位数 | 30.76 秒 | 25.44 秒 |
| 范围 | 26.51–33.88 秒 | 21.14–64.40 秒 |
| 平均模型往返 | 18.13 秒 | 24.04 秒 |
| 平均本地部分（总时间减模型往返） | **12.48 秒** | **7.28 秒** |

本地部分减少 **5.21 秒 / 41.72%**。分阶段的平均启动/派发开销由 5.70 秒降至 1.00 秒，渲染器初始化摊销由 1.07 秒降至 0.19 秒；这支持了复用导入和渲染器能够节省本地时间的假设。

但 M2 第三重复 T4 总耗时 **64.40 秒**，其中模型往返 **58.71 秒**、本地部分 **5.70 秒**。这次长尾发生在 CLI/网络/远端返回链路；现有时钟无法进一步区分原因。它使全部六次的平均总时间增加 **2.28%**。中位数较低不改变计划以摊销平均时间至少降低 **15%** 为采用门槛的约定。

M2 首次请求为 **28.90 秒**；后续五次均值 **31.80 秒**、中位数 23.84 秒，范围 21.14–64.40 秒。六次主任务验收区间合计 **187.90 秒**，摊销每次 31.32 秒，包含首轮而非只选热请求。

常驻进程还用于三次边界检查。其总存续墙钟为 **1,607.82 秒**，本组九次控制器等待区间合计 **258.30 秒**，其余交错等待及停机间隙 **1,349.52 秒**。187.90 秒是六次主任务的区间和，不是一段无间隙的连续会话墙钟。

**判断：P2 的本地优化有效，但本轮完整流程均值不达标，暂不升级为默认。** P1/P2 均未通过采用门槛，因此未启动条件性的 M3 组合及其额外同期基线。

## 7. 边界和质量：正确拒绝、澄清与编辑分开

按单轮复杂任务平均耗时，M2 是两个候选中较快者，因此在边界组与 M0 比较。

| 边界请求 | M0：秒 / tokens | M2：秒 / tokens | 实际结果 |
|---|---:|---:|---|
| 固定方向将超尺寸沙发放到办公室桌上 | 26.49 / 14,991 | 28.91 / 14,997 | 两次均调用原生放置，返回 `None`，最终 `impossible`；未缩放或改桌子 |
| 删除不存在的办公室钢琴 | 23.31 / 14,858 | 19.40 / 14,860 | `noop`，明确未找到，场景不变 |
| 无历史时“把花瓶放在那张桌上” | 25.62 / 14,873 | 21.48 / 14,864 | `clarify`，询问办公室桌还是餐厅桌，场景不变 |

所有正式编辑均通过指定类型/数量、房间、桌面支撑和非目标实例保持检查。URDF 的视觉与碰撞网格存在，实际可被 SAPIEN 加载。保留对象的位姿、视觉网格顶点及碰撞文件内容均与本轮输入一致。

所有插入采用一致的保守 XY 凸包检查：桌面范围容差 2 mm，物体间面积重叠上限 `1e-5 m²`；实测最大重叠面积为 **0**。该检查用于筛查范围和重叠，不替代精确三维碰撞检测。

正式实验共对 **90 个新增物体**做了独立释放诊断：原碰撞网格、摩擦 .6/.5、原生默认密度、1 mm contact offset、240 Hz 模拟 2 秒。全部仍在支撑物上，最大平移 **9.15 mm**，最大旋转 **1.54°**，低于预设 2 cm / 5°。诊断状态不写回正式 fixed-joint 场景。

协议单元测试 7 项通过；[证据审计](outputs/efficiency_optimization/20261114_204906/metrics/audit.json)通过，逐项核对了请求与 raw usage、时钟及嵌套阶段、GPU、输出发布、预览和配对 prompt。已检查五张全量输出联系表、T4 高分辨率前后图及性能图；未见错房间、残留旧对象或连续删除时误删其他物体。

每类独立任务只有 3 次重复，结论用于当前小场景的工程取舍；没有据此计算可靠 p95 或外推整屋编辑成功率。

## 8. 工程试跑、消耗与文件位置

首次三次试跑中，删除不存在钢琴的场景没有改变，但模型返回 `impossible`，与验收器要求的 `noop` 不一致。随后在共用协议中明确缺失删除目标应返回 `noop`，三次修订试跑通过，再冻结 `optimization-v2` 进入正式实验。原失败、v1/v2 源码快照及全部用量保留，试跑不混入正式均值。

| 阶段 | 计时轮次 | 模型请求 | 已知总 tokens |
|---|---:|---:|---:|
| 工程试跑（含 1 次协议验收失败） | 6 | 6 | 92,551 |
| 正式单轮 + 连续会话 + 边界 | 60 | 60 | 916,839 |
| 合计 | **66** | **66** | **1,009,390** |

全部请求的 input 为 **1,002,229**，其中 cached input **831,744**；output 为 **7,161**，其中 reasoning **553**。未知 usage 请求 **0**，模型调用失败 **0**。正式运行控制器墙钟 **1,799.29 秒，约 30 分钟**；未触发 78 轮、100 请求、200 万已知 tokens 或 2 小时正式运行上限。订阅用量未换算为 API 美元价格。

产物根目录：`outputs/efficiency_optimization/20261114_204906/`，约 **1.3 GB**。旧 [`efficiency.md`](efficiency.md) 和旧实验目录保留为历史证据。

- [机器可读汇总](outputs/efficiency_optimization/20261114_204906/metrics/summary.json)
- [逐任务、逐会话和配对表](outputs/efficiency_optimization/20261114_204906/metrics/)
- [实际 prompt、响应、计划、场景、检查与原始 usage](outputs/efficiency_optimization/20261114_204906/runs/)
- [冻结候选源码](outputs/efficiency_optimization/20261114_204906/source_snapshot/candidate_v2/)
- [脚本用法及字段说明](scripts/efficiency_optimization/README.md)

### 复跑

在仓库根目录执行，`setup` 会新建输出目录，避免覆盖本轮记录：

```bash
source scripts/efficiency/env.sh
codex login status
python scripts/efficiency_optimization/test_protocol.py
python scripts/efficiency_optimization/benchmark.py setup
python scripts/efficiency_optimization/benchmark.py pilot
# 检查试跑后，再执行正式实验：
python scripts/efficiency_optimization/benchmark.py formal
python scripts/efficiency_optimization/analyze.py
python scripts/efficiency_optimization/audit.py
```

