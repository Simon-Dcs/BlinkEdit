# 上一轮放的杯子，下一轮还能找到吗？

原生会话使用 `current.json` 选定同时生效的场景和状态。首次传入场景，以后只传会话，新一轮用新的输出目录。可在两个命令之间退出 Python，甚至更换允许使用的卡。

```bash
cd /data2/dcs/EmbodiedGen
source /home/dcs/anaconda3/etc/profile.d/conda.sh && conda activate embodiedgen
export GPT_PROVIDER=codex MODEL_NAME=gpt-6-astra
S=outputs/vista_evolution/20261116_172022
C=outputs/edit_efficiency/20261116_225555/inputs/catalog.json
A=(--engine deterministic --lean-parser --efficient-geometry --session outputs/my_session --catalog "$C")
scene-edit-cli "${A[@]}" --gpu 6 --scene "$S/inputs/layout_3/ambiguous_lamps/scene.urdf" \
  --instruction '在办公室桌上放一个杯子' --output outputs/my_turn_1
scene-edit-cli "${A[@]}" --gpu 7 --instruction '删除刚才放的那个杯子' --output outputs/my_turn_2
```

`published.json` 和 `result.json` 保留单次调用的含义。没有 `--session` 时沿用原有路径，包括 `--history`。会话仅接收静态、base 下固定关节、无额外缩放的 URDF，每个物体有一个视觉和一个碰撞网格；GLB 和 Blender 程序走下面的独立后端。

## 看本次实验

[实验报告](../../edit.md) 给出质量、用量和时间；[运行目录](../../outputs/file_state_edit/20261118_2200/) 保存每条链自己的实际产物。以下命令只汇总已有结果，不调用模型：

```bash
R=$PWD/outputs/file_state_edit/20261118_2200
python scripts/file_state_edit/summarize.py "$R" dev
```

- `metrics/*-cases.csv`：每个方法、每条链/题的完成轮数、质量和总成本。
- `metrics/*-turns.csv`：产物写出、执行检查、反馈结束、离线评分四个时间点/分项；只有实际评分才填写质量。早期试跑产物写出时间用文件的写入时间恢复。
- `metrics/*-requests.csv`：编辑、反馈和最终判分分列，含输入、缓存输入、输出及原始请求记录路径。模型等待使用整次 CLI 往返；直接 Codex 的 shell 等待在其 CLI 时间内。
- `edithero/<阶段>/<链>/<方法>/turnNN`：编辑程序、动作、产物记录、固定视角图和区域指标。`3dcodeedit` 中还可找到匹配原图、修改图、审查和盲评分。
- `setup/native`：真实原生会话的放入、重启与删除。与公开任务表分开计量。

## 复跑条件与边界

使用本机已登录的官方 `codex exec`（`codex login status` 应显示 ChatGPT）、`embodiedgen` 环境、Linux `bwrap`。F0/F1 复用 EmbodiedGen 的 `GPTclient(provider="codex")`。直接代理使用仅允许临时任务工作区和系统工具的命名权限配置；生成的程序在独立的无网络进程运行。不要将整个项目挂入执行进程。

实验使用 `outputs/file_state_edit/tools` 下的独立 Blender 4.2.0、5.1.0、`blender_kit` 和为 Blender Python 3.13 安装的 `blender_deps/scipy`。EditHero 用官方固定视图 GPU 渲染，3DCodeEdit 使用官方 CPU 渲染入口。基准仓库用相邻目录中的固定版本，见运行清单。来源归属见运行目录的 `setup/ATTRIBUTION.csv`、`setup/LICENSE-DATA.md` 和 3DCodeEdit 的 `data/ATTRIBUTION.csv`。

新建运行目录，勿清空已有结果。可复用已下载、未修改的数据文件；新实验需要重新冻结清单和独立账本。只下载新链时，`fetch_chain.py REV RUN/setup FAMILY/CHAIN states|renders|part_masks` 按 tar 偏移提取所选链；`--endpoint` 可指定可用的 Hugging Face 镜像。数据索引文件为 `setup/data/chains.json`、`setup/data/cameras.json`；`setup/hf-mirror.com.json` 为数据仓库元信息（含 revision）。

```bash
# N 是新的空运行目录；复用已核对的数据与版本。
N=$PWD/outputs/file_state_edit/my_run
mkdir -p "$N"/{setup,metrics,comparisons}
cp -a "$R/setup/data" "$R/setup/hf-mirror.com.json" "$N/setup/"
python scripts/file_state_edit/freeze.py "$N"
PYTHONPATH=scripts/file_state_edit python -c 'import sys; from pathlib import Path; from sandbox import probe; probe(Path(sys.argv[1]))' "$N/setup"
# 以下会使用订阅额度，仅在相应预算已获准时启动。
export CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python scripts/file_state_edit/run.py "$N" dev --revisions 1
python scripts/file_state_edit/evaluate.py "$N" dev edithero
python scripts/file_state_edit/evaluate.py "$N" dev 3dcodeedit
python scripts/file_state_edit/summarize.py "$N" dev
```

试跑账本跨进程共用；达到请求、已知 tokens 或时长上限后不再启动新请求。正式阶段还需 `metrics/formal-limits.json` 中**已确认**的 `calls`、`tokens`、`seconds`。运行器跳过已完成轮次，不将无结果的在途目录当成成功，不从参考模型救链。`setup/pilot-source/` 保存此次开发试跑的脚本和 GLB 后端；后续共用后端增加了部件身份、父子安装和程序化纹理，不能把新版复跑当成原样重放。
