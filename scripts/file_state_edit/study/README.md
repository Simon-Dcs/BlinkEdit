# 编辑路由正式对照

数据来自上一轮冻结的两组清单。默认三种方法是 `F0`（全模型）、`F0R`（重读＋规则）和 `F1R`（持久化＋同一规则）。结果在 `outputs/file_state_edit/20261119_edit2/`。使用项目的 `embodiedgen` 环境并从仓库根启动。

```bash
export PYTHONPATH="$PWD:$PWD/scripts/file_state_edit" CUDA_VISIBLE_DEVICES=6,7
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GPT_PROVIDER=codex
S=scripts/file_state_edit/study
python "$S/prepare.py"        # 复用原数据，按 archive 和链续传；首次设置运行清单
python "$S/inventory.py"      # 逐项检查原始 GLB、条件图、隐藏视图与掩膜
python "$S/launch.py" offline edithero
python "$S/launch.py" offline 3dcodeedit
```

正式订阅预算获准后，将确认的 `calls`、`tokens`、`seconds` 填入该次运行的 `metrics/formal-limits.json`，才运行以下命令。离线模式不会发模型请求；已渲染的候选只补反馈审查，未分配模型的轮次从各自真实产物续跑。不要把待反馈计为接受，或把尚未评分填为零分。

```bash
python "$S/launch.py" online edithero --workers 4
python "$S/launch.py" online 3dcodeedit --workers 4
python "$S/score.py" edithero   # 固定视角 GPU6/7，官方坐标无损十进制展开
python "$S/score.py" 3dcodeedit # 最终产物由 GPT 按清单逐项盲评；保持与反馈分离
python "$S/geometry.py"         # 对已实际本地生成的程序补充独立网格测量
python "$S/report_tables.py"    # 按轮、按方法、按组统计三笔用量，生成配对结果及 Blender 表
python "$S/billing.py"          # 从 CLI 回执独立核对用量、图像张次和订阅挂钟时间
python "$S/panels.py" && python "$S/comparison_panel.py" && python "$S/cost_figure.py"
```

`fixed_answers.py` 是单列的历史开发集模型答复重放，`replay.py` 比较在完全相同的已执行动作下的本地索引成本。两者均不作为新模型调用或正式 benchmark 的替代。数据许可见运行目录 `setup/ATTRIBUTION.csv`、`setup/LICENSE-DATA.md` 和相邻 3DCodeEdit 仓库的 `data/ATTRIBUTION.csv`。
