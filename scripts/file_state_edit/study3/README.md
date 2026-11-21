# 部件定位和局部代码修改：F2 四组对照

本轮的 **F1** 是上一轮的 **F0＋规则**（磁盘代号 `F0R`），此前称作 F1 的持久化方法改称 **Fz**。历史结果仍用产生时的名字；实验清单的 `method_roles` 在进入执行器时解释新名字。主表复用既有 F0、新 F1 的逐轮回执；其余格子均有独立的新产物。

- [报告](../../../edit3.md) · [本轮清单](../../../outputs/file_state_edit/20261119_edit3/manifest.json)
- `embodied_gen/harness/asset_edit/grounding.py`：读取当前 GLB 的 UV、材质与网格，在可见表面积证据充分时直接选定部件；否则给备选部件绘制编号图。只接受候选中的一个整数编号；不能确定就转回完整编辑。写回后核对所有非目标部件索引。
- `embodied_gen/harness/asset_edit/local_program.py`：根据指令和反馈选择当前源码的相关行，模型仅返回替换范围和内容，程序检查范围、重叠与语法并合回完整文件。不能合并时使用完整编辑器。比例和唯一部件等已有零调用规则优先执行。
- `measure_render.py` 在符合前、中、后三层描述的程序任务中读取当前模型的世界高度；角色不能唯一对应时不产生高度结论。六视角画面仍由官方入口生成，审查和最终裁判不读取内部几何记录。

在仓库根目录使用 `embodiedgen` 环境。已有回执可直接重新汇总，不需要订阅调用：

```bash
export PYTHONPATH="$PWD:$PWD/scripts/file_state_edit"
python scripts/file_state_edit/study3/collect.py
python scripts/file_state_edit/study3/figures.py
```

准备数据用 `setup_run.py --fresh --fetch`。主样本及新样本的选择在运行前固定。完整复跑请先将环境变量 `EDIT3_ROOT` 指向 `outputs/file_state_edit/` 下的新目录，再准备数据并写入已批准的 `metrics/formal-limits.json`。之后，`run.py edit edithero` 和 `run.py edit 3dcodeedit` 分别执行编辑；`grade_ready.py` 在链或程序任务终止后评分。现有目录的已完成编辑及评分会跳过。EditHero 的两条评分队列仅使用物理 GPU 6、7；3DCodeEdit 的官方六视角 Cycles 渲染使用 CPU，每进程 4 个线程。

`metrics/four-requests.csv` 含请求角色、候选定位/局部修改/完整编辑、文字与图像合计 token、图片张次、直接代理工具动作和耗时；`four-turns.csv`、`four-cases.csv` 保留失败、未通过和新旧数据来源。每项盲评是一次独立模型请求，交付用量不包含最终评判。
