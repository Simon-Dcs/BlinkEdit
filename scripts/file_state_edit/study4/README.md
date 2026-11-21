# 比较完整编辑、源码补丁和有运行证据的补丁

[实验报告](../../../edit4.md) · [已冻结的任务、顺序和输入状态](../../../outputs/file_state_edit/20261120_edit4/manifest.json)。运行已有结果的统计不需要登录模型、重新渲染或下载数据：

```bash
source outputs/file_state_edit/20261120_edit4/logs/start-env.sh
python scripts/file_state_edit/study4/report_data.py
python scripts/file_state_edit/study4/tables.py  # 所有终稿评分完成后才生成表格
```

**复跑用新的输出目录。** 在仓库根目录激活 `embodiedgen` 环境，设置 `EDIT4_ROOT` 为 `outputs/file_state_edit/` 下未使用的绝对路径，令 `EDIT3_ROOT=$EDIT4_ROOT`；然后执行 `python scripts/file_state_edit/study4/design.py`。清单使用已下载的公开数据，以及上一轮各资产实际交付的 GLB 与历史。编辑器只能看到公开指令，不能访问评分清单或标准部件表。

- **程序编辑**：F1 用确定性规则或整个程序；F2 保留规则，按静态源码选片段；F2T 的 `trace_blender.py` 在隔离的 Blender 中记录创建网格时的调用栈和参数，只保留运行结束时仍在场景中的对象。`embodied_gen/harness/asset_edit/live_program.py` 把目标部件与实际创建语句联系起来，没有足够证据就走完整编辑。`coverage.py` 可对 24 个输入离线重放首次选路，`cabinet.py` 重新测量交付柜体的几何尺寸。
- **EditHero 计时**：`pairs.py` 固定 8 个输入状态和相应历史，每个状态独立执行两种方法各 3 次，成对、顺序平衡。在一条执行线上，两种方法先后处理同一状态；其他任务可能共享订阅队列。冷启动、编号图、模型（含回退）和写回都计时。官方删除清单和三张隐藏图只在两个计时段结束后用于评分；GLB 渲染固定到物理 GPU 6、7。
- **正式模型调用**：使用项目现有的已登录订阅通道，须有本次授权的 `metrics/formal-limits.json`。模型、图片、反馈和盲评协议见报告。实际运行初期遇到 429，后续使用 `metrics/transport-policy.json` 对编辑、反馈和评分统一排队；队列时间会另存为 `admission.json`。程序实验可用 `study3/run.py edit 3dcodeedit --workers 4` 执行，用 `study4/finish.py grade --workers 3` 完成最终逐项盲评。脚本路径均相对 `scripts/file_state_edit`；环境变量见上面的 `start-env.sh`。
- **中断恢复**：原阶段退出后，`study4/finish.py edit` 或 `study4/finish.py pairs` 继续未结束的任务。模型通道超时会归档，保留已交付的程序或有效反馈；没有可用结果的配对要两边都重做。每次真实调用都保留在预算账本中。正常反馈不通过或模型选错部件不会触发这种恢复。

末尾的两项新想法仅为提案，不包含对应实现或分组。
