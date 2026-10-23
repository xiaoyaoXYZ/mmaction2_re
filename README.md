# MMAction2 工业动作分类实验重建

本次分类基线已经在本机 NVIDIA GeForce MX350（2GB）上完成：真实数据前向/反向、1 个完整 epoch 的训练与验证、保存模型、在新进程重新加载模型并复核预测。结果来自本次运行，未使用旧日志冒充结果，未设定准确率目标。MAE 尚未实现或运行，作为分类基线之后的独立阶段。

## 每一步做了什么，如何确认

| 步骤 | 实际操作 | 成功依据 |
| --- | --- | --- |
| 电脑与原环境 | 枚举 GPU/驱动、Python、磁盘和已有 Conda 环境的软件包 | [初始清单](reports/host_inventory_before.json)；MX350 2048MiB，驱动576.83；最终原环境包名/版本清单一致 |
| 隔离依赖 | 在 `.venv` 新建 Python 3.9.17 环境，禁止继承旧 site-packages | [环境记录](reports/environment_final.json)、[版本锁定](requirements-lock.txt)；pip check通过；CUDA实际算子与梯度探针通过 |
| 数据与标签 | 解压并核查全部3037帧、原标签；新建0–9类别映射、10/10片段划分；处理精确重复区间 | 全量解码、CRC、帧编号、解压后SHA256通过；有效训练1545帧、验证1401帧；实际采样区间精确哈希交集0；[数据汇总](reports/data_summary.md) |
| 前向与反向 | 完整Swin Tiny约27,858,160参数，2个真实batch逐次训练 | 损失分别2.266087、2.522676；主干171个及分类头2个参数张量均有梯度，有限且整体非零；两部分参数均更新；[原始结果](work_dirs/smoke_gpu_01/result.json) |
| 一轮训练验证 | MMEngine Runner训练10个batch，随后验证全部10段 | epoch=1，iter=10；损失均有限，均值2.590541；[训练结果](work_dirs/train_gpu_01/result.json)、[启动日志](logs/train_launch_01.log) |
| 保存与重载 | 保存权重、优化器、配置、随机种子、日志；新进程strict加载并逐段重复预测 | 无缺失/额外键；10段概率正常、重复预测误差≤1e-6；重载指标与训练后验证完全一致；[重载结果及混淆矩阵](work_dirs/evaluate_gpu_01/result.json) |

## 实测结果与限制

- 验证 Top-1：**20.0%（2/10）**；Top-5：**80.0%（8/10）**；平均类别准确率：**20.0%**。
- 这是随机初始化后一轮训练的工程验证，没有独立测试集，不能据此宣称工业部署或泛化效果。不能从10段验证片段推断稳定准确率。
- 该 `ucf101.zip` 是10类工业装配动作的帧序列，**不是公开UCF101基准**，也不含原视频；帧率、采集者/场次/原视频分组未知。
- 原划分让两类完全没有训练数据；原类别 `anya_mopping` 对应实际目录 `an_mopping`。原目录及标注均保留，纠正只体现在新清单与映射。
- 3对片段含26张跨片段完全重复帧，提示来源重叠。3个验证片段分别使用1–72、1–143、1–123的连续编号前缀；尾部91帧仅排除采样，原图不删除。消除精确重复**不等于来源独立**，已在[数据质量报告](reports/data_quality.md)记录。部分片段内部视角变化也提示可能拼接。

## 最终模型和依赖

模型为官方 MMAction2 **SwinTransformer3D(arch='tiny') + I3DHead(10类)**。保持完整网络结构，输入8帧、帧间隔2、112×112、batch=1、FP32、gradient checkpointing，SGD(lr=0.001, momentum=0.9)、种子20261017；关闭水平翻转以保留左右动作语义。没有加载预训练权重。相对Starter的16帧224×224/AdamW/梯度累积方案，这是适配2GB显卡的分类流程基线，不能称为原论文同设置复现。小批量检查的PyTorch峰值张量分配显存为517.2MiB，不包含全部驱动/显示开销。

Python 3.9.17；PyTorch 2.1.2+cu118；torchvision 0.16.2+cu118；MMAction2 1.2.0；MMEngine 0.10.7；mmcv-lite 2.1.0；NumPy 1.26.4。完整依赖见 `requirements-lock.txt`。旧源码0.24.1、旧C3D配置和原环境均未被替换。

`mmcv-lite` 不提供检测任务的编译算子。本实验入口按需注册已安装的官方分类模块，避免加载无关检测模型；模型层、损失、数据变换、优化器和训练循环均使用官方实现，未修改site-packages源码。请使用本目录的 `scripts/run_baseline.py`；此环境不承诺运行检测、姿态或MMAction2所有示例。模块源码SHA256记录在每个结果JSON中。

## 主要产物

- [完整检查点（含优化器，约226.7MiB）](work_dirs/train_gpu_01/epoch_1.pth)
- [本轮最佳权重](work_dirs/train_gpu_01/best_acc_top1_epoch_1.pth)（仅1轮，“最佳”只表示这一轮）
- [配置](configs/swin_tiny_industrial.py)；每次输出目录还保存展开后的配置及数据清单快照。
- [逐片预测、混淆矩阵](work_dirs/evaluate_gpu_01/result.json)；[总览JSON](reports/experiment_summary.json)。
- `logs/` 保存安装、失败下载、启动与版本检查日志；`work_dirs/` 保存本次运行日志和MMEngine指标。
- 原始ZIP位于 `E:\Codes`，只读使用。解压副本位于 `data/ucf101`，新标注位于 `data/splits`。不需要额外视频解码器来读取本次JPG帧序列。

完整检查点 SHA256：`1cc648417efff8ad7664fd258841de760d5fe1c455a3aaec5eee46bf9831d7bb`。

## 重跑

在PowerShell进入本目录后运行（不需要激活或改变全局环境）：

```powershell
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe .\scripts\run_baseline.py smoke
& .\.venv\Scripts\python.exe .\scripts\run_baseline.py train
# 将路径替换成上一步新建的train输出目录
& .\.venv\Scripts\python.exe .\scripts\run_baseline.py evaluate --checkpoint .\work_dirs\train_gpu_01\epoch_1.pth
```

也可执行 `./run_experiment.ps1`，按顺序检查、训练、保存并重载。每次使用新输出目录，已完成的运行不会被覆盖；任一步失败即停止。不要使用其他Conda环境中的python或旧0.x的训练入口。

若需重建同一软件环境，使用Python3.9新建另一个venv，再执行：

```powershell
python -m venv .venv_recreated
& .\.venv_recreated\Scripts\python.exe -m pip install --find-links .\downloads --extra-index-url https://download.pytorch.org/whl/cu118 -r .\requirements-lock.txt
```

`downloads/` 保存已校验的核心安装包；其余依赖需从PyPI下载。原 `.venv` 使用 `D:\Anaconda\envs\mmaction2` 的Python解释器/标准库创建，包目录独立且 `include-system-site-packages=false`，因此不要删除该基础Python安装。首次下载出现截断/哈希失败，均未安装；最终官方CUDA轮包通过完整SHA256校验（见 `downloads/*.verified.json`）。

## MAE 后续阶段

本次没有MAE训练结果。后续在单独目录、配置和日志中实现，并先明确图像MAE或视频掩码自编码器及编码器结构；不能把Swin分类权重当作MAE预训练权重。预训练仅使用训练侧数据；验证片段及其来源重叠问题需要先解决，才能进行可信的预训练/微调对照。MX350 2GB下先验证小规模掩码重建、编码器权重加载，再讨论扩展；不设定准确率承诺。

## 核对过的官方资料

- [PyTorch历史版本及CUDA11.8组合](https://pytorch.org/get-started/previous-versions/)
- [MMAction2 1.2.0版本约束](https://github.com/open-mmlab/mmaction2/blob/v1.2.0/mmaction/__init__.py)
- [MMCV安装与lite差异](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html)
- [官方Video Swin实现](https://github.com/open-mmlab/mmaction2/blob/v1.2.0/mmaction/models/backbones/swin.py)

Starter包内说明作为待审查资料；本次操作范围以用户的重建请求为准。详细核查见 [模型包审计](reports/package_audit.md)。
