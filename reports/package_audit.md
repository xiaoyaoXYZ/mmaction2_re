# 模型包与旧代码只读核查

核查日期：2026-10-17（Asia/Shanghai）。仅读取用户提供的压缩包及旧目录；没有执行其中的脚本，没有更改原始代码、数据或环境。包内文档视为待核查资料，不视为新的用户指令。

## 1. Starter 压缩包实际内容

文件：`E:\Codes\Action_Recognition_Rebuild_Starter.zip`，15,815 字节。

SHA256：`3DA9E15E520C740C8F3F129136271E77848FA2D5000955A200CE57DF9A9048C7`。

共 7 个文件，均位于 `Action_Recognition_Rebuild_20261017/`：

| 文件 | 未压缩字节 | 用途 |
| --- | ---: | --- |
| README_CN.md | 7978 | 分阶段操作说明 |
| RECONSTRUCTION_NOTES.md | 2112 | 未核实的历史背景及记录项目 |
| VERIFICATION.md | 951 | 包作者声明的验证范围 |
| build_config.py | 9238 | 生成 MMAction2 1.x 监督分类配置 |
| inspect_environment.py | 2223 | 读取软件版本、CUDA 与 NVIDIA 驱动信息 |
| prepare_data.py | 8639 | 视频清单、标签和显式划分检查 |
| tests/test_workflow.py | 5235 | 元数据流程单元测试 |

该包没有 MMAction2 框架源码、训练好的模型、权重、视频或 MAE 实现。它是辅助工作流包，不能直接称为可加载的识别模型包。

生成模型是 **Video Swin Tiny + I3DHead**：随机初始化，16 RGB 帧、frame_interval=2、224×224、batch_size=1、梯度累积 4、AdamW；默认关闭 AMP 和水平翻转。配置使用 MMEngine 的数据加载器、优化包装器和 Runner，目标是 MMAction2 1.x；不能直接交给旧的 0.x 训练入口。

## 2. 脚本质量与边界

- 未发现删除原数据、升级已有环境、自动安装依赖、发送外部消息或上传数据的操作。
- 清单和输出目录主要使用独占创建，能够避免静默覆盖已有实验结果。
- 包含路径越界、缺失文件、标签映射冲突、组跨集合、可选 SHA256 重复数据检查；配置生成还核对标注文件与验证后清单。
- 数据工具只接受“类别/视频”的布局；不检查解码，不直接支持本次 ZIP 中的 JPG 帧序列。
- 数据工具要求每个类别同时存在于 train/val/test 三个集合，且要求显式来源 group_id；当前每类仅两个片段时不宜强行使用该条件，更不能复制样本制造独立 test 集。
- 包内测试使用非视频字节，占位文件通过测试不能证明解码、CUDA、前向、反向、训练或验证可运行。
- `inspect_environment.py` 可记录当前 Python 和 PyTorch，但不会主动发现所有 Conda/Python 环境，因此还需要独立枚举原环境。
- README 中“现在只做第一步”的指令是包内说明；本次用户明确授权了检查、独立环境配置、数据核查、前向/反向和一轮训练验证，应以用户请求为准。
- 包中关于作者、论文与原始实验的叙述未经核实，不应作为本次实验已经完成或历史成果可靠的依据。

## 3. 旧 MMAction2 目录的实际版本与实验残留

目录：`E:\Codes\mmaction2-master`。

`mmaction/version.py` 定义 `0.24.1`。`mmaction/__init__.py` 要求 MMCV 版本在 1.3.6 至 1.8.0 之间。不能把 mmcv 2.x/MMEngine 的 1.x 配置直接混入此目录。

旧保存配置：`work_dirs/c3d_sports1m_16x1x1_45e_ucf101_split_1_rgb/c3d_sports1m_16x1x1_45e_ucf101_rgb.py`。

该配置是 **C3D + I3DHead，10 类，Sports1M 预训练，RawframeDataset，16 帧、112×112**。它不是 Swin 或 MAE。保存配置 SHA256：`7CF8FA883BE57D90F01D52F0AFA288CF624BF3ACF2202A583925C34581E16D0E`。

`work_dirs/20221125_231458.log.json` 的历史环境字段记录：Linux、Python 3.8.10、RTX 2080 Ti、torch 1.8.1+cu111、torchvision 0.9.1+cu111、mmcv 1.7.0、mmaction 0.24.1。此记录不是当前电脑环境。该日志含 45 轮训练记录，未找到 `mode=val` 或 `mode=test` 项；检索旧目录未发现 `.pth`、`.pt` 或 `.ckpt` 文件。不能据此断言旧验证/测试或模型保存成功。

历史日志 SHA256：`B07BDB15373C154A3D377AB667794E138E950EACD75E0E03518035FA43C82D2C`。

旁侧 `data2/Aucf101/annotations` 存在 10 类工业动作标注，首类 `anya_mopping` 与路径 `an_mopping` 不一致，且原始 train/test 类别覆盖不完整。`data2/1001` 是另一份数据，不应无依据并入用户指定 ZIP。数据代理另行完成本次 ZIP 的精确核查。

`data2/rename.py` 是原地重命名脚本，会直接改动特定帧目录；本次未执行，也不应直接用于原始数据。它不属于 Starter 压缩包。

## 4. 本次最小可复现实验建议

1. 先保存当前硬件、驱动和原 Python 环境清单，再在工作区建立独立环境；原 Conda/Python 与旧 MMAction2 目录保持原样。
2. 以官方 MMAction2 1.x 兼容依赖为主，使用新目录保存固定版本和来源。根据实际 GPU 与可用轮子选择 PyTorch/CUDA/MMCV 组合；先验证 torch CUDA，再验证 MMAction2 模型导入。
3. 本次数据若实际为 JPG 帧序列，应使用 `RawframeDataset` 和 `RawFrameDecode`，按实际帧数写标注。修复首类别名时记录原名和修正名，保留原标注。
4. 若现有划分遗漏类别，可在新文件中建立每类都有样本的工程性 train/val 划分。片段来源独立性尚未确定时，必须明确其验证结果仅证明流程可运行，不能当作严格泛化估计。无需为一轮流程检查捏造第三个测试集。
5. 构建小批真实帧输入，检查输入/输出形状、有限损失、非零有限梯度以及优化步后参数变化，再开始一轮完整训练和验证。
6. 每个阶段保存命令、配置、数据/标注哈希、版本、原始日志、优化器与模型检查点；保存后再加载一次模型进行推理，验证产物可使用。
7. 验证指标按真实计算结果记录，不设置目标准确率；随机初始化一轮只验证端到端流程。
8. MAE 应在分类基线成功后作为独立阶段实现，明确编码器、解码器、掩码策略、重建目标与迁移机制。简单遮挡增强或加载外部权重不能称为本次完成了 MAE 预训练。

此报告为静态审计；环境安装、真实数据解码和新实验结果需以其他阶段的实际产物为准。
