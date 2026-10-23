# baseline_v0 封存记录

现有分类基线完整记录：代码、配置、版本清单、原始/当前标注、3037张帧图、日志、逐片预测和两份模型检查点均在 snapshot.zip。
所有条目逐一校验SHA256；清单见 manifest.json，总包校验见 record.json。旧文件在封存时与上轮产物清单一致。

原 `.venv` 环境原地保留，未安装或改动依赖；包版本冻结记录另存 environment-freeze.txt，ZIP含旧环境报告和pyvenv.cfg。ZIP不包含数GB的环境二进制或下载缓存，不能单独当作可迁移Python环境。

恢复时先校验ZIP的SHA256，将内容解压到新的空目录，不覆盖本项目；用Python3.9与requirements-lock.txt重建独立环境。配置中的文件夹模板按新目录展开；旧日志里的绝对路径仅作历史记录。原始外部ZIP的来源路径与哈希记录在归档报告中。

本记录只读使用。后续改动和测试写到audit_stage1或新的版本目录，不重写snapshot.zip。
Top-1 20%仅是当前20片段工程划分上的一轮结果，不是论文复现或可信泛化性能。
