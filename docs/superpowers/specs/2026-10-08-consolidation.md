# SONIC 抓取与数据增广仓库

用户要求：整合 GPT 辅助获得成功参考、SONIC token、单种子增广和最新加速采集；尽量复用源码，接收者能按 README 启动。

采用现有 mobile60 可迁移场景快照，补充原始 teleop/lean、几何筛选、成功种子和 encoder/planner。保留原目录结构；只做路径模板、无需源 Git 的溯源和启动入口适配，不改模型、接触、PD、物理频率、控制历史和成功门限。GPT 指会话辅助设计与实测迭代，不宣称外部 API 已独立获得成功。

交付本地 Git 仓库及包含模型、资产和种子的离线交付包。Git 保留源码、提示词、依赖安装脚本、原始文件哈希、基准证据；大型资源在交付包中。接收者需 Linux x86_64、NVIDIA RTX GPU、驱动和 Python 3.11；提供 Isaac Sim 5.1 / torch 2.7 cu128 / Isaac Lab 快照安装脚本。

统一 CLI 提供 prepare、doctor、candidate、collect、seed-reference 和 export/check。prepare 在新目录物化 __RUNTIME__ 模板并重算内容清单。candidate 复用 compile_candidate/compile_mobile；collect 默认 lean_pair/lean_batch。成功种子不必重新搜索，GPT 迭代新技能是非确定性开发流程。保持所有运行和失败证据，拒绝覆盖输出。

README 区分参考 token 和实际下发 token，逐条 replay 和抽检，以及首次成功和视觉验收。记录旧 50 对的 13.2 分钟平均总耗时、11.7 分钟中位数；精简配对 211.45+163.77 秒仅执行区间。验证包括源码来源检查、迁移目录启动、真实候选编码、完整一条生成/replay/导出与读取；不声称另一台干净机器验证。
