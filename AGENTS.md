# 维护约定

优先复用payload内原控制源码。源目录中的__RUNTIME__是路径模板，运行launch.py物化，勿修改原工作区实验。

不得无证据修改模型、PD、接触门限、双手安装、物理200Hz/控制50Hz/相机25Hz或decoder历史。输出放仓库外新目录，保留失败。prepared、首次成功、独立回放、数据可读、视觉验收分别报告。

源码新增或资源变动后重算resources.json，并使用新runtime验证。大模型、资产、原始数据不进Git，交付时用pack.py包含资源。API凭据不得入库。变更前读README与docs/SOURCE_CHANGES.json。

- 用户授权的media/rendering_comparisons三份MP4可入Git，其他模型/场景/原始轨迹仍不入Git。渲染设置和已知边界见docs/RENDERING.md，保持未完成时序验收标签。
