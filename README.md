Action-Agent-CV-Analysis: 基于时序动作识别与生物力学的实时评估 Agent
1. 项目简介 (Introduction)
本项目构建了一个集成计算机视觉、深度学习与逻辑推理的长链驱动 Agent。系统利用 MediaPipe 实时捕获人体关键点数据，通过预训练的 LSTM 网络进行时序动作分类，并结合生物力学逻辑（如关节角度评估）为用户提供实时的专业指导建议。该项目展示了如何将底层的感知数据转化为高层的智能决策反馈。
2. 核心技术架构 (Architecture)
   感知层 (Perception Layer)：利用 MediaPipe Pose 提取 33 个人体关键点，构建包含 $(x, y, z, visibility)$ 的 $132$ 维特征向量。模型层 (Model Layer)：采用双层 LSTM 神经网络，通过长度为 $30$ 帧的滑动窗口处理时序数据，捕捉动态特征。逻辑层 (Agent Logic Layer)：系统根据模型输出的动作标签（如 Squat），触发特定的生物力学评估 Agent，实时计算关节夹角并生成纠错建议。
3. 环境配置 (Setup)
# 克隆仓库
git clone https://github.com/aike-zyl/Action-Agent-CV-Analysis.git

# 安装依赖
pip install mediapipe opencv-python numpy torch
4. 核心逻辑实现 (Core Logic Flow)
该 Agent 的决策链条如下：动作识别：LSTM 将当前序列分类为特定动作（如：深蹲）。力学分析：Agent 提取对应的关键点坐标（髋、膝、踝），计算实时运动角度。推理建议：对比标准生物力学阈值（如：膝关节角度是否小于 $95^{\circ}$），生成自然语言反馈建议。
5. 研究成果与应用 (Achievements)
实时性能：在高性能计算环境下实现了低延迟的实时闭环评估。

跨领域迁移：相关的深度学习优化逻辑已成功应用于 RDD2022 道路损伤检测项目，在 YOLOv8 架构下取得了 mAP50 = 0.516 的检测精度。

硬件适配：针对 Winspace SLC3 等专业器材的使用场景进行了生物力学建模适配。
6. 开发栈 (Tech Stack)
语言: Python

框架: PyTorch, MediaPipe

算法: LSTM, YOLOv8
