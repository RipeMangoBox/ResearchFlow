---
title: "WorldModelBench: Judging Video Generation Models As World Models"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/video-generation
  - physics-aware-evaluation
  - vlm-as-a-judge
  - reward-gradients
  - dataset/WorldModelBench
  - dataset/WorldModelBench-Hard
  - opensource/partial
core_operator: "用跨领域图像/文本条件、细粒度物理与指令遵循评分，以及67K人类标注微调的2B裁判器，评估视频生成模型能否充当世界模型"
primary_logic: |
  评测“视频生成是否具备世界模型能力” → 构造覆盖7个应用域的350个图像/文本条件并收集67K细粒度人类标注 → 按指令遵循、常识质量、5类物理规律进行10分制评分并训练自动judger → 揭示当前模型在物理一致性、动作完成度与I2V能力上的真实边界
claims:
  - "WorldModelBench与VBench在帧级画质胜率上相关系数为0.69，但与物理一致性相关的胜率相关仅为0.28，说明通用视频质量基准不足以衡量世界模型式的物理可信度 [evidence: analysis]"
  - "基于67K人类标签微调的2B judger在350个基准样本上的总分预测平均误差为4.1%，能够较稳定逼近人类打分 [evidence: comparison]"
  - "在14个被测模型中，I2V版本系统性落后于对应T2V版本，例如CogVideoX为7.31 vs 6.75、OpenSora-Plan为7.61 vs 6.62 [evidence: comparison]"
related_work_position:
  extends: "VideoPhy (Bansal et al. 2024)"
  competes_with: "VBench (Huang et al. 2024); VideoPhy (Bansal et al. 2024)"
  complementary_to: "VADER (Prabhudesai et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Evaluating_World_Models/arXiv_2025/2025_WorldModelBench_Judging_Video_Generation_Models_As_World_Models.pdf
category: Survey_Benchmark
---

# WorldModelBench: Judging Video Generation Models As World Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.20694), [Project](https://worldmodelbench-team.github.io/)
> - **Summary**: 这篇工作把“视频好不好看”和“视频能不能作为世界模型”分开评测，构建了一个专门检查指令完成度、常识质量与物理可信度的跨域基准，并用大规模人类标注训练自动裁判器。
> - **Key Performance**: 2B judger 对人类总分的平均预测误差为 4.1%；WorldModelBench 与 VBench 在物理相关胜率上的相关系数仅 0.28。

> [!info] **Agent Summary**
> - **task_path**: 文本/首帧条件 + 生成视频 -> 世界模型能力评分（指令遵循/常识/物理）
> - **bottleneck**: 现有视频生成基准主要奖励画质与时序平滑，无法稳定暴露“看起来逼真但违反物理或没完成动作”的世界建模错误
> - **mechanism_delta**: 把评测从单一视频质量分改成跨域条件集 + 8项细粒度问题 + 人类对齐的2B judger
> - **evidence_signal**: 67K人类标注上的14模型比较，以及与VBench的低物理相关性分析
> - **reusable_ops**: [细粒度物理违规分类, 人类标注裁判器微调]
> - **failure_modes**: [高画质视频仍可能漂浮或穿模, I2V模型普遍弱于对应T2V模型]
> - **open_questions**: [judger能否泛化到完全未见prompt与更长视频, 用judger做奖励是否会过拟合基准]

## Part I：问题与挑战

这篇论文要解决的**真瓶颈不是生成器不会“画视频”**，而是社区还没有一个可靠方法去判断这些视频模型是否真的学到了“世界如何演化”。

### 1. 现有评测漏掉了什么
已有基准大多看：
- 帧级画质是否清晰；
- 时间一致性是否平滑；
- 文本是否大致匹配视频。

但对“世界模型”而言，这还不够。因为一个视频即使：
- 很清晰，
- 很流畅，
- 也大致符合文本，

仍可能出现：
- 物体悬空，违反重力；
- 物体互相穿透；
- 物体尺寸异常变化，违背质量守恒/固体力学；
- 动作只做了一半，没真正完成指令。

### 2. 为什么现在必须解决
因为视频生成模型已经被拿来讨论为机器人、自动驾驶、交互仿真中的世界模型。如果评测仍然只盯着“好看”，那训练和选型就会持续朝错误目标优化。

### 3. 输入/输出接口与边界
- **输入**：文本，或文本 + 首帧图像。
- **输出**：生成的短视频。
- **评测输出**：总分 10 分，包含  
  - 指令遵循 0–3 分  
  - 常识质量 0–2 分  
  - 物理遵循 0–5 分
- **覆盖范围**：7 个应用域、56 个子域、350 个条件，支持 T2V 和 I2V。
- **边界条件**：它测的是“未来几帧是否合理”，不是长时程规划、交互式闭环控制，也不是严格物理仿真。

## Part II：方法与洞察

### 方法骨架

#### 1. 用“应用域条件”替代泛化美学prompt
作者没有继续堆更大、更泛的开放域prompt，而是围绕世界模型真正关心的场景构造条件：
- robotics
- driving
- industry
- gaming
- human activities
- animation
- natural scenes

每个样本由**首帧图像 + 动作文本**组成，先从现有视频数据中取参考视频，再用 GPT-4/4o 做描述，最后人工校验，尽量保证“条件本身可实现”。

#### 2. 把世界模型能力拆成 3 个可诊断维度
**(a) Instruction Following**
- 0：主体缺失或静止
- 1：动了，但动作方向/语义错
- 2：部分完成
- 3：完整正确完成

**(b) Physics Adherence**
检查 5 类常见物理违例：
- Newton’s First Law
- Conservation of Mass / Solid Mechanics
- Fluid Mechanics
- Impenetrability
- Gravitation

**(c) Commonsense**
- frame-wise quality
- temporal quality

这一步的关键不是“多了几个指标”，而是把过去混在一起的错误来源拆开了：**动作没做对**、**物理不可信**、**只是画质差**，现在能分开看。

#### 3. 用大规模人类 dense label 校准评测
他们收集了：
- 8336 个完整投票
- 67K 标签
- 65 名标注者
- 平均每视频 1.70 票

并报告：
- 分数差在 ±2 内的一致率为 87.1%
- pairwise agreement 为 70%

这说明基准不只是“设计了一套分数”，而是确实用人类判断把评分口径定住了。

#### 4. 再把人类评测蒸馏成自动 judger
作者把单次投票转成 8 个问答任务，微调一个 2B VLM 作为自动裁判器。随后又进一步把这个 judger 接到 reward gradients 框架里，展示它不仅能“评”，还可能“教”。

### 核心直觉

**从因果上看，这篇论文真正改变的是“测量瓶颈”。**

- **改变前**：评测信号主要来自画质、流畅度、文本粗匹配。  
- **改变后**：评测信号变成“给定具体初始状态后，动作是否完成、未来状态是否物理可行”。  
- **结果**：模型是否具备世界建模能力，不再被“视频看起来挺真”所掩盖。

更具体地说：

**单一质量分 → 细粒度约束问题集 → 世界模型错误更可见**

因为世界模型错误通常不是“整体都错”，而是局部违反约束：
- 手差一点没碰到目标；
- 车转了相反方向；
- 物体突然变形；
- 液体表现得像胶体；
- 物体漂浮。

把评测写成 8 个明确问题后，这些错误不再被平均掉。

**开放域审美prompt → 应用驱动交互prompt → 更容易触发真正难点**

机器人开门、驾驶让行、人体接触、投掷/跳跃，这些任务天然会暴露：
- 主体-环境交互建模不足；
- 人体/机械臂动作结构不稳；
- 接触与受力关系不对。

### 战略权衡

| 设计选择 | 得到的诊断能力 | 代价/风险 |
|---|---|---|
| 350 个“小而难”的应用域条件 | 更高密度地触发交互与物理错误 | 覆盖面仍小于大型通用基准 |
| 物理五分类 + 指令四级评分 | 可解释、可定位具体失败模式 | 对复杂长链物理/因果仍偏粗粒度 |
| 67K 人类标注 | 让评测口径真正对齐人类判断 | 标注成本高，众包噪声需控制 |
| 2B 自动 judger | 评测可扩展、还能提供训练奖励 | judge 本身可能带来分布偏置或被“刷分” |

## Part III：证据与局限

### 关键证据

1. **分析信号：它确实测到了 VBench 没测到的东西**  
   作者比较模型两两胜率后发现：
   - WorldModelBench 与 VBench 在**帧级画质**上相关系数为 **0.69**
   - 但在**物理相关能力**上只剩 **0.28**

   这说明它不是完全背离已有画质评测，而是在“物理可信度”上补上了传统基准的盲区。

2. **比较信号：高画质 ≠ 强世界模型**  
   论文明确举出：Luma 的 commonsense 画质优于 open Mochi，但 instruction following 更差、physics 得分相近。  
   这正是该 benchmark 的价值：把“好看”和“可信”拆开。

3. **比较信号：I2V 仍是明显短板**  
   三组成对模型都出现 I2V < T2V：
   - CogVideoX: 7.31 vs 6.75
   - OpenSora-Plan: 7.61 vs 6.62
   - OpenSora: 6.11 vs 5.83

   这给出了一个非常直接的社区结论：**很多 I2V 模型还没被调到能承担世界模型角色。**

4. **自动化信号：judger 已经足够可用**  
   2B judger 对人类总分的平均预测误差为 **4.1%**。  
   同时，论文还展示它能作为 reward model，定性改善 OpenSora 的闪烁、指令不遵循与部分物理违例。

### 1-2 个最关键指标
- **Judger 平均总分预测误差**：4.1%
- **WMB vs VBench 的物理相关相关系数**：0.28

### 局限性

- **Fails when**: 需要评估长时程视频、复杂多阶段因果链、交互式闭环 rollout、或超出 7 个应用域的新分布时；另外，judger 的主要验证是“同一 prompt 上看未见模型输出”，不是“完全未见 prompt 泛化”。
- **Assumes**: 参考视频 + GPT-4/4o 生成的条件描述足够准确；67K 众包标签能代表世界模型质量；评测者可以承担高昂生成成本与部分闭源 API 依赖。论文里还使用了 GPT-4/GPT-4o/Gemini 参与数据构建或辅助推理，这会影响完全复现门槛。
- **Not designed for**: 真实控制成功率评估、策略规划质量、安全认证、3D 一致性验证、严格物理模拟器替代。

### 可复用组件
- **WorldModelBench**：350 条跨域条件
- **WorldModelBench-Hard**：45 条轻量但高区分度子集
- **8维细粒度评分协议**：特别适合检查动作完成度与物理违例
- **2B judger**：可作为未来视频模型的自动评测器
- **judge-as-reward 接口**：可接到 reward gradients / post-training 管线中

## Local PDF reference

![[paperPDFs/Evaluating_World_Models/arXiv_2025/2025_WorldModelBench_Judging_Video_Generation_Models_As_World_Models.pdf]]