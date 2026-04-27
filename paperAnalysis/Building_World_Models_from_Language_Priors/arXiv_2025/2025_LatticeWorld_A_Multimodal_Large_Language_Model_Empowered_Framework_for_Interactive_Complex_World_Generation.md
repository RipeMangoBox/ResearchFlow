---
title: "LatticeWorld: A Multimodal Large Language Model-Empowered Framework for Interactive Complex World Generation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/interactive-world-generation
  - task/scene-layout-generation
  - state-token
  - multimodal-fusion
  - procedural-generation
  - dataset/LoveDA
  - dataset/Wild
  - opensource/no
core_operator: 先把文本与高程先验翻译成可解释的符号布局矩阵和层级环境配置，再交给 Unreal Engine 渲染成可交互多智能体世界
primary_logic: |
  文本布局描述 + 可选高程图/草图 + 环境配置指令
  → MLLM 生成 32×32 符号布局矩阵与粗到细 JSON 配置
  → 译码为 UE 可读的布局遮罩、资产参数与 agent 参数
  → 实时渲染出具有物理模拟和多智能体交互的 3D 世界
claims:
  - "在固定高度与可变高度两种布局生成设置下，LatticeWorld 相比 GPT-4o、Claude 3.7 Sonnet、DeepSeek-R1/Qwen2-VL-Max 生成的布局更符合文本与地形约束 [evidence: comparison]"
  - "在与人工工业流程的对比中，LatticeWorld 将动态环境生产时间从 55 天降至不足 0.6 天，效率提升超过 90× [evidence: comparison]"
  - "系统能够在 UE 中生成带巡逻、追逐和攻击等规则行为的多智能体动态环境，而不只是静态 3D 场景 [evidence: case-study]"
related_work_position:
  extends: "SceneCraft (Hu et al. 2024)"
  competes_with: "SceneX (Zhou et al. 2024); 3D-GPT (Sun et al. 2023)"
  complementary_to: "Habitat 3.0 (Puig et al. 2023); Isaac Sim (NVIDIA 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2025/2025_LatticeWorld_A_Multimodal_Large_Language_Model_Empowered_Framework_for_Interactive_Complex_World_Generation.pdf
category: Embodied_AI
---

# LatticeWorld: A Multimodal Large Language Model-Empowered Framework for Interactive Complex World Generation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2509.05263), [Demo](https://youtu.be/8VWZXpERR18)
> - **Summary**: 该工作把多模态指令先压缩成 32×32 符号化场景蓝图与层级环境配置，再借助 UE5 落地为具有物理模拟和多智能体交互的可玩 3D 世界。
> - **Key Performance**: 工业制作时间由 55 天降至 <0.6 天（>90×）；在 LoveDA/Wild 的布局生成比较中优于 GPT-4o、Claude 3.7 Sonnet、DeepSeek-R1/Qwen2-VL-Max 等通用模型。

> [!info] **Agent Summary**
> - **task_path**: 文本布局描述/高程图或草图/环境配置指令 -> UE 可渲染的多智能体交互 3D 世界
> - **bottleneck**: 开放式用户意图难以直接映射为同时满足空间一致性、地形约束、海量引擎参数与 agent 配置的可执行世界状态
> - **mechanism_delta**: 用“符号布局矩阵 + 粗到细配置 JSON”替代直接生成像素或引擎脚本，再把高保真物理与实时渲染外包给 UE
> - **evidence_signal**: 跨固定高度与可变高度布局比较优于通用 MLLM，且人工产线时间从 55 天缩短到 <0.6 天
> - **reusable_ops**: [symbolic-layout-matrix, coarse-to-fine-config-json]
> - **failure_modes**: [32x32网格难表达细粒度室内或高密城市结构, agent行为仍以规则驱动为主]
> - **open_questions**: [如何把规则化agent升级为可学习策略, 如何在保持可控性的同时扩展到更细粒度开放世界]

## Part I：问题与挑战

**What/Why**：这篇论文要解决的，不是“生成一张像 3D 世界的图”，而是**把用户的多模态意图变成一个引擎可执行、可实时交互、带物理与 agent 的世界**。

### 真正问题是什么
现有路线大致有两类，但都卡在“可执行世界”这一步：

1. **神经渲染/扩散式 3D 生成**
   - 强在视觉外观。
   - 弱在交互、物理、实时性。
   - 更像“生成内容”，不像“生成可玩的世界”。

2. **平台式内容生成（如 Blender 代码生成）**
   - 强在建模工作流衔接。
   - 但通常偏离线内容制作，不擅长实时多 agent、复杂物理和工业级交互。

LatticeWorld 认为真正瓶颈有三层：

- **空间规划瓶颈**：文本说“左边是森林、右边是雪山、中间有湖”，模型要稳定生成符合相对位置的场景布局并保持全局一致。
- **约束融合瓶颈**：一旦加入高程图/草图，布局还得 obey 地形常识，比如雪应更接近高处、水不该跑到山脊上。
- **引擎执行瓶颈**：最终输出不能只是图像，还要变成 UE 能读的布局、材质、天气、植被密度、建筑摆放和 agent 参数。

### 为什么现在值得做
- **LLM/MLLM 已经足够擅长符号理解与结构化序列生成**，不必从头学一个完整 3D 生成器。
- **UE5 这类工业引擎已经提供实时渲染、物理、插件生态和多 agent 场景承载能力**。
- **Embodied AI / world model / simulation 数据需求快速上升**，手工做场景太慢太贵。

### 输入/输出接口与边界
- **输入**：
  - 布局文本 `xL`
  - 可选视觉条件 `vL`（高程图或草图）
  - 环境配置文本 `xC`（季节、天气、风格、agent 类型/数量/状态等）
- **输出**：
  - 一个可在 UE 中运行的 3D 动态世界 `W`
- **边界条件**：
  - 主要面向**大尺度户外地形世界**
  - 依赖已有资产类别与引擎插件
  - 当前主角控制仍以输入设备为主，agent 行为也主要是规则驱动

---

## Part II：方法与洞察

**How**：核心做法是把“直接生成完整世界”拆成两步：  
先让 MLLM 只负责它更擅长的**离散规划**，再让 UE 负责它更擅长的**图形执行与物理交互**。

### 方法总览
LatticeWorld 包含三段式管线：

1. **场景布局生成器 `LLM_L`**
   - 输入：文本布局描述 + 可选高程图/草图
   - 输出：`32×32` 的符号矩阵（如 A/B/F/W 等字母表示资产类别）
   - 作用：先产出一个可解释、可检验的“世界蓝图”

2. **环境配置生成器 `LLM_C`**
   - 输入：环境配置文本 + 高程图 + 已生成的布局矩阵
   - 输出：JSON 形式的环境配置
   - 内容：包括场景属性与 agent 参数  
     - 场景属性：季节、天气、风格、时间、材质、密度等
     - agent 参数：类别、数量、状态、初始位置等

3. **程序化渲染管线**
   - `ΨL`：把符号矩阵变成引擎可读的布局 mask
   - `ΨC`：把 JSON 配置翻译成 UE 原生参数
   - `Render`：结合高程图、布局和配置，在 UE 中生成动态世界

### 核心直觉

**改变了什么**：  
作者没有让模型直接生成最终 3D 场景、UE 脚本或神经渲染结果，而是引入一个**低熵、强约束、可解释的中间表示**：  
- 布局层：32×32 符号矩阵  
- 配置层：粗到细的 JSON 环境配置

**哪个瓶颈被改变了**：  
原来的输出空间是“连续几何 + 海量引擎参数 + 交互逻辑”的混合高维空间，LLM 很难直接稳定建模。  
引入中间表示后，问题被改写为：
- **空间关系建模** → 离散 token 序列生成
- **配置控制** → 分层参数生成
- **物理/视觉真实感** → 交给 UE

**能力因此怎么变了**：  
即便只用 **LLaMA-2-7B** 这类轻量模型，也能：
- 更稳定地遵守空间关系
- 更好地利用高程先验
- 输出可落地到工业引擎的世界状态
- 从“静态内容生成”跃迁到“可交互世界生成”

### 为什么这个设计有效
1. **符号布局矩阵降低了学习难度**
   - LLM 天生更擅长序列和离散结构，而不是直接画精确几何。
   - 32×32 网格把空间关系转成了“可说、可写、可校验”的布局语言。

2. **高程图把 3D 先验显式注入了布局生成**
   - 不是事后修补，而是在布局生成时就引入地形约束。
   - 这样雪、水、岩石、草地等分布更符合地势。

3. **粗到细环境配置把参数爆炸分层化**
   - 顶层控制全局语义：季节、天气、风格、时间。
   - 底层控制细节：密度、材质、旋转、坡度、建筑摆放、agent 状态。
   - 这把“上千参数联动”变成一个更可控的生成过程。

4. **UE 接管了学习系统不擅长的部分**
   - 真实感渲染
   - 物理模拟
   - 多 agent 运行时
   - 插件生态（如天气、体积云、流体等）

### 关键模块拆解

#### 1. 符号化布局语言
作者把语义分割布局压成 `32×32` 字符矩阵，再序列化成文本，让 LLM 直接输出。

这一步的意义不是“压缩图像”，而是**把 2D/3D 空间规划重写成语言模型能稳定处理的状态 token**。  
相邻字符天然编码了邻接关系，字符位置天然编码了区域位置。

#### 2. 视觉条件融合
对于可变高度地形，作者用：
- **CLIP ViT-B/32** 提取高程图特征
- 一个轻量 CNN projector 把视觉特征映射到语言 embedding 空间
- 三阶段训练：
  1. 先让 CLIP 更懂高程图
  2. 再对齐视觉特征与语言空间
  3. 最后端到端调 `LLM_L + projector`

这一步的本质是：**让高程图不再只是后处理约束，而是布局生成的前置条件**。

#### 3. 环境配置生成
`LLM_C` 不是直接吐“引擎代码”，而是生成结构化 JSON。  
这使得系统可以把：
- 全球语义（春夏秋冬、晴雨昼夜、卡通/写实）
- 局部细节（植被密度、建筑朝向、材质）
- agent 设置（羊、马、机器人、鹰等）

统一到一个中间配置层，再交给翻译器 `ΨC` 接入 UE。

#### 4. 渲染落地
- 布局矩阵先转成各类资产的 mask
- 再做拉伸和边缘融合
- 配置 JSON 变成 UE 原生属性
- 最后结合高程图生成场景、刷资产、挂天气和 agent 行为

所以它不是一个“只会讲 world idea 的 LLM”，而是一个**能把语言计划变成引擎状态**的系统。

### 战略取舍

| 设计选择 | 得到的能力 | 代价/边界 |
|---|---|---|
| 32×32 符号布局矩阵 | 可解释、可控、便于训练和调试 | 细粒度几何与复杂局部结构被压缩 |
| 高程图/草图作为视觉条件 | 布局更符合地形与常识 | 需要额外视觉输入或 sketch-to-heightmap 模块 |
| 粗到细配置 JSON | 降低参数冲突，便于语言控制环境 | 仍受规则系统和资产库覆盖范围限制 |
| UE 负责渲染/物理/交互 | 获得工业级实时性与多 agent 能力 | 可复现性依赖 UE、插件和工程实现 |

---

## Part III：证据与局限

**So what**：这篇论文真正的能力跃迁，不是“画得更像”，而是**把世界生成从静态内容合成推进到可执行、可交互、可用于 agent 训练的工业工作流**。

### 关键实验信号

1. **布局生成对比信号（最核心）**
   - 论文分别测试了：
     - 固定高度：仅文本生成布局
     - 可变高度：文本 + 高程图/草图生成布局
   - 对比对象包括 GPT-4o、Claude 3.7 Sonnet、DeepSeek-R1、Qwen2-VL-Max。
   - 结论：LatticeWorld 在这两类设置中都更能遵守文本空间关系与地形约束。  
   - 这说明专门设计的**符号布局表示 + 高程条件对齐**，确实比直接拿通用大模型生成更稳。

2. **系统级可控性信号**
   - 在保持布局基本不变时，系统可通过环境配置文本切换：
     - 季节
     - 天气
     - 光照/时间
     - 艺术风格
     - agent 类型与数量
   - 这说明布局层和配置层的解耦是有效的：**同一地形/布局可以被重渲染为不同世界风格**。

3. **动态环境能力信号**
   - 论文展示了多种 agent（羊、马、机器人、鹰等）在场景中的部署与交互。
   - agent 具有基础感知与规则行为，如接近主角后追逐/攻击。
   - 这至少证明其输出已超越“静态 3D scene”，进入“可运行 world state”。

4. **工业效率信号（最有落地价值）**
   - 与人工工业流程相比，LatticeWorld 把总制作时间从 **55 天** 降到 **<0.6 天**。
   - 这对应 **90× 以上** 的效率提升。
   - 这是论文最强的工程价值证据。

### 证据强弱判断
我会把本文证据强度定为 **moderate**，原因是：
- 有多组对比和实际系统展示；
- 也有明确的工业效率数字；
- 但大量结果仍偏**定性展示**，公开文本中缺少系统性的 ablation、标准化量化指标和更严格的 cross-engine / cross-domain 验证。

### 局限性
- **Fails when**: 需要细粒度室内结构、复杂建筑拓扑、高密城市布局、精确局部摆放，或需要长时程学习式 agent 策略时，32×32 布局表示和规则行为会明显不够。
- **Assumes**: 依赖 UE5 及其插件（如 Niagara Fluids 等）、预制资产库、GPT-4o 标注与提示工程、私有 Wild 数据集、A100 训练资源；建筑摆放与部分常识一致性仍由手工规则保障。
- **Not designed for**: 从零生成全新 3D 资产/材质、学习式物理仿真、多人主控角色、细粒度 body-part 控制、仅靠视频观察端到端生成开放世界。

### 可复用组件
- **symbolic layout language**：把空间布局离散化为 LLM 友好的中间语言
- **vision-to-language projector**：把高程图这类几何先验接入 LLM
- **coarse-to-fine config schema**：把海量引擎参数组织成可控层次
- **engine translation layer (`ΨL`, `ΨC`)**：把中间表示迁移到其他引擎（如 Unity）也有潜力

### 一句话判断
如果你关心的是**“如何让语言模型真正生成一个能跑起来的世界，而不是只生成世界的图像/代码片段”**，这篇 paper 的核心价值很清楚：  
它证明了**符号化中间表示 + 引擎执行**是一条现实可落地的路线。

## Local PDF reference

![[paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2025/2025_LatticeWorld_A_Multimodal_Large_Language_Model_Empowered_Framework_for_Interactive_Complex_World_Generation.pdf]]