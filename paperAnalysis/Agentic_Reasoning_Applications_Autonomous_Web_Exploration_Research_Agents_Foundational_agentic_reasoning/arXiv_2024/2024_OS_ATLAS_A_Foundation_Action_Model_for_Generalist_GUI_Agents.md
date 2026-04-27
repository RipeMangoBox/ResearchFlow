---
title: "OS-ATLAS: A Foundation Action Model for Generalist GUI Agents"
venue: arXiv
year: 2024
tags:
  - Embodied_AI
  - task/gui-grounding
  - task/gui-agent
  - data-synthesis
  - unified-action-space
  - multitask-imitation-learning
  - dataset/ScreenSpot-V2
  - dataset/OSWorld
  - opensource/partial
core_operator: 用跨平台 GUI grounding 预训练和统一动作空间微调，把截图-指令直接映射为跨平台可执行 GUI 动作
primary_logic: |
  GUI 截图/任务指令/历史动作 → 用 1358 万元素级跨平台 grounding 数据学习界面定位先验 → 在统一动作空间上做多任务动作微调并保留 custom action 扩展 → 输出跨 web/mobile/desktop 的坐标化 GUI 动作
claims:
  - "OS-Atlas-Base-7B 在 ScreenSpot 标准设定下平均 grounding accuracy 达 82.47%，高于 UGround-7B 的 73.30% [evidence: comparison]"
  - "OS-Atlas-7B 在零样本 OOD agent 设定下，于 GUI-Act-Web、OmniAct-Web、OmniAct-Desktop 的 step success rate 分别为 57.02%、59.15%、56.73%，均超过 GPT-4o [evidence: comparison]"
  - "去掉 grounding pre-training 或 unified action space 会在 web、desktop、mobile 的零样本 OOD step success 与 grounding 上同时下降，说明跨平台 GUI 预训练与动作语义统一是关键因果因素 [evidence: ablation]"
related_work_position:
  extends: "SeeClick (Cheng et al. 2024)"
  competes_with: "UGround (Gou et al. 2024); SeeClick (Cheng et al. 2024)"
  complementary_to: "GPT-4o; Set-of-Mark prompting (Yang et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2024/2024_OS_ATLAS_A_Foundation_Action_Model_for_Generalist_GUI_Agents.pdf
category: Embodied_AI
---

# OS-ATLAS: A Foundation Action Model for Generalist GUI Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2410.23218), [Project](https://osatlas.github.io/)
> - **Summary**: 这篇工作把 GUI agent 的核心难点拆成“先看准界面元素、再用统一动作语义去执行”，通过跨平台大规模 grounding 预训练 + 统一动作空间，把开源 VLM 推到可在陌生 GUI 上执行动作的 foundation action model。
> - **Key Performance**: ScreenSpot 平均 grounding accuracy 82.47%（OS-Atlas-Base-7B，无 planner）；OSWorld 成功率 14.63%（GPT-4o + OS-Atlas-Base-7B），高于 +SeeClick 的 9.21%。

> [!info] **Agent Summary**
> - **task_path**: GUI screenshot + task instruction + optional action history -> executable GUI action / coordinates across web, mobile, desktop
> - **bottleneck**: 开源 VLM 缺少跨平台 GUI 视觉先验，且多数据集动作命名冲突导致 OOD grounding 与动作泛化差
> - **mechanism_delta**: 先用 1358 万元素级跨平台 GUI grounding 数据做预训练，再用统一动作空间做多任务动作微调
> - **evidence_signal**: 六个 benchmark 的跨平台比较 + 去掉预训练/统一动作空间的 ablation
> - **reusable_ops**: [cross-platform GUI data synthesis, unified action space]
> - **failure_modes**: [web-only pretraining 难迁移到 desktop/mobile, 长程开放任务上仍明显落后人类]
> - **open_questions**: [如何低成本扩展高质量 desktop/mobile 交互数据, 如何让 custom actions 扩展后仍保持动作语义稳定]

## Part I：问题与挑战

**What / Why：真实瓶颈不是缺一个会输出 CLICK/TYPE 的模型，而是缺一个对 GUI“看得准、且动作语义一致”的开源动作底座。**

GUI agent 现在有两条路：  
一条是读 HTML / accessibility tree，另一条是直接看截图。前者在真实软件里常常拿不到，或者信息过长、含噪、结构不稳定；后者更贴近真实使用场景，但会把问题压缩到一个更硬的核心：**模型必须把自然语言意图稳定地落到屏幕上的可执行元素**。

这篇论文指出，开源 GUI agent 做不好的根因主要有两个：

1. **GUI 视觉先验缺失**  
   通用 VLM 很少在 GUI 截图上大规模预训练，所以它知道“图像是什么”，却不一定知道“界面里什么能点、什么能输、什么是滚动区域”。这会直接伤害 GUI grounding，尤其在陌生 app、陌生平台、陌生布局下更明显。

2. **动作监督不一致**  
   多个数据集混训时，同一逻辑动作常被写成不同名字，例如 `tap` vs `click`、`input` vs `type`。这不是小问题，而是**监督层面的冲突**：模型会把本来可共享的知识学散，最终损伤 OOD 泛化。

为什么现在值得做？因为 GUI agent 已经开始从“基于结构化文本的半封闭环境”转向“直接读屏的开放环境”，但目前最强系统多依赖 GPT-4o / Gemini 这类闭源 VLM。研究社区缺的不是又一个 demo，而是一个**可复用、可扩展、开源可训练的 GUI action foundation**。

**输入 / 输出接口**在论文里被拆成三种模式：

- **Grounding mode**：截图 + 元素描述/子任务指令 → 坐标/框
- **Action mode**：截图 + 任务指令 + 历史动作 → 单步可执行动作
- **Agent mode**：在具体 benchmark 上继续 SFT 后 → 面向特定场景的动作模型

**边界条件**也很明确：这篇论文的核心是**单步 GUI grounding / action prediction**，不是完整的长程 planner。它可以接 planner，但自己不解决高层任务分解。

## Part II：方法与洞察

**How：作者同时改了两件真正影响泛化的东西——训练分布，和动作监督接口。**

### 方法主线

OS-ATLAS 的训练分成两阶段：

1. **GUI Grounding Pre-training**
   - 目标：先让模型学会“看 GUI”
   - 数据格式：`<screenshot, referring expression or instruction, element coordinate>`
   - 规模：约 **224 万张截图、1358 万个 GUI 元素**
   - 结果：得到 `OS-Atlas-Base`

2. **Action Fine-tuning**
   - 目标：把“会找元素”进一步变成“会执行动作”
   - 输入：截图 + 任务指令 + 历史动作
   - 输出：动作类型 + 参数（如坐标、文本、方向）

### 数据基础设施：先把 GUI 视觉分布补上

论文最硬核的贡献其实是数据层。

- **Web**：从 FineWeb URL 渲染完整网页，不只截上半屏；再抽取可点击元素、坐标和表达，并做规则过滤。
- **Desktop / Mobile**：  
  - Android 用 AndroidEnv  
  - Linux 用 OSWorld  
  - Windows / macOS 因虚拟化困难，直接在物理机上采集  
  - 通过 A11y tree 获取元素信息，并用 DFS / Random Walk 自动探索界面状态
- **Instruction Grounding 补充**：对现有轨迹数据，使用 **GPT-4o + Set-of-Mark** 生成更细粒度的子指令

这里的关键取舍很重要：作者没有把希望押在昂贵的人类 instruction 数据上，而是优先构建一个**可以持续扩张的跨平台 grounding 数据生产线**。后面的 ablation 也支持这一点：**可规模化的 REG / grounding 数据，已经足以撑起很强的 GUI grounding 底座。**

### 统一动作空间：再把监督语义统一

第二个核心设计是 **Unified Action Space**。

- **Basic Actions**：跨平台共享的基础动作  
  - `CLICK`
  - `TYPE`
  - `SCROLL`
- **Custom Actions**：平台或任务特有动作  
  - 如 `open app`、`drag` 等

它解决的不是“动作种类不够多”，而是“动作语义太碎”。作者实证表明，统一动作空间把独特动作类型从 **17 降到 10**，同时消解了 `tap/click`、`press home/home`、`type/input` 这类命名冲突。

### 核心直觉

OS-ATLAS 的关键不是“换个更大 backbone”，而是**把最伤 OOD 的两个瓶颈前移并修掉**：

- **从通用视觉分布 → GUI 专用视觉分布**  
  模型先在海量 GUI 截图上学到按钮、输入框、图标、滚动区的空间先验，因此面对陌生界面时，更容易先“落点正确”。

- **从冲突监督 → 共享动作语义**  
  统一动作空间让 web / mobile / desktop 之间语义相同的交互共享参数，而不是被不同标签切碎。

- **从封闭动作集合 → basic + custom 的扩展式接口**  
  基础动作保证迁移，custom actions 保证 OOD 新操作还能接得住。

可以把它的因果链压缩成一句话：

> 跨平台 GUI 视觉先验 + 统一的动作语义监督  
> → 降低定位误差与标签冲突  
> → 提升跨平台、跨应用、跨任务的零样本执行能力

### 策略权衡

| 设计选择 | 解决的瓶颈 | 收益 | 代价 / 风险 |
|---|---|---|---|
| 大规模 synthetic REG 为主、IG 为辅 | GUI 数据稀缺 | 可快速扩到多平台，先把 grounding 做强 | 指令语义细腻度仍部分依赖 GPT-4o 标注 |
| 统一 basic actions | 动作命名冲突 | 跨平台共享知识，zero-shot 更稳 | 可能压平少数平台特有细节 |
| basic + custom 拆分 | 固定动作表难覆盖新任务 | 保留 OOD 可扩展性 | custom action 若定义不清，会重新引入歧义 |
| grounding plug-in + agent mode 并存 | 端到端 agent 难调试、难复用 | 可做独立 grounding 模块，也能做完整 action model | planner 失误本身不由 grounding 模块修复 |

## Part III：证据与局限

**So what：能力跃迁主要体现在——开源模型终于不只是在熟悉界面里“会点”，而是能跨平台、跨应用地做 OOD GUI grounding 与 step-level action。**

### 关键证据信号

- **信号 1：Grounding 比较强且跨平台一致**  
  在 ScreenSpot 标准设定下，**OS-Atlas-Base-7B 平均 grounding accuracy = 82.47%**，高于 **UGround-7B 的 73.30%**。  
  配合 GPT-4o planner 的 grounding mode 下，进一步到 **85.14%**。  
  这说明它不是只在单一平台上变强，而是 mobile / desktop / web 都在涨。

- **信号 2：作为 grounding plug-in 真能提升 agent**  
  在 OSWorld 中，把 GPT-4o agent 的坐标模块替换成 OS-Atlas-Base-7B 后，成功率到 **14.63%**，高于 **+SeeClick 的 9.21%**。  
  虽然离 human **72.36%** 还很远，但至少证明：**更强 grounding 会转化成更强 agent 执行。**

- **信号 3：零样本 OOD 行为泛化明显提升**  
  OS-Atlas-7B 在零样本设定下：
  - GUI-Act-Web SR：**57.02%** vs GPT-4o **41.84%**
  - OmniAct-Web SR：**59.15%** vs GPT-4o **34.06%**
  - OmniAct-Desktop SR：**56.73%** vs GPT-4o **50.67%**
  - GUI-Odyssey SR：**26.96%** vs GPT-4o **5.36%**

  这基本回答了论文最重要的问题：**开源模型能不能在陌生 GUI 上替代闭源商业模型？**  
  结论不是“完全替代”，但至少已经到“有竞争力的开放底座”。

- **信号 4：ablation 直接支持因果解释**  
  去掉 grounding pre-training，或者去掉 unified action space，web / desktop / mobile 的 OOD step success 与 grounding 都下降。  
  另外，只用 web 数据预训练时，对 desktop / mobile 的迁移明显变差。  
  这说明性能提升不是偶然来自 backbone，而是来自：
  1. **跨平台 GUI 数据预训练**
  2. **动作语义统一**

- **信号 5：数据 scaling 仍在起作用**  
  随着 grounding 数据规模增大，IoU 和 accuracy 持续上升，尤其 web 域更明显。  
  这意味着这条路线还没到头，未来继续堆高质量 GUI 数据很可能还能涨。

### 局限性

- **Fails when**: 需要长程规划、错误恢复、跨应用 workflow 的真实开放任务时，系统仍明显落后人类；例如 OSWorld 最好结果仅 14.63%，远低于 human 的 72.36%。此外，只用 web 预训练时，对 desktop/mobile 的迁移明显不足。
- **Assumes**: 依赖大规模跨平台 synthetic grounding 数据、DOM/A11y tree 可访问性，以及 Windows/macOS 物理机采集；instruction grounding 标注与 grounding-mode planner 还依赖 GPT-4o，这对完全开源复现有实际影响；部分分析只在 4B 上进行，受 GPU 预算限制。
- **Not designed for**: 端到端高层任务规划、长期记忆/反思、无需环境适配的任意新动作执行；移动端主要覆盖 Android，论文没有展示 iOS 级别的系统化评测。

还需要注意两个评测边界：

1. **OmniAct 的 zero-shot OOD 只评第一步动作**，因为原 benchmark 不提供动态环境。
2. 作者发现 **ScreenSpot 原标注约有 11.32% 错误**，并发布 ScreenSpot-V2；这意味着此前很多 grounding 结论本身带有 benchmark 噪声。

### 可复用组件

- 跨平台 GUI grounding 数据合成 toolkit
- 约 224 万截图 / 1358 万元素的 grounding 语料
- ScreenSpot-V2 清洗版评测集
- unified action space 设计（CLICK / TYPE / SCROLL + custom actions）
- 可外接 planner 的 grounding plug-in 接口

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2024/2024_OS_ATLAS_A_Foundation_Action_Model_for_Generalist_GUI_Agents.pdf]]