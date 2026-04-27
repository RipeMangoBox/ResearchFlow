---
title: "ViMo: A Generative Visual GUI World Model for App Agent"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/gui-navigation
  - task/app-agent-planning
  - diffusion
  - state-token
  - "dataset/Android Control"
  - dataset/AITW
  - dataset/AndroidWorld
  - opensource/promised
core_operator: "用符号化文本占位符把GUI未来预测拆成“图形扩散生成 + 文本语义补全”两步，从而生成可用于App代理规划的下一界面。"
primary_logic: |
  当前GUI截图 + 用户动作 → OCR/规则将动态文本替换为STR占位符并保留难生成的静态文本 → 扩散模型预测下一步STR图形布局 → LLM为各文本占位符生成对应文本并回填 → 得到下一GUI图像并用于候选动作比较
claims:
  - "ViMo在GUI质量评测中取得最高综合分：自动评测谐均分为0.76、用户研究谐均分为0.88，均优于HTML-vision、IP2P*和UI-diffuser [evidence: comparison]"
  - "将ViMo作为插件接入App代理后，T3A的step accuracy从43.13%提升到49.20%，M3A从46.01%提升到50.16% [evidence: comparison]"
  - "保留静态文本与使用自然语言动作指令都对性能有显著贡献；去掉任一设计都会降低T3A和M3A上的step accuracy [evidence: ablation]"
related_work_position:
  extends: "InstructPix2Pix (Brooks et al. 2023)"
  competes_with: "UI-Diffuser (Wei et al. 2024); HTML-vision"
  complementary_to: "M3A (Rawles et al. 2024); T3A (Rawles et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2025/2025_ViMo_A_Generative_Visual_GUI_World_Model_for_App_Agent.pdf
category: Embodied_AI
---

# ViMo: A Generative Visual GUI World Model for App Agent

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.13936), [Project](https://ai-agents-2030.github.io/ViMo/)
> - **Summary**: 论文提出首个面向 App Agent 的可视化 GUI 世界模型，把“下一屏预测”拆成图形生成与文本补全两条通路，从而让代理能先“看见”动作后果再做决策。
> - **Key Performance**: GUI质量综合分相对现有世界模型平均提升 29.14%；AndroidWorld 在线任务成功率从 33.19% 提升到 40.95%。

> [!info] **Agent Summary**
> - **task_path**: 当前GUI截图 + 候选动作/任务目标 -> 预测下一GUI -> 选择更优下一步动作
> - **bottleneck**: GUI里的文本对像素误差极其敏感，纯像素生成会把字生成糊；而纯文本世界模型又丢失位置/颜色/布局等行动关键细节
> - **mechanism_delta**: 把动态文本从像素空间提升为符号占位符，令扩散模型只预测图形与文本位置，LLM再补全文本语义
> - **evidence_signal**: 同时在自动评测、70人用户研究、step accuracy 和 AndroidWorld 在线导航上取得一致提升
> - **reusable_ops**: [symbolic-text-representation, action-conditioned-next-screen-simulation]
> - **failure_modes**: [text-placeholder-shape-error, recursive-rollout-error-accumulation]
> - **open_questions**: [how-to-model-longer-horizon-without-drift, how-to-reduce-dependence-on-closed-llms]

## Part I：问题与挑战

这篇论文要解决的不是“能不能生成 GUI 图像”，而是**能不能生成对代理决策真正有用的下一屏 GUI**。

### 1. 真问题是什么
App agent 在短程操作上已经能工作，但一旦任务变成长链条，它就容易在局部选择里犯错。核心原因是：

- 代理只能看到当前屏幕；
- 它不知道某个候选动作执行后，下一屏会长什么样；
- 没有“想象动作后果”的能力，就难以做长程规划。

所以这里真正需要的是一个 **world model**：给定当前 GUI 和用户动作，预测下一步 GUI 状态。

### 2. 现有方法卡在哪里
作者指出两类已有方案都不够：

1. **语言式 world model**  
   能描述“下一步大概会发生什么”，但会丢掉 GUI 决策非常依赖的细节，比如：
   - 按钮具体位置
   - 元素颜色/样式
   - 文本框是否真的存在
   - 下一步可操作元素是否可见

2. **纯像素 GUI 生成**  
   图形部分常常还行，但 GUI 中的小字一旦像素有一点偏差，就会：
   - 文本不可读
   - 按钮语义错位
   - 界面“看起来像”，但实际上不能指导下一步动作

### 3. 输入/输出接口与边界
论文的基本接口很清楚：

- **输入**：当前 GUI 截图 + 一个自然语言动作指令
- **输出**：预测的下一张 GUI 图像

它主要面向的是：

- 手机 App GUI
- 一步状态转移预测
- 以截图为主的观察环境

它**不依赖 DOM/HTML 源码**，这点很重要，因为手机 App 往往拿不到类似网页的结构化源码。

---

## Part II：方法与洞察

ViMo 的关键不是“更强的生成模型”，而是**重新定义了 GUI 预测该怎么表示**。

### 方法总览

ViMo 分三步：

1. **STR：把 GUI 里的文本替换成符号占位框**
2. **STR Predictor：用扩散模型预测下一步 GUI 的图形与文本位置**
3. **GUI-text Predictor：用 LLM 给每个占位框补上对应文本**

最后再把文本回填到预测图像里，形成完整下一屏 GUI。

### 1. STR：Symbolic Text Representation
作者提出 STR 的目的，是把最难的部分——**文字像素级渲染**——从扩散模型手里拿走。

具体做法：

- 先用 OCR 检测 GUI 中文本；
- 把动态文本区域盖成统一的黑边白底矩形占位符；
- 对键盘、时钟面板等**静态文本**，尽量保留原像素，不强迫 LLM 去重建复杂固定布局。

这样一来，模型不再需要回答“这几个字每个像素该怎么画”，只需要回答：

- 这里未来有没有文本？
- 文本框会出现在什么位置？
- 大小大概如何？

这相当于把问题从**高精度文本渲染**，降成了**文本定位与语义补全**。

### 2. STR Predictor：扩散只负责“界面图形状态转移”
有了 STR 之后，下一步 GUI 预测就变成：

- 输入当前 STR
- 再输入用户动作
- 预测下一步 STR

作者基于 Stable Diffusion / InstructPix2Pix 风格的条件扩散来做这件事。  
它擅长的是：

- 页面布局变化
- 控件出现/消失
- 颜色/样式变化
- 文字占位框位置变化

也就是说，它专心学 **GUI graphics dynamics**，不再被文本像素拖累。

### 3. GUI-text Predictor：LLM 负责“填词”
扩散模型生成完下一步 STR 后，ViMo 再：

- 检测图中的文本占位框；
- 给每个框分配 ID；
- 结合当前 GUI、动作指令、预测后的 STR 上下文；
- 让 LLM 预测每个框应该填什么文字。

一个很实用的小设计是：  
ViMo 会先判断哪些文本在前后界面中其实没变，直接从当前 GUI 复制过去，只让 LLM 预测真正变化的文本，进一步减小错误面。

### 4. 如何服务 App agent
ViMo 不是直接替代 agent，而是作为一个**前瞻模块**插进去：

- agent 先提出多个候选动作；
- ViMo 分别模拟每个动作后的下一屏；
- 选择模型再根据这些“想象出来的后果”选最优动作。

所以它提升的不是单纯“生成质量”，而是**动作选择质量**。

### 核心直觉

**发生了什么变化？**  
从“直接生成完整 GUI 像素”改成“图形状态转移 + 文本语义补全”的分治式建模。

**改变了哪个瓶颈？**  
把 GUI 里最脆弱、最不适合扩散硬画的文本通道剥离出来，降低了视觉预测的熵和误差敏感性。

**带来了什么能力变化？**  
预测结果从“看着像下一屏”提升为“真的能支撑下一步动作判断的下一屏”。

更因果地说：

- GUI 图形变化是连续视觉问题，扩散模型擅长；
- GUI 文本内容是离散语义问题，LLM擅长；
- 把两者硬绑在一个像素生成器里，会让小文本成为系统瓶颈；
- 把它们拆开后，图像模型学布局，语言模型学内容，错误不会互相放大得那么严重。

### 战略权衡

| 设计选择 | 解决的瓶颈 | 得到的能力 | 代价/风险 |
|---|---|---|---|
| STR 占位符替换动态文本 | 文本像素生成极易失真 | 下一屏更可读、更可操作 | 依赖 OCR 与占位符检测质量 |
| 保留静态文本 | 键盘/时钟等固定布局文本难预测 | 减少无意义文本重建错误 | 需要额外规则判断“静态” |
| 用动作指令而非坐标命令做条件 | 低层命令语义弱 | 更好对齐用户意图 | 需要把日志动作转成自然语言 |
| 默认做一步 rollout | 降低误差累积 | 更实用、更稳 | 长程规划能力仍有限 |

---

## Part III：证据与局限

### 关键证据信号

1. **世界模型本身确实更好用，而不只是更好看**  
   在 GUI quality 评测中，ViMo 在自动指标和用户研究上都拿到最高综合分。  
   这说明它同时改进了：
   - 视觉一致性
   - 对动作指令的遵循
   - 对后续操作的可用性

2. **能力跳跃体现在下游决策，不止体现在生成分数**  
   把 ViMo 接到现有 agent 后：
   - T3A：43.13% -> 49.20%
   - M3A：46.01% -> 50.16%

   这比“图像更逼真”更关键，因为它证明 ViMo 生成的 GUI 对动作选择有实际价值。

3. **真实环境里也有收益**  
   在 AndroidWorld 在线导航中，任务成功率从 33.19% 提升到 40.95%。  
   这说明 ViMo 并不只是离线合成器，而是在真实交互场景里也能帮助代理。

4. **消融支持了机制归因**  
   论文不是只报总分，还验证了两个关键因果点：
   - 保留静态文本是有用的；
   - 用自然语言动作指令做条件优于低层 action command；
   - 多步 rollout 不是越长越好，超过一定步数会因误差累积而掉点。

### 1-2 个最值得记的指标
- **GUI 质量**：自动综合分 0.76，用户研究综合分 0.88
- **在线导航**：AndroidWorld 任务成功率 **33.19% -> 40.95%**

### 局限性

- **Fails when**: 预测出的文本占位符没有保持规则矩形外观时，后续检测/回填会失败；递归多步预测时误差会累计，长 horizon 下界面会逐步漂移；面对高度随机或依赖隐藏后端状态的 App 转移时，单凭截图可能不够。
- **Assumes**: 依赖 OCR 检测、强 LLM（论文默认大量使用 GPT-4o / Gemini）以及成对 GUI 转移数据；假设动作可被较好地表达成自然语言；推理侧至少需要 16GB GPU。
- **Not designed for**: 精确替代真实 emulator 的安全执行；对支付、发消息等高风险事务给出可验证保障；仅靠一次长 rollout 完成长程规划。

### 资源与可复现性提醒
- 最低部署需求：**16GB GPU**
- 单次 STR 生成约 **8 秒**
- GUI-text 预测约 **30 秒**
- 论文报告的完整一次请求（含 3 个候选动作、模型加载与通信开销）约 **2 分钟**
- 关键模块依赖闭源 LLM API，这会影响复现成本与稳定性
- 项目页已放出，但权重/完整开放属于 **promised** 状态

### 可复用组件
- **STR 表示**：适合任何“图形+文本混合”的界面预测任务
- **动作条件下一屏模拟器**：可作为 agent 的 imagined rollout 模块
- **静态文本保留策略**：适合固定模板 UI 的文本处理
- **候选动作先模拟再选择**：可插到现有 App agent / GUI agent 管线中

![[paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2025/2025_ViMo_A_Generative_Visual_GUI_World_Model_for_App_Agent.pdf]]