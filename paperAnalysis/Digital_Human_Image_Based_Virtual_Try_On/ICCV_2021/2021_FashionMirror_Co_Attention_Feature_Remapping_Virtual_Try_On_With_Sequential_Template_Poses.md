---
title: "FashionMirror: Co-Attention Feature-Remapping Virtual Try-On With Sequential Template Poses"
venue: ICCV
year: 2021
tags:
  - Video_Generation
  - task/video-generation
  - task/virtual-try-on
  - co-attention
  - optical-flow
  - feature-remapping
  - dataset/FashionVideo
  - opensource/partial
core_operator: "用共注意力直接预测脱衣区与试穿服装区，再以骨架驱动的特征流在特征空间重映射人物与服装，实现连续姿态下的虚拟试穿"
primary_logic: |
  源人物图像 + 目标服装图像 + 引导姿态序列 → 共注意力预测 removed mask / try-on clothing mask，并由相邻姿态估计特征流与选择掩码 → 在特征空间融合上一帧人体、源人物锚点与服装特征 → 输出时序平滑的连续试穿视频
claims:
  - "Claim 1: 在 FashionVideo 的 2000 个重建视频片段上，FashionMirror 在报告的基线中取得最优 SSIM 0.923、LPIPS 0.057 和 VFID 3.097/1.033 [evidence: comparison]"
  - "Claim 2: 相比预处理语义分割，CMN 的平均掩码预测耗时从 0.3469s 降至 0.1983s，推理时间减少 42.84% [evidence: comparison]"
  - "Claim 3: 去掉 hs boost、去掉 λci 或改成 multi-flow 会将 VFID(3D-ResNet) 从 1.033 恶化到 1.690、1.206、1.551，说明源人物锚点、层级服装权重和单流设计都对时序试穿有效 [evidence: ablation]"
related_work_position:
  extends: "FWGAN (Dong et al. 2019)"
  competes_with: "FWGAN (Dong et al. 2019); FashionOn (Hsieh et al. 2019)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/ICCV_2021/2021_FashionMirror_Co_Attention_Feature_Remapping_Virtual_Try_On_With_Sequential_Template_Poses.pdf
category: Video_Generation
---

# FashionMirror: Co-Attention Feature-Remapping Virtual Try-On With Sequential Template Poses

> [!abstract] **Quick Links & TL;DR**
> - **Links**: GitHub: https://github.com/FashionMirror/FashionMirror
> - **Summary**: 该工作把视频虚拟试穿从“依赖外部人体解析 + 像素级衣服贴图”改成“共注意力掩码预测 + 特征级人体/服装重映射”，从而在连续姿态下生成更稳定、更自然的试穿视频。
> - **Key Performance**: FashionVideo 上达到 SSIM 0.923、LPIPS 0.057、VFID(3D-ResNet) 1.033；掩码预测平均耗时由 0.3469s 降至 0.1983s（-42.84%）。

> [!info] **Agent Summary**
> - **task_path**: 单张源人物图像 + 目标服装图像 + 引导姿态序列 -> 连续姿态虚拟试穿视频
> - **bottleneck**: 语义分割预处理慢且会误差传递，像素级衣服 warping 在遮挡与新视角下不稳，逐帧生成又容易闪烁
> - **mechanism_delta**: 用共注意力直接预测脱衣区域与目标穿衣区域，并用骨架驱动的特征流在特征空间重映射人体与服装，而不是在像素空间硬融合
> - **evidence_signal**: 在单一 FashionVideo 基准上同时取得最优 SSIM/LPIPS/VFID，并在 120 人用户研究中获 52.76% 最高投票
> - **reusable_ops**: [co-attention mask prediction, skeleton-conditioned feature warping]
> - **failure_modes**: [大姿态跳变时流场学习可能失稳, 训练仍依赖外部解析器生成伪标签与外部姿态估计]
> - **open_questions**: [在真实任意衣服-任意人物配对上能否保持优势, 能否扩展到高分辨率实时部署]

## Part I：问题与挑战

**任务定义**  
给定源人物图像 \(h^s\)、目标服装图像 \(C\) 和引导姿态序列，生成人物穿上目标服装后的连续视频帧。核心要求不是单帧好看，而是：

1. 衣服要对齐到正确人体区域；
2. 遮挡关系要合理；
3. 连续帧不能闪烁；
4. 最好还能支持不同视角下的衣物外观变化。

**真正的瓶颈**  
这篇论文抓得很准：问题不只是“把衣服贴上去”，而是**现有 try-on 管线的表示层级选错了**。

- **像素级 warping 的硬伤**：  
  以前的方法常把服装图直接 warp 到人体上，再做像素融合。这样一旦对齐有偏差，错误会直接显示出来；而且前视图里没有的信息，模型也很难补出侧视图内容。
- **逐帧生成缺乏时序约束**：  
  如果每帧独立做 try-on，会有明显 flicker。  
  如果先合成第一帧再做 pose transfer，衣服信息又会被锁死在第一帧并逐步累积误差。
- **依赖预处理人体解析**：  
  语义分割虽然能告诉模型“衣服该在哪”，但预处理耗时，而且一旦 parsing 错了，后面全错。

**为什么现在值得解**  
论文的应用场景是“fashion mirror”——用户站在镜子前，换衣并连续摆姿势查看效果。这个场景天然要求：

- **低延迟**
- **多视角一致**
- **视频级平滑**

所以它不是单纯追求更高图像分数，而是在追一个更接近真实零售体验的接口。

**输入/输出与边界条件**

- **输入**：单人源图、目标衣服图、引导姿态序列
- **输出**：连续试穿视频
- **边界条件**：
  - 单人场景
  - 2D pose 驱动
  - 256×256 分辨率
  - 定量实验主要建立在 FashionVideo 上
  - 由于缺少“任意衣服 + 连续姿态视频”的公开 paired 数据，论文的客观指标主要是**重建式评测**而非完全开放式 try-on

---

## Part II：方法与洞察

整体上，FashionMirror 是一个**两阶段框架**：

1. **Stage I：Parsing-free Co-attention Mask Prediction**
2. **Stage II：Human and Clothing Feature Remapping**

### 核心直觉

这篇论文最关键的变化可以概括成一句话：

**把“衣服该穿到哪里”和“人物如何从上一帧运动到下一帧”都搬到特征空间里处理，而不是在像素空间硬贴图。**

这带来三个因果变化：

1. **从通用人体解析 → 任务相关掩码**
   - 改变前：依赖外部 semantic parsing，把人体拆成很多 body-part channel。
   - 改变后：直接预测两个与 try-on 最相关的掩码：
     - `Mr`：源图里要移除的衣服区域
     - `Mc`：目标衣服应该落到的人体区域
   - 结果：减少外部预处理依赖，也把监督聚焦到“服装相关空间”而不是完整人体语义。

2. **从像素级 warping → 特征级 remapping**
   - 改变前：衣服 warp 错一点，结果就直接穿帮。
   - 改变后：先在特征空间对齐，再交给卷积层细化。
   - 结果：网络有机会“修”错位，也更有机会补全未直接观测到的新视角衣物内容。

3. **从独立帧生成 → 相邻姿态驱动的递推生成**
   - 改变前：每帧独立做，时序不连续。
   - 改变后：通过相邻姿态估计 feature flow，再决定下一帧更多继承上一帧还是回看源图。
   - 结果：时序更平滑，且不容易长期漂移。

### 方法拆解

#### 1) 共注意力掩码网络 CMN

CMN 输入源人物图和目标服装图，输出：

- **removed mask \(Mr\)**：源人物身上原衣服该被去掉的区域
- **try-on clothing mask \(Mc\)**：目标衣服在当前人体上的落位区域

做法上，它先分别提取 human/clothing features，再计算二者的**feature similarity**，通过共注意力把“人物上哪些区域和目标衣服有关”显式找出来，然后据此预测掩码。

这里的关键不是又造了一个 segmentation，而是：

- 它预测的是**try-on 任务专用掩码**
- 它避免在推理阶段跑完整人体解析
- 它让后续模块拿到的是“穿衣相关结构信息”，而不是全套 body-part map

#### 2) 骨架流提取 SFE

第二阶段先把姿态序列转成 skeleton 表示。论文没有只用 18 个关节点，而是用了更细的 137 点表示，包含身体、手和脸。

然后，SFE 从相邻骨架里预测：

- **feature-level optical flow \(F^t\)**：上一帧的人体特征怎么移动到下一帧
- **selection mask \(m^t\)**：下一帧特征应该更多来自上一帧，还是来自源人物锚点

这里还有两个很实用的设计：

- **只学相邻小位移帧的流**：训练时只随机跳 0–2 帧，降低大位移 flow 学习难度
- **不用外部 flow GT**：而是用 sample correctness loss 在 VGG 特征空间里学一个对 try-on 更合适的数据驱动 flow

这实际上把“通用光流”换成了“为人体姿态迁移与试穿服务的光流”。

#### 3) 人体与服装特征重映射 HSG

最终生成器每一帧做三件事：

1. 用 `Mr` 把源图原衣服先抹掉，避免旧衣服干扰
2. 用 flow 把上一帧人体特征 warp 到下一帧姿态
3. 再叠加：
   - **源人物特征锚点**：保留身份/脸部/人体细节
   - **目标服装特征**：根据 `Mc` 对齐并注入当前帧

这一步很重要：它不是只“追着前一帧跑”，而是始终保留一个**source human anchor**。  
所以它能减轻递归生成常见的 drift 问题。

### 战略取舍

| 设计选择 | 改变的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 共注意力掩码替代预处理 parsing | 从通用人体分割转为任务相关服装区域定位 | 更快，减少 parsing 误差传递 | 训练时仍需解析器生成伪监督 |
| 特征级 remapping 替代像素级贴图 | 从硬对齐变为可细化的软对齐 | 更稳，遮挡处理更自然，可补部分新视角内容 | 更依赖特征提取与解码器能力 |
| 相邻骨架流学习 | 降低大位移光流学习难度 | 时序更平滑，减少 flicker | 对超大姿态跳变的泛化有限 |
| selection mask + source boost | 给递归视频生成提供静态外观锚点 | 减少长期漂移，保住脸和人体细节 | 源图偏差也可能被持续带入 |

---

## Part III：证据与局限

### 关键证据

**1. 对比实验信号：它不是只提升单帧，而是单帧与视频一致性一起提升。**  
在 FashionVideo 上，FashionMirror 同时拿到最优：

- **SSIM 0.923**
- **LPIPS 0.057**
- **VFID(I3D) 3.097**
- **VFID(3D-ResNet) 1.033**

这点很关键，因为很多方法会在“单帧好看”和“视频稳定”之间二选一；而这里作者声称两者兼得。

**2. 用户研究信号：主观观感优势明显。**  
120 名志愿者、13 组输入上，FashionMirror 获得 **52.76%** 投票，高于任一单个基线；而且在**同服装类型**和**不同服装类型**两种子集里都领先。  
这说明它的优势不是某一种简单情形下才成立。

**3. 效率信号：共注意力掩码确实替代了一部分 parsing 成本。**  
CMN 平均耗时 **0.1983s**，而语义分割预处理是 **0.3469s**，时间减少 **42.84%**。  
所以“parsing-free”在推理效率上不只是叙事，而是有实际收益。

**4. 消融信号：关键收益来自设计组合，而不是某一个巧合模块。**  
完整模型优于：

- **w/o hs boost**
- **w/o λci**
- **multi-flow**

尤其 multi-flow 反而更差，说明这里不是“flow 越多越好”，而是**单一清晰的人体运动流 + 源图锚点 + 服装特征补充**这个组合更稳定。  
论文的可视化也显示：去掉源图增强后，flow 容易散到整张图，而不是聚焦在人体区域。

**5. 指标解读信号：IS 并不可靠。**  
FashionOn 的 IS 更高，但视觉结果和用户研究并不更好。作者指出 try-on 场景下基于 ImageNet 的 IS 并不总能反映真实试穿质量。  
这意味着该论文的核心胜利证据更应看 **LPIPS / VFID / user study**。

### 局限性

- **Fails when**: 姿态跨帧变化过大、目标服装需要强新视角补全、或者服装拓扑差异特别大时，特征流与掩码仍可能错位；因为其 flow 学习主要建立在 0–2 skip 的相邻小运动上。
- **Assumes**: 单人视频、较准确的 OpenPose 骨架、训练期可用人体解析器 [12] 生成 `Mr/Mc` 伪真值；也就是说它是**推理阶段 parsing-free**，不是完全不依赖 parsing。定量评测主要在“同衣服重建”设定上进行，离真实任意服装试穿还有一层距离。训练资源上使用了两张 2080 Ti。
- **Not designed for**: 高分辨率商用实时部署、3D cloth physics、多人交互遮挡场景、无姿态序列输入的自由式试穿。

**额外复现提示**  
论文正文给出了 GitHub 仓库链接，但正文明确说明的是“视频示例”可见；完整代码/训练脚本开放程度建议实际核验，因此这里保守标为 `opensource/partial`。

### 可复用组件

- **CMN**：可作为任何 image/video try-on 系统里的 task-specific garment mask predictor
- **SFE + sample correctness**：可复用于 pose-guided human animation 的特征迁移
- **source-anchor + selection-mask 融合**：可复用于递归视频生成里抑制 drift

---

![[paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/ICCV_2021/2021_FashionMirror_Co_Attention_Feature_Remapping_Virtual_Try_On_With_Sequential_Template_Poses.pdf]]