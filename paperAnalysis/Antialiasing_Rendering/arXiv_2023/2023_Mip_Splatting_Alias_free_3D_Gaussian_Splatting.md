---
title: "Mip-Splatting: Alias-free 3D Gaussian Splatting"
venue: arXiv
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/novel-view-synthesis
  - anti-aliasing
  - mip-filtering
  - gaussian-smoothing
  - dataset/Blender
  - "dataset/Mip-NeRF 360"
  - opensource/no
core_operator: 基于训练视角的采样率上界为每个3D高斯施加带限平滑，并以近似像素盒滤波的2D Mip滤波替代固定屏幕空间膨胀
primary_logic: |
  多视图图像与相机参数 → 按训练视角可见性估计每个高斯的最大可重建频率并施加3D低通约束 → 渲染时用2D Mip滤波替代固定2D膨胀以近似像素积分 → 在缩放变化下输出更稳定、低混叠的新视角图像
claims:
  - "在 Blender 的单尺度训练/多尺度测试设置下，Mip-Splatting 的平均 PSNR/LPIPS 为 31.97/0.024，优于 3DGS 的 24.84/0.063 和 3DGS+EWA 的 29.40/0.034 [evidence: comparison]"
  - "在 Mip-NeRF 360 的 8×下采样训练、跨分辨率测试设置下，Mip-Splatting 的平均 PSNR 提升到 27.37，超过 3DGS 的 23.25 与 3DGS+EWA 的 25.43，同时平均 SSIM 达到 0.803 [evidence: comparison]"
  - "消融表明 3D smoothing 主要抑制 zoom-in 高频伪影，而 2D Mip filter 主要抑制 zoom-out 混叠；去掉前者会使 Mip-NeRF 360 平均 PSNR 从 27.37 降到 26.93，去掉后者会使 Blender 平均 PSNR 从 31.97 降到 30.76 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "3D Gaussian Splatting (Kerbl et al. 2023); Mip-NeRF (Barron et al. 2021)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Antialiasing_Rendering/arXiv_2023/2023_Mip_Splatting_Alias_free_3D_Gaussian_Splatting.pdf
category: 3D_Gaussian_Splatting
---

# Mip-Splatting: Alias-free 3D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2311.16493), [Project](https://niujinshuchong.github.io/mip-splatting)
> - **Summary**: 这篇工作把 3DGS 里“依赖屏幕空间固定膨胀兜底”的做法，改成“训练视角采样率约束下的 3D 带限表示 + 近似像素积分的 2D Mip 滤波”，从而显著提升跨缩放渲染的稳定性。
> - **Key Performance**: Blender 单尺度训练/多尺度测试平均 PSNR 31.97（3DGS 为 24.84）；Mip-NeRF 360 的 zoom-in 设定平均 PSNR 27.37（3DGS+EWA 为 25.43）

> [!info] **Agent Summary**
> - **task_path**: 多视图图像+相机参数 -> 显式3D高斯场 -> 跨分辨率/跨焦距新视角图像
> - **bottleneck**: 3DGS 没有在 3D 表示层约束可重建频率，且固定 2D 膨胀与真实成像不匹配，导致 zoom-in 出现侵蚀/高频伪影，zoom-out 出现膨胀/混叠
> - **mechanism_delta**: 用“按训练视角采样率决定的 3D smoothing”消除退化细高斯，再用“近似单像素 box filter 的 2D Mip filter”替代固定 dilation
> - **evidence_signal**: 两个数据集上的单尺度训练/多尺度测试都显著优于 3DGS 与 3DGS+EWA，且消融清晰分离了 3D/2D 两个滤波器的职责
> - **reusable_ops**: [per-primitive sampling-rate bound, 3D Gaussian low-pass fusion, pixel-footprint-aware screen-space filtering]
> - **failure_modes**: [极端 zoom-out 时高斯近似 box filter 会带来误差, 采样率估计若受深度近似与遮挡忽略影响可能出现过平滑或欠平滑]
> - **open_questions**: [能否用更精确且仍高效的 box/sensor filter 替代 Gaussian 近似, 能否把采样率估计预计算或 CUDA 化以降低训练开销]

## Part I：问题与挑战

这篇论文要解决的不是“3DGS 不够清晰”，而是一个更本质的问题：**3DGS 的表示与渲染对采样率变化不稳健**。  
当测试视角的焦距、相机距离或输出分辨率和训练时不一致时，原始 3DGS 会出现明显伪影。

### 真正的瓶颈是什么？

作者指出，问题有两个来源，而且分别作用在**表示层**与**成像层**：

1. **3D 表示层没有频率约束**  
   3DGS 在反向渲染优化时，会倾向学出过小、过薄、近似退化的 3D Gaussian。  
   这些“细得不合理”的高斯在训练尺度下，靠后续的 2D dilation 仍能渲染出看似正确的图像，因此优化不会主动纠正它们。

2. **屏幕空间固定 dilation 不是物理一致的成像模型**  
   原始 3DGS 用一个经验性的 2D 膨胀项，避免高斯投影太小。但这个膨胀量不随采样率变化而变化：
   - **zoom-in / 更高采样率**：退化细高斯被放大后暴露出来，出现侵蚀、缝隙和高频伪影；
   - **zoom-out / 更低采样率**：固定膨胀会把本应小于像素 footprint 的能量错误扩散，出现变亮、变粗和混叠。

### 为什么现在值得解决？

因为 3DGS 的优势正是**高质量 + 实时渲染**。  
一旦进入真实应用，用户会自然改变：
- 输出分辨率，
- 焦距，
- 相机距离，
- 甚至设备显示尺度。

如果方法只在训练尺度附近有效，那它虽然“快”，却不够“稳”。作者认为，过去很多 3DGS 评测主要是 in-distribution，同尺度训练/测试，掩盖了这个问题。

### 输入/输出接口与边界条件

- **输入**：多视图 RGB 图像 + 已知相机内外参
- **中间表示**：显式 3D Gaussian 集合
- **输出**：目标视角、目标采样率下的新视角图像

**边界条件**：
- 主要针对静态场景 NVS；
- 重点讨论训练单一尺度、测试多尺度的 OOD 泛化；
- 目标是**无混叠、物理更一致的缩放渲染**，不是超分辨率幻觉。

---

## Part II：方法与洞察

作者的设计非常清楚：  
**不要只在 2D 屏幕末端补锅，而要先把 3D 表示本身变成“带限”的，再让 2D 渲染更接近真实像素积分。**

### 核心直觉

原方法的问题可以概括为：

- **what changed**：从“固定 2D dilation + 无 3D 频率约束”改成“训练视角决定的 3D smoothing + 近似单像素积分的 2D Mip filtering”
- **which bottleneck changed**：把原来 3D 反演中的尺度歧义，变成受 Nyquist 采样上界约束的带限表示；同时把渲染阶段的经验膨胀，改成与像素 footprint 对齐的成像近似
- **what capability changed**：模型不再只在训练尺度看起来对，而是在 zoom-in / zoom-out 时都更稳定

更因果地说：

- **3D smoothing** 改变了“什么样的 3D 结构被允许学出来”；
- **2D Mip filter** 改变了“这些结构在不同采样率下如何投到像素上”。

前者解决**表示歧义**，后者解决**成像混叠**。这也是为什么作者的消融会显示二者分工明确。

### 方法拆解

#### 1) 3D smoothing filter：先把 3D 表示做带限

作者依据 Nyquist-Shannon 采样定理，认为一个 3D 局部结构能被重建到多高频率，受训练图像采样率限制。  
对每个 Gaussian，作者用其在各训练相机中的可见性、深度近似和焦距，估计该 primitive 的**最大可重建采样率**。

直观上看：

- 离相机更近、焦距更大、在图像上投影更清楚的观察，会提供更高采样率；
- 作者对每个 Gaussian 取“所有可见训练视图里最高的那个采样率”作为上界；
- 然后在 3D 空间里给这个 Gaussian 施加一个低通平滑，阻止它变成不合理的高频尖峰。

关键点在于：  
这不是测试时临时加的滤波，而是**训练中直接写进表示本身**。  
训练结束后，这个平滑相当于被融合进 Gaussian 参数里，因此不会随着视角变化而漂移。

**能力变化**：  
它主要抑制 zoom-in 时暴露出来的高频伪影和侵蚀问题。

#### 2) 2D Mip filter：让屏幕空间更像真实像素积分

即便 3D 表示被带限了，当你 zoom-out 时，仍然可能因为采样率下降而产生 aliasing。  
所以作者又替换了 3DGS 的固定 dilation，改用一个**2D Mip filter**。

其设计动机不是“再加一个经验模糊核”，而是：

- 真实成像中，一个像素会对落入其面积内的光子做积分；
- 理想模型是 box filter；
- 为了高效实现，作者用 Gaussian 来近似这个 box filter。

这带来两个差异：

1. 它的目标是**近似像素面积积分**，不是经验性地让点“更大更安全”；
2. 与原 dilation 不同，它会随 footprint 改变能量分布，减少 zoom-out 时的假亮、假粗和混叠。

作者还特别对比了 EWA：  
EWA 更像是在 screen space 里经验性地带限输出，滤波大小常靠经验设置，容易在 zoom-out 时过平滑；Mip-Splatting 的 2D filter 则更明确地对齐“单像素成像”这件事。

### 为什么这套设计有效？

因为它把问题拆对了层级：

- **3D 层**：限制不可观测的虚假高频，减少逆问题的自由度；
- **2D 层**：匹配像素采样过程，减少随采样率变化的渲染失真。

如果只做 2D 处理，退化细高斯仍会被学出来，zoom-in 还是会炸；  
如果只做 3D 处理，zoom-out 仍然会 alias。  
论文最重要的洞察，就是这两件事必须同时做，但职责不同。

### 战略取舍

| 设计选择 | 主要解决的问题 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 3D smoothing filter | 退化细高斯、zoom-in 高频伪影 | 表示层带限，跨尺度更稳 | 需要周期性估计每个 Gaussian 的采样率；过强会有过平滑风险 |
| 2D Mip filter | zoom-out 的 aliasing、亮度/粗细失真 | 更接近像素积分的渲染 | box filter 只被 Gaussian 近似，极端缩小时会有误差 |
| 采样率每 m 次重算 | 让约束随训练过程更新 | 保持 per-Gaussian 约束有效 | 有额外训练开销，当前实现主要在 PyTorch 中完成 |

---

## Part III：证据与局限

### 关键证据信号

#### 信号 1：真正的新设定里，跨尺度泛化明显提升
最有说服力的实验不是同尺度测试，而是**单尺度训练、多尺度测试**。

- 在 **Blender** 上，训练只用全分辨率图像、测试覆盖 Full / 1/2 / 1/4 / 1/8：
  - Mip-Splatting 平均 **PSNR 31.97**
  - 3DGS 为 **24.84**
  - 3DGS + EWA 为 **29.40**

这说明它不是简单地“模糊掉一切”，而是在 zoom-out 场景里显著更稳。

#### 信号 2：zoom-in 场景下，能压住 3DGS 的侵蚀与高频伪影
在 **Mip-NeRF 360** 上，作者把训练图像下采样 8×，再在更高分辨率下渲染，模拟 zoom-in / 靠近相机：

- Mip-Splatting 平均 **PSNR 27.37**
- 3DGS 为 **23.25**
- 3DGS + EWA 为 **25.43**

而且可视化里，3DGS 的细结构会“断、细、裂”，3DGS+EWA 虽然改善，但仍有高频伪影；Mip-Splatting 更接近 GT。  
这直接支撑了论文的核心判断：**问题首先在 3D 表示退化，而不只是 2D 输出不够平滑。**

#### 信号 3：不是拿 OOD 泛化换 ID 退化
在标准的 **Mip-NeRF 360 同尺度训练/同尺度测试**上：

- Mip-Splatting：PSNR **27.79**
- 3DGS + EWA：**27.77**
- 重训练的 3DGS：**27.70**

即：加入 alias-free 机制后，**常规设置没有明显掉点**。  
这是很重要的系统信号，说明方法不是以牺牲基线能力为代价获得 OOD 稳定性。

#### 信号 4：消融验证了两种滤波器确实分工不同
补充材料的消融很关键：

- 去掉 **3D smoothing**：zoom-in 性能下降，高分辨率下高频伪影回来；
- 去掉 **2D Mip filter**：zoom-out 性能显著下降，aliasing 回来；
- 两者都去掉时，甚至会因为大量小 Gaussian 被密度控制机制生成而出现 OOM。

这说明论文的机制不是“拍脑袋堆模块”，而是两个因果旋钮分别打在两个不同故障源上。

### 局限性

- **Fails when**: 极端 zoom-out 时，Gaussian 对 box filter 的近似误差会更明显；另外采样率估计用 Gaussian center 近似深度、并忽略遮挡，会在复杂遮挡边界带来不精确的带限约束。
- **Assumes**: 依赖准确的相机内外参与可见性估计；依赖 3DGS 的静态场景设定与密度控制机制；训练期需要每 100 次迭代重算一次 per-Gaussian sampling rate，当前实现存在额外开销。
- **Not designed for**: 从低分辨率训练数据中“幻觉”出未观测的真实高频细节；严格物理精确的传感器成像建模；与论文设定无关的动态场景或时序一致性问题。

### 可复用组件

这篇工作最值得迁移的，不只是一个具体实现，而是三个通用操作：

1. **per-primitive multiview sampling-rate bound**  
   用训练视图的焦距/深度/可见性，为显式 primitive 估计可重建频率上界。

2. **representation-level low-pass fusion**  
   不把抗混叠只放在渲染端，而是直接把低通写进 3D 表示本身。

3. **pixel-footprint-aware screen-space filtering**  
   用与像素成像更一致的 footprint filter，替代经验性 dilation。

一句话总结“so what”：

> Mip-Splatting 的能力跃迁不在于它让 3DGS 更“锐”，而在于它第一次把 3DGS 从“只在训练尺度看起来对”推进到“在采样率变化下仍然物理上更合理地对”。

![[paperPDFs/Antialiasing_Rendering/arXiv_2023/2023_Mip_Splatting_Alias_free_3D_Gaussian_Splatting.pdf]]