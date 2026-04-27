---
title: 'MultiWorld: Scalable Multi-Agent Multi-View Video World Models'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.18564
aliases:
- 多智能体多视角视频世界模型MultiWorld
- MultiWorld
code_url: https://github.com/CIntellifusion/MultiWorld
method: MultiWorld
modalities:
- Image
---

# MultiWorld: Scalable Multi-Agent Multi-View Video World Models

[Paper](https://arxiv.org/abs/2604.18564) | [Code](https://github.com/CIntellifusion/MultiWorld)

**Topics**: [[T__Video_Generation]], [[T__Agent]], [[T__Robotics]] | **Method**: [[M__MultiWorld]]

| 中文题名 | 多智能体多视角视频世界模型MultiWorld |
| 英文题名 | MultiWorld: Scalable Multi-Agent Multi-View Video World Models |
| 会议/期刊 | 2026 (arXiv预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.18564) · [Code](https://github.com/CIntellifusion/MultiWorld) · [Project] |
| 主要任务 | 多智能体(multi-agent)多视角(multi-view)视频世界建模，动作可控的视频生成，长程轨迹模拟 |
| 主要 baseline | 单智能体世界模型(Sora, Open-Sora, Latte, CogVideoX)、多视角生成方法(MVDiffusion, SyncDreamer)、视频预测模型 |

> [!abstract] 因为「现有世界模型仅能处理单智能体单视角场景，无法建模多智能体交互与多视角一致性」，作者在「扩散Transformer视频生成框架」基础上改了「多智能体多视角联合建模机制与可扩展训练策略」，在「多玩家游戏与多机器人协作任务」上取得「多视角一致性提升与长程动作可控生成」。

- **关键性能**: 多智能体场景下视频生成FID ，多视角视角一致性PSNR/SSIM 
- **关键性能**: 支持3+智能体、多视角同步生成，长程生成达帧
- **关键性能**: 在真实机器人协作任务中实现失败轨迹模拟（见图4）

## 背景与动机

现实世界中的动态场景往往涉及多个智能体(agent)从多个视角同时观察与交互——例如自动驾驶车辆的多摄像头协同、多机器人协作装配、或多人游戏中的群体行为。现有视频世界模型面临一个根本性瓶颈：它们被设计为单智能体、单视角的"第一人称"模拟器，无法同时建模(1)多个智能体之间的交互动力学，以及(2)同一时刻不同视角之间的几何一致性。

具体而言，Sora/Open-Sora等大规模视频生成模型虽能生成高质量视频，但缺乏显式的动作控制与多视角约束；Latte、CogVideoX等扩散Transformer模型专注于单视频序列的时序建模，未考虑空间视角关联。多视角生成方法如MVDiffusion、SyncDreamer虽能实现新视角合成，但局限于静态场景或单物体，无法处理动态多智能体交互。视频预测模型(如。

这些方法的共同局限在于：**将"多智能体"与"多视角"解耦处理**——要么只扩展智能体数量而固定视角，要么只扩展视角数量而简化动态。这导致在需要同时模拟"多个机器人协作搬运"或"多人游戏中相互遮挡"等场景时，出现视角不一致、动作不同步、交互物理不合理等问题。

本文提出MultiWorld，首次将多智能体交互建模与多视角几何一致性统一于可扩展的视频世界模型框架中，实现给定初始多视角观测与每帧动作条件下的可控长程视频生成。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/883162f2-3a03-471b-b778-96aade7a773e/figures/Figure_1.png)
*Figure 1: Fig. 1: MultiWorld generates multi-agent multi-view videos.Given initialviews and per-frame actions, our model produces action-controllable, multi-view con-sistent videos in both multi-player video ga*



## 核心创新

核心洞察：**多智能体的联合状态空间可通过"视角-智能体-时间"三维注意力分解来高效建模**，因为不同智能体在同一视角下共享场景几何、同一智能体在不同视角下共享运动语义，从而使线性复杂度的多智能体多视角联合推断成为可能。

| 维度 | Baseline (单智能体/单视角) | 本文 (MultiWorld) |
|:---|:---|:---|
| 表征空间 | 单视频序列的时空latent $z \in \mathbb{R}^{T \times H \times W \times C}$ | 多视角多智能体联合latent $Z \in \mathbb{R}^{V \times A \times T \times H \times W \times C}$ |
| 注意力机制 | 仅时序自注意力 (temporal self-attention) | 分解式三维注意力：视角内(intra-view)、智能体间(inter-agent)、跨视角(cross-view) |
| 动作条件 | 全局场景级动作向量 | 每智能体每帧动作序列，显式交互编码 |
| 训练扩展 | 单视频独立训练，batch内无结构 | 智能体-视角维度可任意扩展的并行训练策略 |
| 应用场景 | 单机器人/单摄像头模拟 | 多机器人协作、多人游戏、分布式传感器网络 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/883162f2-3a03-471b-b778-96aade7a773e/figures/Figure_2.png)
*Figure 2: Fig. 2: Pipeline of MultiWorld. We propose MultiWorld, a unified framework forscalable multi-agent multi-view video world modeling. In the Multi-Agent ConditionModule (Sec. 3.2), Agent Identity Embedd*



MultiWorld的整体数据流遵循"编码-分解注意力-解码"的pipeline，核心模块如下：

**输入**: 初始多视角观测 $I_0 = \{I_0^{v,a}\}_{v=1,a=1}^{V,A}$（$V$个视角，$A$个智能体）+ 每帧每智能体动作序列 $\{a_t^a\}_{t=1,a=1}^{T,A}$

→ **模块1: 多视角多智能体视觉编码器 (Multi-View Multi-Agent VAE Encoder)**
将各视角各智能体的初始帧独立编码为latent token，输出 $z_0^{v,a} \in \mathbb{R}^{h \times w \times c}$，保持空间结构。

→ **模块2: 分解式三维注意力Transformer (Factorized 3D Attention Transformer)**
核心创新模块。将传统时空注意力分解为三个正交子空间：
- **Intra-view attention**: 同一视角内跨智能体、跨时间聚合（解决"同一摄像头下多个机器人如何协作"）
- **Inter-agent attention**: 同一智能体跨视角、跨时间聚合（解决"同一机器人的多摄像头观测如何一致"）
- **Cross-view attention**: 固定时间步下跨视角、跨智能体聚合（解决"瞬时多视角几何一致性"）

→ **模块3: 动作条件交互模块 (Action-Conditioned Interaction Module)**
将每智能体的动作序列编码为交互token，通过显式的智能体间通信机制(intelligent communication)注入注意力层。

→ **模块4: 多视角多智能体解码器 (Multi-View Multi-Agent VAE Decoder)**
将处理后的latent token独立解码为各视角各智能体的视频帧，保持像素级一致性。

→ **输出**: 未来多视角多智能体视频序列 $\{\hat{I}_t^{v,a}\}_{t=1,a=1,v=1}^{T,A,V}$

```
初始观测 + 动作序列
    ↓
[Multi-View Multi-Agent VAE Encoder] → latent tokens z^{v,a}
    ↓
[Factorized 3D Attention Transformer]
  ├─ Intra-view attention (视角内跨智能体)
  ├─ Inter-agent attention (智能体间跨视角)  
  └─ Cross-view attention (瞬时跨视角几何)
    ↓
[Action-Conditioned Interaction Module] ← per-agent per-frame actions
    ↓
[Multi-View Multi-Agent VAE Decoder]
    ↓
未来多视角多智能体视频 {Î_t^{v,a}}
```

## 核心模块与公式推导

### 模块1: 分解式三维注意力机制（对应框架图 Transformer核心）

**直觉**: 完整的多智能体多视角联合注意力具有 $O((V \cdot A \cdot T)^2)$ 复杂度，不可扩展；通过分解为三个低维子空间，可将复杂度降至 $O(V^2 + A^2 + T^2)$ 量级。

**Baseline公式** (标准时空Transformer如Latte):
$$\text{Attn}_{\text{base}}(Q,K,V) = \text{softmax}\left(\frac{QK^\text{top}}{\sqrt{d_k}}\right)V$$
其中 $Q,K,V \in \mathbb{R}^{(V \cdot A \cdot T \cdot h \cdot w) \times d}$ 为展平后的全联合token，计算不可行。

**符号**: $V$=视角数, $A$=智能体数, $T$=时间步, $h,w$=空间分辨率, $d$=特征维度; $z^{v,a}_t \in \mathbb{R}^{hw \times d}$ 为特定视角-智能体-时间步的token。

**变化点**: Baseline将 $(V,A,T,h,w)$ 展平为一维序列，丢失了多智能体多视角的结构性先验，且复杂度随智能体/视角数平方爆炸。本文假设：**视角一致性、智能体交互、时序动力学可正交分解**。

**本文公式（推导）**:
$$\text{Step 1: 视角内注意力 (Intra-view)}: \quad z^{v,\cdot}_t = \text{Attn}\left(Q=z^{v,a}_t, K=z^{v,a'}_t, V=z^{v,a'}_t\right)_{a'=1}^{A}$$
同一视角 $v$ 下，聚合所有智能体信息，解决"摄像头v看到多个智能体如何交互"。

$$\text{Step 2: 智能体间注意力 (Inter-agent)}: \quad z^{\cdot,a}_t = \text{Attn}\left(Q=z^{v,a}_t, K=z^{v',a}_t, V=z^{v',a}_t\right)_{v'=1}^{V}$$
同一智能体 $a$ 下，聚合所有视角信息，保证"机器人a的多摄像头观测一致"。

$$\text{Step 3: 跨视角几何注意力 (Cross-view)}: \quad z^{v,a}_{\cdot} = \text{Attn}\left(Q=z^{v,a}_t, K=z^{v',a'}_t, V=z^{v',a'}_t\right)_{v'\neq v \text{ or } a'\neq a}$$
固定时间步下，显式建模视角-智能体间的几何对应关系。

$$\text{最终}: \quad z_{\text{out}} = \text{LN}\left(\text{FFN}\left(\text{Concat}\left[z^{v,\cdot}_t, z^{\cdot,a}_t, z^{v,a}_{\cdot}\right]\right)\right) + z_{\text{in}}$$
通过Concat+FFN融合三个子空间输出，LayerNorm+残差保证稳定训练。

**对应消融**: 

---

### 模块2: 动作条件化与智能体交互编码（对应框架图 Action Module）

**直觉**: 多智能体场景需要显式建模"谁做了什么"以及"动作如何影响其他智能体"，而非简单拼接全局动作向量。

**Baseline公式** (标准视频预测如Sora的动作条件):
$$c_{\text{base}} = \text{MLP}(a_{\text{global}}) \in \mathbb{R}^d, \quad \text{通过adaLN注入}: \gamma \cdot z + \beta$$
全局动作向量 $a_{\text{global}}$ 无法区分不同智能体的独立决策。

**符号**: $a_t^a \in \mathbb{R}^{d_a}$ 为智能体 $a$ 在时刻 $t$ 的动作; $E_a \in \mathbb{R}^{d_a \times d}$ 为可学习的智能体嵌入。

**变化点**: Baseline假设单一全局动作控制整个场景，不适用于多智能体分布式决策。本文引入**每智能体动作嵌入 + 显式智能体间通信**。

**本文公式（推导）**:
$$\text{Step 1: 个体动作编码}: \quad e_t^a = \text{MLP}\left([a_t^a; E_a]\right) \in \mathbb{R}^d$$
将动作与智能体身份拼接编码，区分"机器人1抓取"与"机器人2放置"。

$$\text{Step 2: 智能体间通信 (Inter-agent Communication)}: \quad m_t^a = \sum_{a' \neq a} \alpha_{a,a'} \cdot W_m e_t^{a'}$$
其中 $\alpha_{a,a'} = \text{softmax}_a\left((W_q e_t^a)^\text{top} (W_k e_t^{a'})\right)$ 为注意力权重，实现"机器人a关注机器人b的动作意图"。

$$\text{Step 3: 融合注入}: \quad c_t^{v,a} = e_t^a + m_t^a, \quad \hat{z}_t^{v,a} = \text{AdaLN}(z_t^{v,a}, c_t^{v,a})$$
将个体动作与通信消息融合，通过自适应层归一化注入Transformer各层。

$$\text{最终动作条件损失}: \quad L_{\text{action}} = \mathbb{E}_{v,a,t}\left[\|a_t^a - \hat{a}_t^a\|^2\right]$$
额外添加动作重建约束，保证生成的视频与给定动作严格对齐。

**对应消融**: 

## 实验与分析

| Method | 多视角一致性 (PSNR↑) | 动作可控性 (Acc↑) | 长程稳定性 (FVD↓) | 智能体扩展性 (A=3/5/10) |
|:---|:---|:---|:---|:---|
| Sora (单智能体/单视角) | N/A |  |  | 不支持 |
| Open-Sora |  |  |  | 不支持 |
| Latte |  |  |  | 不支持 |
| MVDiffusion (多视角静态) |  | N/A | N/A | 不支持动态 |
| **MultiWorld (本文)** | **** | **** | **** | **支持任意扩展** |


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/883162f2-3a03-471b-b778-96aade7a773e/figures/Figure_3.png)
*Figure 3: Fig. 3: Qualitative comparison of multi-agent multi-view video generationin a multi-player video game. Our method achieves more accurate action followingability and better multi-view consistency compa*



**主结果分析**: 在多玩家游戏数据集（见图3定性比较）与多机器人操作任务（见图5长程生成）上，MultiWorld实现了的多视角PSNR提升与的动作执行准确率。关键支撑在于：分解式注意力机制使视角数 $V$ 和智能体数 $A$ 可独立扩展而不耦合爆炸；显式动作条件化保证了每帧生成严格遵循输入动作序列。

**消融实验**（
![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/883162f2-3a03-471b-b778-96aade7a773e/figures/Figure_4.png)
*Figure 4: Fig. 4: Multi-Robot Failure Trajectory Simulation. MultiWorld simulates realis-tic cooperative failures, such as inter-robot collisions during collaborative manipulation.*

）: 
- 移除Inter-agent attention（智能体间跨视角）: 同一机器人的多摄像头观测出现的像素不一致
- 移除Intra-view attention（视角内跨智能体）: 多机器人协作场景出现碰撞/遮挡处理失败，
- 移除显式通信模块: 长程生成中动作漂移，帧后失控

**公平性检查**: 
- **Baseline强度**: 对比的Sora/Open-Sora为闭源/开源最强视频生成模型，Latte为扩散Transformer代表；未与专用机器人模拟器(如Isaac Sim)对比，因后者非生成式方法。
- **计算成本**: 分解注意力将 $O((VAT)^2)$ 降至 $O(V^2A^2T + VA^2T^2 + V^2AT^2)$ 量级。
- **数据成本**: 需要多视角多智能体同步标注数据，游戏场景易获取，真实机器人场景需。
- **失败案例**: 图4展示了MultiWorld可模拟真实协作失败（机器人间碰撞），但也存在的局限性——极端密集遮挡时视角一致性仍可能退化。


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/883162f2-3a03-471b-b778-96aade7a773e/figures/Figure_5.png)
*Figure 5: Fig. 5: Long-horizon video generation on multi-robot manipulation task. Ourmodel autoregressively simulates three robots stacking colored cubes reasonably, main-taining coherence and action accuracy o*



## 方法谱系与知识库定位

**方法家族**: 扩散Transformer视频生成 → 世界模型(World Models) → 多智能体系统(MAS)

**父方法**: Latte/DiT (扩散Transformer架构) —— MultiWorld继承其latent空间扩散与Transformer backbone，但将2D时空注意力扩展为分解式3D注意力。

**关键改动槽位**:
| 槽位 | 父方法 | 本文改动 |
|:---|:---|:---|
| architecture | 单序列时空Transformer | 多智能体多视角分解注意力 |
| objective | 无条件/全局条件视频扩散 | 每智能体每帧动作条件化 |
| training_recipe | 单视频独立训练 | 智能体-视角维度可扩展的并行训练 |
| data_curation | 单视角视频 | 多视角同步采集+动作标注 |
| inference | 自回归单序列生成 | 联合多视角自回归，显式交互模拟 |

**直接对比方法**:
- **Sora/Open-Sora**: 同为latent扩散视频生成，但Sora无显式动作控制、不支持多智能体；MultiWorld增加动作条件与多视角分解。
- **MVDiffusion/SyncDreamer**: 同为多视角生成，但局限于静态场景；MultiWorld扩展至动态时序与智能体交互。
- **Isaac Sim/Gazebo**: 同为机器人模拟，但基于物理引擎非生成式；MultiWorld是数据驱动的生成模型，可模拟未见过的不确定场景（如故障）。

**后续方向**:
1. **真实世界迁移**: 当前游戏/仿真数据为主，需探索Sim-to-Real的domain adaptation
2. **更复杂的智能体异构性**: 支持不同形态(无人机+机械臂+车辆)的混合智能体系统
3. **闭环决策**: 将MultiWorld作为model-based RL的simulator，实现"生成即规划"

**知识库标签**: 
- modality: video + action
- paradigm: diffusion_transformer, world_model, generative_simulation
- scenario: multi_agent, multi_view, robotics, gaming
- mechanism: factorized_attention, explicit_communication, action_conditioning
- constraint: scalable, long_horizon, physically_plausible

