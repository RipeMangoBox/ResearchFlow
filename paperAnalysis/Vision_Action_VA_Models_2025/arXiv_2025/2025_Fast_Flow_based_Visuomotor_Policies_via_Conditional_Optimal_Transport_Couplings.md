---
title: "Fast Flow-based Visuomotor Policies via Conditional Optimal Transport Couplings"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/visuomotor-control
  - task/imitation-learning
  - flow-matching
  - optimal-transport
  - dataset/MimicGen
  - dataset/Push-T
  - dataset/Maze2D
  - opensource/no
core_operator: "在训练时用条件最优传输把噪声与相似观测下的动作轨迹配对，学习更直的条件流，从而支持 1–2 步的多模态机器人动作生成。"
primary_logic: |
  观测历史与专家动作 → 用 PCA+K-means 将连续观测离散成条件簇，并基于“动作距离 + 条件距离”执行条件 OT 配对 → 在原始观测条件下训练 flow matching 向量场沿更直路径从噪声生成动作 → 以极少积分步输出短时域动作序列
claims:
  - "在 6 个仿真控制任务上，2-step COT Policy 的平均成功率/得分为 0.818，高于 2-step CFM 的 0.790、Adaflow 的 0.783，以及 20-step Diffusion Policy 的 0.781 [evidence: comparison]"
  - "在条件合成分布 moons 与 fork 上，忽略条件的 OT-CFM 会产生有偏样本，而 COT-CFM 用 1–2 个 Euler 步即可获得比 CFM/OT-CFM 更低的 2-Wasserstein 距离 [evidence: comparison]"
  - "COT Policy 通过观测离散化近似连续条件 OT，在保持与 CFM/DP 相近训练收敛速度的同时，提高了低 NFE 下的轨迹多样性与真实机器人任务完成效率 [evidence: analysis]"
related_work_position:
  extends: "Improving and Generalizing Flow-based Generative Models with Minibatch Optimal Transport (Tong et al. 2024)"
  competes_with: "Diffusion Policy (Chi et al. 2023); Adaflow (Hu et al. 2024)"
  complementary_to: "Consistency Policy (Prasad et al. 2024); One-step Diffusion Policy (Wang et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Fast_Flow_based_Visuomotor_Policies_via_Conditional_Optimal_Transport_Couplings.pdf
category: Embodied_AI
---

# Fast Flow-based Visuomotor Policies via Conditional Optimal Transport Couplings

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.01179), [Project](https://ansocho.github.io/cot-policy)
> - **Summary**: 论文把 flow matching 中“噪声—动作”的无条件 OT 配对改成“受观测条件约束的条件 OT 配对”，并用 PCA+K-means 近似连续视觉条件，从而让机器人策略在 1–2 步推理下仍保持高质量、多模态动作生成。
> - **Key Performance**: 6 个仿真任务中 2-step 平均 0.818，优于 2-step CFM 的 0.790 和 20-step Diffusion Policy 的 0.781；相对 20-step DP 约用 10× 更少 NFE

> [!info] **Agent Summary**
> - **task_path**: RGB 图像+本体状态历史 / 离线示范模仿学习 -> 短时域动作序列
> - **bottleneck**: few-step ODE 采样下既要保持多模态又要实时控制，而无条件 OT 会在条件任务中造成错误配对与有偏流场
> - **mechanism_delta**: 将观测条件纳入 minibatch OT 代价，并先用 PCA+K-means 把连续观测量化成可配对的离散条件，再训练条件 flow
> - **evidence_signal**: 2-step COT 在 6 个仿真任务平均 0.818，超过 CFM/DP/Adaflow，且真实机器人三任务中成功率更高、完成更快
> - **reusable_ops**: [conditional minibatch OT pairing, PCA+K-means observation discretization]
> - **failure_modes**: [K 过小会退化成 OT-CFM 式偏置配对, 高维动作或极低 NFE 下仍存在 low-NFE 与 high-NFE 性能差距]
> - **open_questions**: [如何自适应选择 K 与 gamma, 能否与单阶段蒸馏结合实现稳定 1-step 控制]

## Part I：问题与挑战

**What/Why**：这篇论文真正要解决的，不是“生成模型能不能拟合动作分布”，而是**在实时控制预算下，能否用 1–2 次网络评估就生成不塌缩、不过偏的条件动作**。

### 1) 问题接口
- **输入**：观测历史 \(o\)，包含 RGB 图像和 proprioception。
- **输出**：短时域动作序列 \(a\)。
- **训练设置**：离线模仿学习，样本来自专家示范。

### 2) 现有方法为什么不够
- **Diffusion / Flow Matching 策略**很擅长拟合多峰动作分布，但推理要数值积分 ODE/SDE。
- 对机器人来说，哪怕 **20-step** 采样也会明显拉低控制频率，造成“算一段、停一下、再算下一段”的**间歇式动作**。
- 蒸馏能把多步压成一步，但通常需要教师模型、额外训练阶段或更贵的训练流程。

### 3) 真正瓶颈在哪
作者识别了两个耦合在一起的瓶颈：

1. **路径曲率瓶颈**  
   普通 CFM 用独立配对训练，学到的流往往更弯，few-step 数值积分误差大。

2. **条件错配瓶颈**  
   无条件 OT-CFM 虽然能让路径更直，但在**条件生成**里，如果 OT 配对时忽略观测条件，就会把不同观测下的动作模式错误耦合。  
   结果是：训练时流场监督变“直”了，但**条件上变偏了**。

### 4) 边界条件
- 机器人观测是**高维连续变量**，几乎每个样本的 condition 都不同，无法直接按 condition 做 minibatch conditional OT。
- 方法默认**相近观测对应相近局部动作分布**，否则离散化条件就会伤害配对质量。
- 目标是**短时域 visuomotor policy**，不是长时程规划或在线 RL。

## Part II：方法与洞察

**How**：作者引入的关键因果旋钮，是**改变 flow matching 的配对分布**——从无条件的 \(q(x_0, x_1)\) 改成近似条件的 \(q(x_0, x_1 \mid c)\)。

### 方法流程
1. 从数据中取一批 `(动作 a, 观测 o)`，再采样一批高斯噪声 `x0`。
2. 对观测做条件压缩：  
   - 图像经 **PCA** 降维  
   - proprio 保留  
   - 再用 **K-means** 得到离散条件簇 `c = Q(E(o))`
3. 给噪声样本分配一个**随机置换后的条件** `c0`，保证噪声侧和动作侧条件边缘分布一致。
4. 在 OT 里不是只看 `x0` 与 `a` 的距离，而是看：

   **代价 = 动作距离 + γ × 条件距离**

   这样得到的是**条件感知的 minibatch OT 配对**。
5. 用这些配对后的 `(x0, a)` 训练条件 flow；**真正喂给策略网络的仍是原始观测 `o`**，离散条件只服务于配对，不替代感知输入。

### 核心直觉

- **改了什么**：把训练监督里最关键的“谁和谁配对”，从只看样本几何，改成同时看**样本几何 + 条件相似性**。
- **改变了哪种瓶颈**：  
  - 约束了监督信号的支持集，避免跨条件错误搬运；  
  - 同时保留 OT 对“拉直流轨迹”的好处。
- **能力为什么变强**：  
  在 flow matching 里，配对关系直接决定目标速度方向。  
  如果配错了，模型学到的是**错误方向的条件速度场**；  
  如果既直又条件一致，few-step 时就不容易：
  - 塌成 conditional mean
  - 跑到条件不对应的动作模态
  - 因为曲线太弯而需要很多积分步

一个很重要的结构性洞察是：

- **K 太小**：所有观测几乎被压成同一类，COT 会退化到 **OT-CFM**
- **K 太大**：每个样本几乎都是独立 condition，COT 又会退化到 **I-CFM**

也就是说，COT 的收益正来自于这两种极端之间的**中间离散粒度**。

### 为什么这套设计有效
- **无条件 OT 失败的根因**不是 OT 本身，而是它忽略了条件变量，导致“直但偏”。
- **条件离散化**解决的是可计算性问题：连续高维 observation 没法直接做 conditional minibatch OT。
- **原始观测仍用于条件化网络**，解决的是信息损失问题：量化只影响配对，不强迫策略只看粗糙簇标签。

### 战略权衡

| 设计选择 | 改变的约束/分布 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| OT-CFM → COT pairing | 配对必须尊重观测条件 | low-NFE 下更少偏置，路径更直 | 需要选择 `γ` 与条件表征 |
| 连续观测 → PCA + K-means | 让“每个样本都是独特 condition”变成可重复的 condition bucket | 条件 OT 可在视觉策略上落地 | `K` 敏感，过粗会偏、过细会退化回 CFM |
| Euler 1-step → Midpoint 2-step | 降低数值积分误差 | few-step 成功率更稳 | 延迟略增，动态任务仍可能出现停顿 |
| COT vs 蒸馏 | 不引入教师/二阶段训练 | 保持接近原始 FM/DP 的训练复杂度 | 还不是严格 one-step 解法 |

## Part III：证据与局限

**So what**：能力跃迁不只是“分数略高”，而是**2-step 的 flow policy 已经能超过更慢的 20-step Diffusion Policy，并在真实机器人上明显减少卡顿和完工时间**。

### 关键实验信号

1. **主结果：few-step 性能直接超基线**  
   在 6 个仿真任务上，**2-step COT Policy 平均 0.818**，高于：
   - 2-step CFM：0.790
   - 2-step Adaflow：0.783
   - 20-step Diffusion Policy：0.781  
   这说明它的核心增益不是“多跑几步更好”，而是**同等甚至更低推理预算下更强**。

2. **机制验证：合成条件分布清楚展示“直 vs 偏”的区别**  
   在 moons / fork 条件分布上：
   - CFM：不偏，但路径弯，1-step 很差
   - OT-CFM：路径直，但条件上有偏
   - COT-CFM：同时避免两者问题  
   这是对论文核心因果链最直接的支撑。

3. **多模态验证：低 NFE 下仍保留动作多样性**  
   作者提出 Trajectory Variance 指标；在 maze、coffee 等任务中，COT 的 TV 整体高于 CFM，表明 few-step 下没有简单塌成单一路径。

4. **真实机器人：更少停顿、更快完成**  
   在 push-T、cup-stacking、cup-in-drawer 三个真实任务中，COT 相比 CFM 在 1-step Euler / 2-step Midpoint 下通常都有更高 SR 和更低 TTC。  
   例如 real push-T 的 Euler 1-step：**COT 0.8 vs CFM 0.2**，且完成时间明显更短。

5. **训练成本信号：没有额外蒸馏阶段**  
   论文报告其训练收敛速度与 CFM/DP 近似，PCA、K-means 和 minibatch OT 的额外开销在 GPU 上相对反向传播可忽略。  
   这使它比蒸馏型加速策略更“便宜地快起来”。

### 局限性
- **Fails when**: `K` 选择不合适时会退化——过小接近 OT-CFM 并引入条件偏置，过大接近 CFM 而失去直线路径优势；在高维动作或极低 NFE 下，low-NFE 与 high-NFE 仍存在明显性能差距；对动态且碰撞敏感的任务，solver 选择仍会影响表现。
- **Assumes**: 需要有覆盖较好的离线专家示范，且局部观测相似性能够映射到局部动作分布相似性；依赖 GPU 加速的 PCA/K-means/OT 与较大 U-Net 策略网络，论文训练环境使用 2×RTX 4090，真实机器人推理使用 RTX 2080。
- **Not designed for**: 真正的单步蒸馏控制、长时程规划/信用分配、或大幅 OOD 视觉泛化；真实机器人评测规模也较小（3 个任务、每任务 5 次 rollout），因此硬件泛化结论要保守看待。

### 可复用组件
- **conditional minibatch OT pairing**：可直接迁移到其他 conditional flow/diffusion policy。
- **PCA + K-means 条件离散化**：一种不额外训练辅助网络的连续条件 OT 近似。
- **Trajectory Variance (TV)**：适合评估多步动作轨迹的多模态保真度。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Fast_Flow_based_Visuomotor_Policies_via_Conditional_Optimal_Transport_Couplings.pdf]]