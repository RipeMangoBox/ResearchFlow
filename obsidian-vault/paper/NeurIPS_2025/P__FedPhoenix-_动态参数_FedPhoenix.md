---
title: 'Rising from Ashes: Generalized Federated Learning via Dynamic Parameter Reset'
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 'FedPhoenix: 动态参数重置的广义联邦学习'
- FedPhoenix
- FedPhoenix improves generalization
acceptance: Poster
method: FedPhoenix
modalities:
- Image
paradigm:
- federated
- supervised
---

# Rising from Ashes: Generalized Federated Learning via Dynamic Parameter Reset

**Topics**: [[T__Federated_Learning]] | **Method**: [[M__FedPhoenix]] | **Datasets**: [[D__CIFAR-10]], [[D__CIFAR-100]], [[D__Tiny-ImageNet]] (其他: Office-31)

> [!tip] 核心洞察
> FedPhoenix improves generalization in Federated Learning by stochastically resetting partial parameters each round to destroy overfitting features and guide learning toward multiple generalized features.

| 中文题名 | FedPhoenix: 动态参数重置的广义联邦学习 |
| 英文题名 | Rising from Ashes: Generalized Federated Learning via Dynamic Parameter Reset |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.xxxxx) · [Code](待补充) · [Project](待补充) |
| 主要任务 | Federated Learning（联邦学习），解决非独立同分布（non-IID）客户端数据导致的泛化性能下降 |
| 主要 baseline | FedAvg（primary）、FedProx、SCAFFOLD、FedMut、ClusteredSampling、FedGen、Federated deconfounding and debiasing learning |

> [!abstract] 因为「联邦学习中客户端数据异质性导致全局模型过拟合客户端特定特征、陷入局部最优」，作者在「FedAvg 均匀聚合」基础上改了「服务器端动态参数重置机制，以统计量选择参数并阶段性衰减扰动比例」，在「CIFAR-10」上取得「相比 SOTA FL 方法最高提升 20.73% 准确率」

- **CIFAR-10**: 相比 SOTA FL 方法准确率最高提升 **20.73%**（Table 8）
- **服务器开销**: 每轮额外计算量仅 **M(6 + 10θK) FLOPs**，ResNet-18 在 NVIDIA A100 上耗时 **< 0.1 秒**
- **核心机制**: 完全服务器端执行，零客户端计算开销

## 背景与动机

联邦学习（Federated Learning, FL）允许多个客户端在不共享原始数据的前提下协同训练模型。然而，实际场景中客户端数据往往高度异质（non-IID）——例如不同用户的手机照片风格迥异、不同医院的病患分布不同。这种异质性导致每个客户端的本地模型过拟合自身数据的特有模式，而服务器端的聚合操作（如 FedAvg 的简单加权平均）将这些过拟合特征粗暴混合，使全局模型陷入局部最优，泛化性能急剧下降。

现有方法如何应对这一问题？**FedAvg** [1] 作为奠基算法，对所有客户端参数进行无差别均匀平均，完全忽略参数的重要性差异；**FedProx** 在客户端本地目标中加入近端项以约束局部更新偏离全局模型，但仍在客户端侧增加计算负担；**SCAFFOLD** [17] 引入控制变量（control variates）修正客户端漂移，通过额外的状态向量传递来稳定收敛，但通信和存储开销显著增加；**Federated deconfounding and debiasing learning** [19] 尝试从因果推断角度去除混淆偏差以提升分布外泛化，但依赖于特定的数据生成假设。

这些方法的共同局限在于：**它们都试图“保留”或“修正”所有参数，而从未主动“摧毁”已学习的特征**。当客户端数据异质性极强时，模型某些维度早已编码了客户端特定的伪相关（spurious correlation），继续微调只会加深这种过拟合。更关键的是，现有探索策略完全依赖客户端异质性带来的隐式扰动，缺乏显式的机制跳出局部最优。

本文的核心动机正是：与其被动地平均所有参数，不如主动识别并重置那些最可能编码客户端特定过拟合特征的参数，通过“破坏-重建”的循环引导模型发现更具泛化性的特征表示，并随训练阶段动态调整破坏强度以平衡探索与收敛。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5efde232-313b-41a3-a6b6-6837c119990c/figures/fig_001.png)
*Figure: Accuracy and loss curves of FedAvg and FedPhoenix with α = 0.6*



## 核心创新

**核心洞察**：服务器端可以通过统计量识别跨客户端方差高的参数维度（这些维度最可能编码客户端特定过拟合特征），通过随机重置这些参数主动摧毁过拟合特征，从而迫使模型探索更具泛化性的解空间；同时通过阶段性衰减扰动比例，早期激进探索、后期稳定收敛，因为参数重置强度若不加衰减将导致训练无法收敛。

| 维度 | Baseline (FedAvg) | 本文 (FedPhoenix) |
|:---|:---|:---|
| 参数聚合方式 | 对所有参数无差别均匀加权平均 | 基于客户端更新统计量选择部分参数进行随机重置 |
| 探索策略 | 无显式探索，依赖客户端异质性的隐式扰动 | 主动摧毁已学特征以跳出局部最优，显式引导发现泛化特征 |
| 训练调度 | 固定轮次聚合，无阶段性变化 | 阶段性衰减扰动比例 θ：早期大扰动探索，后期小扰动微调 |
| 计算位置 | 客户端额外计算（FedProx/SCAFFOLD）或零额外计算（FedAvg） | 完全服务器端执行，零客户端开销 |

## 整体框架



FedPhoenix 完全在服务器端改造聚合流程，保留标准 FL 的客户端本地训练，数据流如下：

1. **Client Local Training（客户端本地训练）**：每轮开始时，服务器将当前全局模型分发给选中的 K 个客户端；各客户端在本地数据上执行标准 SGD，得到本地更新后的模型参数，回传模型更新量（非原始数据）。

2. **Server Statistics Computation（服务器统计计算）**：服务器收集所有客户端的更新量 {Δw^k}_{k=1}^K，逐维度计算统计量（如方差），识别哪些参数维度在跨客户端间变化剧烈——这些维度最可能编码了客户端特定的过拟合特征。

3. **Parameter Selection for Reset（参数选择）**：基于统计量阈值或排序，选定待重置参数集合 S；选择同时受当前扰动比例 θ_t 控制，决定重置范围大小。

4. **Stochastic Parameter Reset（随机参数重置）**：对选中参数执行随机重置（如按特定分布重新采样），生成扰动后的全局模型；未被选中的参数保持原值或经标准平均处理。

5. **Stage-wise Decay Update（阶段衰减更新）**：根据当前训练阶段更新 θ_{t+1} = θ_t · γ（或类似衰减函数），逐步降低扰动强度，使训练从探索期过渡到收敛微调期。

```
客户端本地训练 → 上传更新量 → 服务器统计计算 → 参数选择(S, θ_t) → 随机重置 → 阶段衰减θ → 下发新全局模型
         ↑___________________________________________________________________________________________↓
```

整个流程的关键在于：重置操作完全发生在服务器，客户端无需任何修改；统计计算和参数采样的额外开销为 M(6 + 10θK) FLOPs，在 ResNet-18 规模下可忽略不计。

## 核心模块与公式推导

### 模块 1: 统计参数选择准则（对应框架图步骤 2-3）

**直觉**: 跨客户端方差高的参数维度更可能编码了客户端特定的伪相关，应优先被重置以消除过拟合。

**Baseline 公式** (FedAvg): 无参数选择，所有参数参与均匀平均
$$\mathcal{S}_{\text{FedAvg}} = \emptyset$$

**变化点**: FedAvg 不区分参数重要性，导致客户端特定特征被保留并累积；本文引入基于统计量的显式选择机制。

**本文公式（推导）**:
$$\text{Step 1: 计算逐维度统计量} \quad \text{Var}_i = \frac{1}{K}\sum_{k=1}^{K}(\Delta w_i^k - \bar{\Delta w_i})^2$$
其中 $\Delta w_i^k$ 表示第 k 个客户端在第 i 个参数维度上的更新量，$\bar{\Delta w_i}$ 为跨客户端均值；方差衡量该维度上的客户端分歧程度。

$$\text{Step 2: 阈值筛选} \quad \mathcal{S} = \{i \text{mid} \text{Stat}(\{\Delta w_i^k\}_{k=1}^K) > \tau\}$$
选择统计量超过阈值 τ 的参数维度加入重置集合；也可采用 Top-r 排序策略选取固定比例。

$$\text{Step 3: 扰动比例约束} \quad |\mathcal{S}| \approx \theta_t \cdot M$$
实际重置数量受当前扰动比例 θ_t 限制，确保重置强度随训练阶段可控。

**对应消融**: Table 12 显示不同采样分布（均匀、高斯等）对选择后重置效果的影响。

---

### 模块 2: 阶段性扰动衰减调度（对应框架图步骤 5）

**直觉**: 训练早期模型尚未稳定，激进重置有助于跳出初始局部最优；后期需减少扰动以保证收敛到平坦区域。

**Baseline 公式** (FedAvg): 无扰动，无阶段性变化
$$\theta_t^{\text{FedAvg}} \equiv 0, \quad \forall t$$

**变化点**: FedAvg 缺乏探索-利用权衡机制；本文引入显式的阶段性衰减策略，将训练划分为多个阶段，每阶段内 θ 不同。

**本文公式（推导）**:
$$\text{Step 1: 阶段划分} \quad t \in \text{Stage}_s, \quad s = 1, 2, \ldots, S$$
将总训练轮次划分为 S 个阶段（如早期、中期、后期）。

$$\text{Step 2: 阶段内固定扰动} \quad \theta_t = \theta_s, \quad \forall t \in \text{Stage}_s$$
同一阶段内使用固定扰动比例，保证该阶段内探索强度一致。

$$\text{Step 3: 跨阶段衰减} \quad \theta_{s+1} = \theta_s \cdot \gamma, \quad \gamma \in (0, 1)$$
阶段间按衰减系数 γ 递减，如 Step Decay 或 Cosine Annealing 变体。

$$\text{最终}: \theta_t = f(\text{stage}(t)) = \theta_1 \cdot \gamma^{\text{stage}(t)-1}$$

**对应消融**: 虽未明确给出具体数值，但文中指出去掉阶段性衰减将导致训练不稳定或收敛失败。

---

### 模块 3: 服务器端重置计算开销（对应框架图全流程）

**直觉**: 证明服务器端重置的实用性，需量化额外计算并证明其可忽略。

**Baseline 公式** (FedAvg): 仅参数平均，无额外开销
$$C_{\text{FedAvg}}^{\text{server}} = O(M)$$

**变化点**: FedPhoenix 增加了统计计算和随机采样，需证明总开销仍极低。

**本文公式（推导）**:
$$\text{Step 1: 统计计算开销} \quad C_{\text{stat}} = 6M \quad \text{（均值、方差等统计量计算）}$$

$$\text{Step 2: 参数采样开销} \quad C_{\text{sample}} = 10\theta K M \quad \text{（对选中参数的重置采样）}$$
其中 K 为每轮参与客户端数，θ 为扰动比例，系数 10 来自具体采样操作的 FLOPs 估算。

$$\text{最终}: C_{\text{FedPhoenix}}^{\text{server}} = M(6 + 10\theta K)$$

**对应消融**: Table 5 显示该开销与 FedProx（客户端近端项）、FedMut（服务器变异）、ClusteredSampling（成对相似度）、FedGen（生成器蒸馏）等方法相比具有竞争力，ResNet-18 在 NVIDIA A100 上每轮 < 0.1 秒。

## 实验与分析



本文在 CIFAR-10、CIFAR-100、Tiny-ImageNet 和 Office-31 四个基准上验证 FedPhoenix 的有效性。核心结果如 Table 8（CIFAR-10）所示，FedPhoenix 在异质性设置（Dirichlet 分布 α=0.6）下相比 SOTA FL 方法取得最高 **20.73%** 的准确率提升。在 CIFAR-100（Table 3）和 Tiny-ImageNet（Table 6, 7）上，FedPhoenix 同样一致优于 FedAvg、FedProx 和 SCAFFOLD 等基线。Office-31（Table 9）上的实验进一步验证了跨域泛化场景下的优势。



从 Figure 1 的准确率与损失曲线可见，FedPhoenix（α=0.6）相比 FedAvg 展现出更稳定的上升轨迹和更低的最终损失，说明动态重置有效缓解了客户端过拟合导致的震荡。关键差距在于：FedAvg 在训练后期准确率 plateau 明显，而 FedPhoenix 通过阶段性重置持续发现更优解。



消融实验揭示了各组件的必要性。Table 10 对比了重置不同目标层的效果：**仅重置卷积层** 比仅重置全连接层或全部重置更有效，说明底层特征编码的客户端特定伪相关是主要问题。Table 11 验证了 **部分重置优于全模型重置**，全重置破坏过多已学通用特征导致性能骤降。Table 12 比较了不同采样分布（均匀、高斯等）对重置效果的影响，指导实际部署选择。

公平性考量：基线选择涵盖了经典方法（FedAvg）、优化方法（FedProx、SCAFFOLD）和专用泛化方法（Federated deconfounding），但缺少 MOON、FedDyn、pFedMe、APFL 等较新基线。实验仅限 CNN 架构（ResNet-18）和视觉任务，NLP/Transformer 适用性未验证。作者明确披露：收敛分析仅覆盖全参与场景，部分参与的理论分析尚不完整；且阶段衰减最终 θ→0 时可能退化为 FedAvg 行为。

## 方法谱系与知识库定位

**方法谱系**: FedPhoenix 属于 **FedAvg → 服务器端聚合改造**  lineage，直接父方法为 **FedAvg** [1]。核心改动 slot：credit_assignment（均匀平均 → 统计量驱动的选择性重置）、exploration_strategy（无显式探索 → 主动特征摧毁）、training_recipe（固定轮次 → 阶段性扰动衰减）。

**直接基线差异**:
- **FedAvg**: 无差别平均所有参数 vs 本文选择性重置部分参数
- **FedProx**: 客户端侧近端约束 vs 本文完全服务器端执行、零客户端开销
- **SCAFFOLD** [17]: 控制变量修正漂移 vs 本文通过重置跳出局部最优而非修正方向
- **Federated deconfounding** [19]: 因果去混淆假设驱动 vs 本文统计驱动、无需特定数据生成假设
- **FedMut**: 服务器端变异操作 vs 本文基于统计量的定向重置而非随机变异

**后续方向**: (1) 扩展至 Transformer 架构和 NLP 任务，验证参数重置对注意力头的适用性；(2) 部分参与场景下的收敛理论补全；(3) 自适应统计量选择（非预设方差）以进一步提升重置精度。

**标签**: modality=image / paradigm=supervised federated learning / scenario=heterogeneous client data (non-IID) / mechanism=dynamic parameter reset with statistical selection / constraint=server-only computation, zero client overhead

