---
title: 'Relaxing partition admissibility in Cluster-DAGs: a causal calculus with arbitrary variable clustering'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 循环C-DAG因果演算：放松分区可接受性
- Cyclic C-DAG Cau
- Cyclic C-DAG Causal Calculus
- By relaxing partition admissibility
acceptance: Poster
cited_by: 1
method: Cyclic C-DAG Causal Calculus
modalities:
- Text
- graph
paradigm: theoretical/analytical
---

# Relaxing partition admissibility in Cluster-DAGs: a causal calculus with arbitrary variable clustering

**Topics**: [[T__Reasoning]] | **Method**: [[M__Cyclic_C-DAG_Causal_Calculus]]

> [!tip] 核心洞察
> By relaxing partition admissibility, cyclic C-DAGs can be defined with extended d-separation and causal calculus that is sound and atomically complete with respect to Pearl's do-calculus, enabling valid interventional inference at the cluster level for arbitrary variable clusterings.

| 中文题名 | 循环C-DAG因果演算：放松分区可接受性 |
| 英文题名 | Relaxing partition admissibility in Cluster-DAGs: a causal calculus with arbitrary variable clustering |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2511.01396) · [DOI](https://doi.org/10.48550/arxiv.2511.01396) |
| 主要任务 | Causal reasoning / Causal effect identification |
| 主要 baseline | Anand et al. (2023) C-DAG framework; Pearl's do-calculus |

> [!abstract] 因为「现实世界中聚类层面的循环关系自然存在（如宏观经济部门、脑区之间的反馈回路），但现有C-DAG框架禁止此类循环聚类」，作者在「Anand et al. (2023) 的C-DAG框架」基础上改了「移除partition admissibility约束，引入unfolding构造和广义d-分离」，在「理论层面」上取得「首个针对循环C-DAG的sound且atomically complete的因果演算」

- **理论完整性**: 该演算相对于Pearl's do-calculus具有soundness和atomic completeness
- **约束放松**: 允许任意变量聚类，包括自环和聚类间循环，无需partition admissibility
- **核心工具**: 通过unfolding构造将循环C-DAG映射到无限无环结构进行分析

## 背景与动机

在因果推断中，研究者常常需要在变量聚类（cluster）层面进行推理，而非单个变量层面。例如，宏观经济学家可能关注"金融部门"对"实体经济部门"的因果影响，神经科学家可能研究"前额叶皮层"与"海马体"之间的信息流动。这些场景天然涉及循环关系——金融部门影响实体经济，实体经济反过来也影响金融部门——但现有的Cluster-DAG（C-DAG）框架却禁止这种循环聚类。

Anand et al. (2023) 提出的C-DAG框架要求**partition admissibility**：变量聚类必须诱导出无环的C-DAG。这一约束通过限制合法聚类的集合，排除了大量实际应用中自然出现的聚类方式。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d19efa4c-e395-489a-8652-4a39b3b98706/figures/Figure_1.png)
*Figure 1 (example): Left: a DAG G^all = (V, E). Right: a graph G^C = (V^C, E^C) compatible with G^all, with a partition σ = {σ_X, σ_Y, σ_Z} of V.*



现有处理方法存在明显局限：
- **Anand et al. (2023) 的C-DAG框架**：直接禁止循环聚类，将问题排除在框架之外；
- **Jaber et al. (2022) 的abstraction-specific criteria**：针对特定抽象场景设计分离准则，缺乏普适性；
- **Pearl's do-calculus**：作为底层基础，但仅适用于无环因果图，无法直接处理聚类层面的循环抽象。

这些方法的共同短板在于：**当聚类本身具有循环结构时，缺乏系统的语义定义和推理工具**。本文通过引入循环C-DAG及其配套的广义d-分离与因果演算，首次将C-DAG框架扩展到任意变量聚类，包括含自环和循环的聚类结构。

## 核心创新

核心洞察：循环C-DAG可以通过**unfolding到无限无环结构**来定义其语义，因为无限展开保留了原始循环结构的所有路径信息，从而使在聚类层面直接进行因果推理成为可能。

| 维度 | Baseline (Anand et al. 2023) | 本文 |
|:---|:---|:---|
| 聚类约束 | Partition admissibility：聚类必须诱导无环C-DAG | 任意聚类允许，包括循环和自环 |
| 分离准则 | 标准d-分离（基于无环图的路径阻断） | 广义d-分离（基于unfolding的结构分析） |
| 因果演算 | 无专门演算；依赖转换图或特定规则 | 直接在循环C-DAG上操作的完整演算 |
| 完备性 | 无循环情形下的identification | 相对于do-calculus的sound且atomically complete |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d19efa4c-e395-489a-8652-4a39b3b98706/figures/Figure_2.png)
*Figure 2 (example): Three examples of C-DAGs (a–c) and their corresponding full and essential ancestral graphs (d–f) illustrating m-separation.*



整体流程从底层微观因果图出发，经过六个核心模块：

1. **底层无环因果图**（Underlying acyclic causal graph）：输入为变量集合 $V$ 上的标准DAG $\mathcal{G}^{all} = (V, E)$，表示微观层面的因果结构。

2. **任意变量聚类**（Arbitrary clustering）：对 $V$ 进行任意划分 $\sigma = \{\sigma_X, \sigma_Y, \sigma_Z, ...\}$，**不施加无环性约束**。每个聚类 $\sigma_i$ 成为C-DAG的一个节点。

3. **循环C-DAG构建与兼容性检验**（Cyclic C-DAG with compatibility）：基于聚类构造 $\mathcal{G}^C = (V^C, E^C)$，其中 $V^C$ 为聚类集合，边 $E^C$ 反映聚类间的依赖关系。通过**兼容性条件** $\mathcal{C} \text{models} \mathcal{G}$ 验证该循环C-DAG是否对应于某个底层无环图的合法抽象。

4. **Unfolding构造**（Unfolding to infinite structure）：将循环C-DAG $\mathcal{G}_{\mathcal{C}}$ 展开为无限无环结构 $\text{Unfold}(\mathcal{G}_{\mathcal{C}})$，这是分析循环语义的核心理论工具。

5. **广义d-分离检验**（Generalized d-separation）：在循环C-DAG上判定条件独立性，通过unfolding结构上的标准d-分离等价实现。

6. **循环C-DAG演算应用**（Cyclic C-DAG calculus）：直接在循环C-DAG上执行干预识别，将干预查询转化为可计算的因果效应表达式。

```
底层DAG G^all ──→ 任意聚类 σ ──→ 循环C-DAG G^C_σ 
                                      │
                                      ▼
                              Unfold(G^C_σ) [无限无环结构]
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            广义d-分离检验                    循环C-DAG演算
            (条件独立性判定)                  (干预识别与因果效应推导)
```

## 核心模块与公式推导

### 模块 1: 兼容性条件与循环C-DAG定义（对应框架图 步骤1-3）

**直觉**: 循环C-DAG必须对应某个底层无环因果图的合法抽象，不能凭空构造聚类间的依赖关系。

**Baseline 公式** (Anand et al. 2023):
$$\mathcal{C} \text{models} \mathcal{G} \quad \text{and} \quad \mathcal{G}_{\mathcal{C}} \text{ is acyclic} \quad \text{(partition admissibility)}$$
符号: $\mathcal{C}$ = 变量聚类, $\mathcal{G}$ = 底层无环因果图, $\mathcal{G}_{\mathcal{C}}$ = 聚类后的C-DAG

**变化点**: Baseline要求$\mathcal{G}_{\mathcal{C}}$无环，这排除了大量自然聚类。本文移除无环性约束，仅保留语义兼容性。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{C} \text{models} \mathcal{G} \iff \forall C_i, C_j \in \mathcal{C}, \exists \text{ compatible edge structure in } \mathcal{G}_{\mathcal{C}} \quad \text{(保留底层依赖关系的语义对应)}$$
$$\text{Step 2}: \text{Remove acyclicity constraint on } \mathcal{G}_{\mathcal{C}} \quad \text{(允许 } \sigma_X \rightarrow \sigma_Y \rightarrow \sigma_Z \rightarrow \sigma_X \text{ 等循环)}$$
$$\text{最终}: \mathcal{G}^C = (V^C, E^C) \text{ where } E^C \text{ may contain cycles and self-loops, subject only to } \mathcal{C} \text{models} \mathcal{G}$$

---

### 模块 2: Unfolding构造与广义d-分离（对应框架图 步骤4-5）

**直觉**: 无限展开将循环"拉直"为无限深的树状结构，使标准d-分离可在展开后的结构上应用，从而间接定义循环结构上的分离语义。

**Baseline 公式** (标准d-分离, Pearl 2009):
$$(X \perp\!\!\!\perp Y \text{mid} Z)_{\mathcal{G}} \quad \text{for acyclic } \mathcal{G}$$
符号: $\perp\!\!\!\perp$ = d-分离, $\mathcal{G}$ = 无环因果图

**变化点**: 标准d-分离在无环图上通过路径阻断判定，但循环结构中存在无限回溯路径，传统阻断规则失效。本文通过unfolding将循环结构"展开"为无限无环结构。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{G}^* = \text{Unfold}(\mathcal{G}_{\mathcal{C}}) \quad \text{(将循环C-DAG展开为无限无环结构)}$$
$$\text{Step 2}: X \perp_{\mathcal{G}_{\mathcal{C}}} Y \text{mid} Z \iff X \perp_{\mathcal{G}^*} Y \text{mid} Z \quad \text{(循环图上的分离 = 展开图上的标准d-分离)}$$
$$\text{最终}: (X \perp_{\mathcal{G}_{\mathcal{C}}} Y \text{mid} Z) \iff (X \perp_{\text{Unfold}(\mathcal{G}_{\mathcal{C}})} Y \text{mid} Z)$$

该广义d-分离基于**结构分析**而非单纯路径阻断：unfolding保留了原始循环结构的所有可能遍历路径，使得条件独立性判定能够考虑循环依赖的完整语义。

---

### 模块 3: 循环C-DAG因果演算（对应框架图 步骤6）

**直觉**: 将Pearl's do-calculus的三个基本规则适配到循环C-DAG设置，用广义d-分离替代标准d-分离作为规则触发条件，实现直接在聚类抽象上的干预识别。

**Baseline 公式** (Pearl's Rule 2):
$$P(y \text{mid} do(x), z, w) = P(y \text{mid} x, z, w) \quad \text{if } (Y \perp\!\!\!\perp X \text{mid} Z, W)_{\mathcal{G}_{\overline{X}}}$$
符号: $do(x)$ = 对$X$的干预, $\mathcal{G}_{\overline{X}}$ = 移除指向$X$的边后的图

**变化点**: 原规则要求在无环图$\mathcal{G}_{\overline{X}}$上判定d-分离。本文将条件替换为循环C-DAG上的广义d-分离，并证明每条规则对应do-calculus的一个primitive step。

**本文公式（推导）**:
$$\text{Step 1}: \text{Define } \mathcal{G}^*_{\mathcal{C}, \overline{X}} = \text{Unfold}(\mathcal{G}_{\mathcal{C}, \overline{X}}) \quad \text{(对干预后的循环C-DAG进行unfolding)}$$
$$\text{Step 2}: P(v \text{mid} do(x)) = \sum_{z} P(v \text{mid} x, z) P(z) \quad \text{if } (V \perp_{\mathcal{G}_{\mathcal{C}, \overline{X}}} X \text{mid} Z) \text{ via unfolding} \quad \text{(Rule 2 analog)}$$
$$\text{最终}: \text{Cyclic C-DAG calculus rules } \mathcal{R}_1, \mathcal{R}_2, \mathcal{R}_3 \text{ with generalized d-separation conditions, atomically complete w.r.t. do-calculus}$$

**对应消融**: 本文无实验消融；理论证明表明若移除unfolding构造（回归标准d-分离），则循环结构上的分离判定将不完整，导致演算unsound。

## 实验与分析

本文为一项**纯理论工作**，未包含任何实证实验或数值评估。作者在实验检查清单中明确标注"N/A"，未提供数据集、训练流程或性能指标。



尽管如此，论文的核心贡献可通过理论性质来评估：

**理论结果概述**: 本文的主要"结果"是三个形式化定理：（1）循环C-DAG兼容性定义的合理性；（2）广义d-分离的soundness和completeness；（3）循环C-DAG演算相对于do-calculus的soundness和**atomic completeness**。其中atomic completeness意味着演算中的每条规则对应do-calculus的一个primitive step，这是比单纯completeness更强的性质，保证了推理步骤的粒度最优性。



**缺失的分析与局限**:
- **计算复杂性**: Unfolding构造产生无限结构，其在大型循环C-DAG上的实际计算可行性未分析；
- **自动化算法**: 未开发用于自动识别循环C-DAG上因果效应的实用算法；
- **实证验证**: 未在宏观经济或神经科学等宣称的应用场景中进行案例验证；
- **与最强baseline的比较**: 由于无实验，未与近期相关工作（如[6][7]关于summary causal graphs的identifiability结果）进行定量比较。

**公平性检查**: 本文定位为理论奠基工作，其baseline选择（Anand et al. 2023; Pearl's do-calculus; Jaber et al. 2022）覆盖了C-DAG框架、标准因果演算和抽象特定准则三个方向，选择合理。但缺少与同期工作[6][7]在summary graph identifiability方面的直接理论对比。

## 方法谱系与知识库定位

**方法族**: Cyclic C-DAG Causal Calculus ← **父方法**: Anand et al. (2023) C-DAG framework

**变更插槽**:
| 插槽 | 变更类型 | 说明 |
|:---|:---|:---|
| partition_admissibility_constraint | 移除 | 取消聚类必须诱导无环C-DAG的限制 |
| d_separation_criterion | 替换 | 标准d-分离 → 基于unfolding的广义d-分离 |
| causal_calculus | 新增 | 首个直接在循环C-DAG上操作的完整因果演算 |

**直接baseline对比**:
- **Anand et al. (2023)**: 本文直接扩展其C-DAG框架，移除partition admissibility约束，保留兼容性语义；
- **Pearl's do-calculus**: 本文演算在循环C-DAG上达到与其等价的推理能力（atomic completeness），但操作层面为聚类抽象而非原始变量；
- **Jaber et al. (2022)**: 提供calculus-based identification的方法论启发，但本文针对循环C-DAG的特定结构设计了全新的unfolding机制。

**后续方向**:
1. **计算可行性**: 开发避免显式无限unfolding的有限算法，或识别可高效处理的循环C-DAG子类；
2. **实证应用**: 在宏观经济模型（部门间循环依赖）或神经科学（脑区间反馈回路）中验证框架的实际效用；
3. **自动化工具**: 构建从数据到循环C-DAG的自动发现与identification系统，衔接因果发现（causal discovery）与因果效应识别。

**标签**: modality=graph | paradigm=theoretical/analytical | scenario=causal reasoning with variable clustering | mechanism=unfolding construction, generalized d-separation | constraint=relaxed acyclicity at cluster level

