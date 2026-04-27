---
title: 'WorldMark: A Unified Benchmark Suite for Interactive Video World Models'
type: paper
paper_level: B
venue: NeurIPS
year: 2026
paper_link: https://arxiv.org/abs/2604.21686
aliases:
- WorldMark：交互式视频世界模型的统一评测基准
- WorldMark
- 评测碎片化的根本原因不是缺乏指标
acceptance: accepted
code_url: https://github.com/leofan90/Awesome-World-Models
method: WorldMark
modalities:
- Image
---

# WorldMark: A Unified Benchmark Suite for Interactive Video World Models

[Paper](https://arxiv.org/abs/2604.21686) | [Code](https://github.com/leofan90/Awesome-World-Models)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Video_Generation]], [[T__Embodied_AI]] | **Method**: [[M__WorldMark]] | **Datasets**: WorldMark First-Person Real, WorldMark First-Person Stylized

> [!tip] 核心洞察
> 评测碎片化的根本原因不是缺乏指标，而是缺乏公共测试场地。WorldMark的核心洞察是：将动作接口标准化（统一映射层）与测试条件标准化（相同场景+相同动作序列）分离处理，前者通过适配器模式解决异构控制格式问题，后者通过精心策划的测试套件保证可比性。这一设计使得现有指标在新的标准化条件下立即变得可比较，无需重新发明评测指标本身。有效性来源于工程严谨性而非算法创新。

## 基本信息

**论文标题**: WorldMark: A Unified Benchmark Suite for Interactive Video World Models

**作者**: （未在提供的分析中提取完整作者列表）

**发表场所**: （未在提供的分析中提取完整会议/期刊信息）

**年份**: （未明确提取）

**代码/数据链接**: （未在提供的分析中提取）

**备注**: 本报告基于Agent提取的验证分析结果生成，部分元数据信息未在原始分析中完整提供。

## 核心主张

WorldMark 是首个面向交互式视频世界模型的统一评测基准套件，通过**标准化动作接口**将异构控制模态（WASD文本、姿态参数、游戏手柄、动作向量等）映射为共享的"移动+偏航旋转"词汇表，并建立**三维评估框架**（视觉质量、控制对齐度、世界一致性），实现对现有世界生成模型的公平跨模型比较。

**关键证据**: (1) 50张共享参考图像（25真实+25风格化）配对人称视角；(2) Gemini-3.1-Pro自动选择每图像5条语义一致动作序列；(3) 对YUME 1.5、HY-World 1.5、Matrix-Game 2.0、HY-Game、Open-Oasis、Genie 3等6个模型完成系统评测。

**置信度**: 0.95（核心主张有明确章节支撑）

## 研究动机

现有评测基准存在三大缺陷：

1. **缺乏动作控制**: VBench/VBench++ 仅评估生成质量，无法衡量交互控制能力；
2. **被动路径而非主动交互**: WorldScore 使用被动相机路径，MIND 依赖真值轨迹条件，均非真正的交互式控制；
3. **控制接口不统一**: 各模型使用私有控制格式（WASD字幕、姿态参数、游戏手柄等），导致**无法公平比较**——WorldModelBench 和 4DWorldBench 虽为直接竞争者，但缺乏标准化动作映射层。

**研究重要性**: 交互式世界模型是具身智能、游戏AI、虚拟环境仿真的核心基础设施，缺乏统一评测标准严重阻碍了该领域的技术迭代和社区协作。

## 方法流程

WorldMark 基准流程包含五个核心模块：

```
[参考图像套件] ──→ [VLM动作选择器] ──→ [统一动作接口] ──→ [交互式视频生成] ──→ [三维评估器]
     │                    │                    │                    │                  │
  25真实+25风格化    Gemini-3.1-Pro      移动+偏航旋转         各模型原生推理      质量/对齐/一致性
  配对人称视角        5条动作序列/图像      自定义适配器翻译        (YUME/HY-World等)   多指标输出
```

**模块详情**:
| 模块 | 输入 | 输出 | 创新性 |
|:---|:---|:---|:---|
| Reference Image Suite | 25真实+25风格化图像 | 配对人称参考图像 | **新增** |
| VLM Action Selector | 参考图像 | 5条上下文适配动作序列 | **新增** |
| Unified Action Interface | 标准化动作词汇 | 模型原生控制格式 | **新增，替换**模型特定接口 |
| Interactive Video Generation | 参考图像+动作序列 | 生成视频 | 复用现有模型 |
| Three-Dimensional Evaluator | 生成视频+真值轨迹 | 三维评分 | **新增，替换**单维评估 |

**核心创新**: 统一动作接口层通过自定义适配器，将异构控制模态翻译为共享词汇，首次实现跨模型的"苹果对苹果"比较。

## 关键公式

WorldMark 定义了三项核心评估指标，均为**新增公式**（无基线公式继承）：

### 1. 平移误差（Translation Error）
$$e_t = \|\mathbf{t}_{\text{gt}} - s\mathbf{t}\|_2$$
> 计算真实相机平移向量与预测平移向量的L2欧氏距离，$s$为尺度因子。**功能**: 衡量模型对相机位置控制的精确度，↓越小越好。

### 2. 旋转误差（Rotation Error）
$$e_r = \arccos\left(\frac{\text{tr}(\mathbf{R}_{\text{gt}}\mathbf{R}^T) - 1}{2}\right) \cdot \frac{180}{\pi}$$

**推导步骤**:
- **Step 1** (李群理论): $\mathbf{R}_{\text{gt}}, \mathbf{R} \in SO(3)$，利用旋转矩阵性质得 $\text{tr}(\mathbf{R}_{\text{gt}}\mathbf{R}^T) = 1 + 2\cos(\theta)$
- **Step 2** (反解角度): $\theta = \arccos((\text{tr}(\mathbf{R}_{\text{gt}}\mathbf{R}^T) - 1)/2)$
- **Step 3** (单位转换): $e_r = \theta \cdot 180/\pi$，弧度→角度制

> **功能**: 评估相机朝向控制准确性，↓越小越好。

### 3. 重投影误差（Reprojection Error）
$$e_{\text{reproj}} = \frac{1}{|\mathcal{V}|} \sum_{(i,j)\in\mathcal{V}} \left\|\mathbf{p}^*_{ij} - \Pi(\mathbf{P}_{ij})\right\|_2$$
> 匹配特征点集 $\mathcal{V}$ 中，3D点投影位置 $\Pi(\mathbf{P}_{ij})$ 与真实对应点 $\mathbf{p}^*_{ij}$ 的平均L2距离。**功能**: 检测时序三维结构漂移，评估世界几何一致性，↓越小越好。

**公式新颖性**: 三项均为WorldMark首次引入的基准评估指标，无直接继承公式。

## 实验结果

WorldMark 对 **6个主流模型** 在 **4个测试子集** 上进行系统评估（注：本文是基准论文，非模型论文，故"proposed_score"为N/A）：

### 第一人称真实场景（Table 4，部分数据）
| 模型 | 美学质量↑ | 成像质量↑ |
|:---|:---|:---|
| YUME 1.5 | **56.94** | **74.36** |

### 第一人称风格化场景（Table 5，完整6模型）
| 模型 | 美学质量↑ | 成像质量↑ | 平移误差↓ | 旋转误差↓ | 重投影误差↓ | 状态一致性↑ | 内容一致性↑ | 风格一致性↑ |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **YUME 1.5** | 57.03 | **69.15** | 0.223 | 2.732 | 0.638 | 5.891 | 5.362 | 4.216 |
| **Matrix-Game 2.0** | 47.74 | 64.24 | **0.182** | 1.561 | 0.672 | 2.873 | 6.457 | 4.934 |
| **HY-World 1.5** | **58.50** | 64.78 | 0.244 | 4.316 | 0.638 | 6.408 | **7.159** | 6.817 |
| **HY-Game** | 44.02 | 40.91 | **0.116** | **0.932** | 0.640 | 4.782 | 5.196 | 4.051 |
| **Open-Oasis** | 30.84 | 28.44 | 0.350 | 3.808 | 1.877 | 3.523 | 3.114 | 2.435 |
| **Genie 3** | 46.84 | 53.27 | 0.261 | 2.835 | **0.256** | **6.835** | **7.306** | **7.523** |

**关键发现**:
- **无单一最优模型**: Genie 3在一致性指标领先，但YUME 1.5成像质量最佳，HY-Game控制对齐最优——验证了三维评估的必要性
- **开源vs闭源差距**: Open-Oasis多项指标显著落后，Genie 3（闭源）在一致性维度表现突出

### 第三人称场景（Table 6，仅3模型支持）
| 模型 | 旋转误差↓ | 美学质量↑ | 风格一致性↑（风格化） |
|:---|:---|:---|:---|
| Matrix-Game 2.0 | 27.606 | 52.78 | 2.942 |
| HY-World 1.5 | **2.137** | **57.69** | 7.236 |
| Genie 3 | 14.905 | 51.04 | **8.541** |

**证据强度**: 0.85（数据完整但第三人称仅3模型、50张图像样本量有限、依赖自动化VLM评估）

## 相关工作

WorldMark 与现有基准的关系按角色分类：

### 直接竞争基准（Primary Baselines）
| 基准 | 核心局限 | 与WorldMark关系 |
|:---|:---|:---|
| **VBench** / **VBench++** | 仅评估生成质量，无动作控制 | 设计空间先驱，WorldMark扩展其多维度评估范式 |
| **WorldScore** | 被动相机路径，非交互式 | **谱系父节点**，WorldMark继承"世界生成评测"范式，替换为交互控制 |
| **WorldModelBench** | 最可比竞争者，但缺乏标准化动作接口 | 直接竞品，WorldMark通过统一动作映射实现差异化 |
| **4DWorldBench** | 3D/4D世界生成，相似范围 | 直接竞品，WorldMark聚焦交互式视频而非静态3D/4D生成 |

### 间接相关基准
| 基准 | 核心局限 | 与WorldMark关系 |
|:---|:---|:---|
| **MIND** | 真值轨迹条件，非真正交互控制 | 区分"条件生成"与"交互控制"的概念边界 |

### 关键引用（3-5篇）
1. **VBench/VBench++**: 视频生成评测奠基工作，WorldMark继承其"全面性"理念，新增控制维度
2. **WorldScore**: 谱系最直接来源，WorldMark将"被动路径"→"主动交互"的关键跃迁
3. **WorldModelBench**: 同期最直接可比工作，WorldMark通过标准化接口解决其公平比较问题
4. **Genie 3** (被测模型): 闭源SOTA级世界模型，验证基准区分度

## 方法谱系

WorldMark 在基准演化链中处于**交互式世界模型评测**的关键节点，继承并改造了多条技术 lineage：

```
VBench (2013) ──→ VBench++ ──┐
                              ├──→ 【WorldMark】
WorldScore ───────→ 【被动路径→主动交互】
```

### 谱系父节点：WorldScore（最直接继承）
| 继承槽位 | 父节点值 | WorldMark改造 | 改造类型 |
|:---|:---|:---|:---|
| `data_pipeline` | 被动相机路径，无标准化参考 | 50张共享参考图像+人称配对+VLM动作选择 | **替换** |
| `inference_strategy` | 无统一控制接口 | 统一动作映射层（移动+偏航旋转） | **新增** |
| `objective` | 单维度评估（如3D一致性） | 三维评估框架（质量/对齐/一致性） | **替换** |

### 谱系父节点：VBench/VBench++（范式扩展）
| 继承槽位 | 父节点值 | WorldMark改造 | 改造类型 |
|:---|:---|:---|:---|
| `inference_strategy` | 无动作控制 | 新增统一动作接口 | **新增** |
| `objective` | 生成质量单维度 | 扩展为质量+对齐+一致性三维 | **替换** |

### 核心跃迁
> **从"评视频生成质量"到"评交互世界模型"**：WorldMark 不是渐进改进，而是**评测对象的本质跃迁**——将基准目标从"生成好看的视频"重新定义为"生成可控、一致、可交互的世界仿真"。

## 局限与展望

### 论文明确陈述的局限
（基于分析提取，原文未明确标注"Limitations"章节，但实验设计隐含以下约束）

### 分析推断的局限
1. **样本量限制**: 仅50张参考图像（25真实+25风格化），难以覆盖交互场景的完整多样性
2. **人称覆盖不均**: 第三人称评估仅3/6模型支持（Matrix-Game 2.0、HY-World 1.5、Genie 3），Open-Oasis/YUME/HY-Game缺失
3. **评估依赖自动化VLM**: Gemini-3.1-Pro用于动作选择和部分一致性评估，未与人类判断充分校准
4. **闭源模型不公平性**: Genie 3作为闭源模型，训练规模、数据量可能与开源模型不可比
5. **动作难度不一致**: VLM选择的动作序列可能在不同场景间难度不均
6. **缺失直接数值对比**: WorldModelBench、4DWorldBench、VBench/VBench++的同条件数值对比未在摘录中展示

### 未来方向
- **扩展参考图像规模**至500-5000张，覆盖更多场景类型（室内、室外、动态物体）
- **增加多模态动作控制**（语音指令、物理交互、物体操作）
- **引入人类评估基准**校准自动化VLM指标
- **建立时序扩展评测**（长视频一致性、累积误差分析）
- **开源完整适配器生态**，降低新模型接入门槛

## 知识图谱定位

WorldMark 在知识图谱中连接以下关键节点：

### 任务节点（Task Nodes）
- **交互式视频生成** (interactive video generation) — 核心任务域
- **世界模型评估** (world model evaluation) — 本文开创的细分任务
- **动作控制的图像到视频生成** (image-to-video with action control) — 技术子任务
- **相机控制视频合成** (camera-controlled video synthesis) — 评估子维度

### 方法节点（Method Nodes）
- **WorldMark**（核心方法节点，置信度0.98）
- **统一动作映射层** (unified action mapping layer) — 关键机制创新
- **VLM-based动作序列选择** — 自动化测试生成机制

### 数据集/基准节点（Dataset/Benchmark Nodes）
- **WorldMark First-Person Real/Stylized** — 第一人称评测子集
- **WorldMark Third-Person Real/Stylized** — 第三人称评测子集（覆盖有限）

### 领域结构贡献
WorldMark **填补了"交互式世界模型评测"的知识空白**，在图谱中建立新的**基准方法论分支**：
- **上游连接**: VBench（视频质量评测）→ WorldScore（世界生成评测）
- **下游连接**: 为YUME、HY-World、Matrix-Game、Genie 3等模型提供标准化评估接口
- **横向连接**: 与WorldModelBench、4DWorldBench形成"世界模型评测"竞争生态

**结构意义**: 将原本分散的"视频生成质量""3D一致性""交互控制"三个孤立评估维度，整合为统一的三维坐标系，使该领域从"各自为政"进入"可比较、可迭代"的新阶段。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f5447535-ff98-4f62-b6a7-7783d0410f45/figures/Figure_1.png)
*Figure 1: Figure 1 Overview of WorldMark. Current interactive world models each define their own control interfaces, scenes,and evaluation protocols, making cross-model comparison impossible. WorldMark resolves*


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f5447535-ff98-4f62-b6a7-7783d0410f45/figures/Figure_2.png)
*Figure 2: Figure 2 Overview of the Image Suite. Diverse scenes and styles are covered, each shown in first-person and generatedthird-person views.*


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f5447535-ff98-4f62-b6a7-7783d0410f45/figures/Figure_3.png)
*Figure 3: Figure 3 The 15 standardized action sequences, ranging from elementary translations and rotations to combined andcyclic trajectories.*


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f5447535-ff98-4f62-b6a7-7783d0410f45/figures/Figure_5.png)
*Figure 5: figure 5 illustrates a qualitative comparison demonstrating high and low performance across Visual Quality,Control Alignment, and World Consistency. In the leftmost example, the top video exhibits hig*


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f5447535-ff98-4f62-b6a7-7783d0410f45/figures/Figure_5.png)
*Figure 5: Figure 5 Qualitative comparison across three evaluation dimensions. The top row demonstrates successful videogeneration (✓), while the bottom row illustrates common failures (✗) in Visual Quality, Con*


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f5447535-ff98-4f62-b6a7-7783d0410f45/figures/Figure_6.png)
*Figure 6: Figure 6 Spearman correlation between human preference rankings and automated metric scores across three WorldConsistency dimensions. Each point represents an evaluated model. The high ρ values demons*


