---
title: 'MAPLE: Multi-scale Attribute-enhanced Prompt Learning for Few-shot Whole Slide Image Classification'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- MAPLE
- MAPLE achieves superior few-shot WS
acceptance: Poster
cited_by: 2
method: MAPLE
modalities:
- Image
- Text
paradigm: supervised
---

# MAPLE: Multi-scale Attribute-enhanced Prompt Learning for Few-shot Whole Slide Image Classification

**Topics**: [[T__Few-Shot_Learning]], [[T__Classification]] | **Method**: [[M__MAPLE]] | **Datasets**: TCGA-BRCA 4-shot, TCGA-RCC 4-shot, TCGA-NSCLC 4-shot, TCGA-BRCA 16-shot, TCGA-RCC 16-shot

> [!tip] 核心洞察
> MAPLE achieves superior few-shot WSI classification by hierarchically integrating multi-scale visual semantics through LLM-generated entity-level and slide-level prompts, with entity-guided cross-attention and cross-scale entity graph learning for fine-grained entity-level prediction.


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5716d13c-e70f-43ed-822d-462b675a625c/figures/Figure_1.png)
*Figure 1 (pipeline): Comparison of MAPLE with existing slide-level alignment methods for the classification of lung adenocarcinoma (LUAD) and lung squamous cell carcinoma (LUSC).*


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5716d13c-e70f-43ed-822d-462b675a625c/figures/Figure_2.png)
*Figure 2 (architecture): Overview of our proposed MAPLE. MAPLE leverages an LLM to identify multi-scale fine-grained pathological attributes from pathology reports to establish a pathology attribute bank.*


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5716d13c-e70f-43ed-822d-462b675a625c/figures/Table_1.png)
*Table 1 (quantitative): Few-shot WSI classification results on TCGA-BRCA, TCGA-RCC and TCGA-NSCLC datasets under the 16-shot setting.*


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5716d13c-e70f-43ed-822d-462b675a625c/figures/Table_2.png)
*Table 2 (ablation): Effect of different heads and entity levels under the 16-shot setting.*


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5716d13c-e70f-43ed-822d-462b675a625c/figures/Figure_3.png)
*Figure 3 (qualitative): (a-e) t-SNE results of patch embeddings via CLIP image encoder under the 16-shot setting on the TCGA-BRCA dataset.*


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5716d13c-e70f-43ed-822d-462b675a625c/figures/Figure_4.png)
*Figure 4 (qualitative): Visualization of entity-relevant patches selected by the entity-guided cross-attention module for lung adenocarcinoma subtyping.*

## 引用网络

### 直接 baseline（本文基于）

- FOCUS: Knowledge-enhanced Adaptive Visual Compression for Few-shot Whole Slide Image Classification _(CVPR 2025, 直接 baseline, 未深度分析)_: Few-shot WSI classification with knowledge enhancement; directly same task and p
- Dynamic Graph Representation with Knowledge-aware Attention for Histopathology Whole Slide Image Analysis _(CVPR 2024, 实验对比, 未深度分析)_: Graph-based approach with knowledge-aware attention for WSI; related prior metho

