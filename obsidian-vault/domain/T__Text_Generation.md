---
title: Text Generation
type: task
domain: task
name_zh: 文本生成
---

# Text Generation (文本生成)

文本生成是自然语言处理的核心生成式任务，旨在让机器自动产出符合语法规范、语义连贯且满足特定需求的文本内容，涵盖从句子到长文档的多粒度生成。

该任务随着预训练语言模型（如GPT系列）的发展取得突破性进展，已从受限领域的模板填充发展到开放域的创意写作。

## 代表方法

- MedConclusion (1 篇)
- Conformal Importance Summarization (1 篇)
- Abstain-R1 (1 篇)
- SWIFT (1 篇)
- EVOREFUSE (1 篇)
- GFT (1 篇)
- HAX (1 篇)
- FACT (1 篇)

## 常用数据集

- Ruler (4 篇)
- Multiple datasets (2 篇)
- DROP (2 篇)
- MRQA (2 篇)
- Computational Overhead (2 篇)
- Multi-bit Scaling (2 篇)
- Synthetic DNA sequences (2 篇)
- WG-S (2 篇)

## 分布

- 年份: 2024 (2) · 2025 (10) · 2026 (15)
- 会议: NeurIPS (10) · arXiv (7) · ACL (2) · TMLR (1) · ICML (1) · 其他 (1)

## 相关论文 (27)

| 论文 | 会议 | 年份 | 核心贡献 |
|------|------|------|----------|
| [[P__MedConclusion：生物_MedConclusion]] | arXiv | 2026 | — |
| [[P__异构任务下LLM自进化记忆提取_SLMEAH]] | arXiv | 2026 | — |
| [[P__共形预测保障重要内容保留的文档摘_Conformal_Import]] | NeurIPS | 2025 | Conformal Importance Summarization is the first framework to |
| [[P__ABSTAIN-R1：基于可验证_Abstain-R1]] | ACL | 2026 | — |
| [[P__Token级自博弈与重要性感知蒸_SWIFT]] | NeurIPS | 2025 | SWIFT improves self-play fine-tuning by assigning token-leve |
| [[P__进化提示优化评估与缓解LLM过度_EVOREFUSE]] | NeurIPS | 2025 | EVOREFUSE, an evolutionary prompt optimization algorithm, ge |
| [[P__单轮多策略情感支持对话生成_MMSSST]] | Unknown | 2026 | — |
| [[P__后训练阶段输出多样性崩溃机制分析_Where_does_outpu]] | Unknown | 2026 | — |
| [[P__微语言模型实现即时响应_MLMEIR]] | Unknown | 2026 | — |
| [[P__网页感知检索分块W-RAC_WRACWE]] | arXiv | 2026 | W-RAC 的核心直觉是：LLM 做语义分块决策时，真正需要的是「哪些内容在语义上相邻」的结构信号，而非原始文本本身。网 |
| [[P__GFT：无偏组优势与动态系数修正_GFT]] | arXiv | 2026 | — |
| [[P__HAX：SSM上下文依赖稀疏注意_HAX_(locality-se]] | NeurIPS | 2025 | Integrating SSMs with Context-Dependent Sparse Attention (CD |
| [[P__代码-文本交替训练抑制LLM幻觉_FACT]] | NeurIPS | 2025 | FACT is the first task-agnostic paradigm that alternates bet |
| [[P__长视频音视频联合脚本生成Omni_OmniScript]] | arXiv | 2026 | 现有视频理解模型将叙事内容压缩为粗粒度段落摘要，导致时间锚定丢失和多模态线索混杂。OmniScript的核心洞察是：将输 |
| [[P__多语言PEFT自适应语义采样持续_COMPASS]] | TMLR | 2026 | 核心直觉是：多语言负迁移的根源不在于语言之间的语言学距离，而在于训练数据与目标使用分布之间的语义覆盖缺口。通过在共享嵌入 |
| [[P__LLM位置脆弱性：偏移效应重塑记_Positional_Offse]] | NeurIPS | 2025 | Memorization in LLMs exhibits positional fragility: verbatim |
| [[P__MoE专家上循环：低成本扩容新范_EUSCEF]] | Unknown | 2026 | — |
| [[P__低秩分解与MoE协同的提示微调框_PT-MoE]] | NeurIPS | 2025 | PT-MoE achieves state-of-the-art PEFT performance by combini |
| [[P__基于稀疏自编码器的黑盒LLM水印_SAEMark]] | NeurIPS | 2025 | SAEMARK enables scalable, quality-preserving, personalized m |
| [[P__离散扩散的信息论精确似然估计_Information-Theo]] | NeurIPS | 2025 | The I-MDSE and I-MDCE relations provide tight, principled es |
| [[P__UDM-GRPO：面向均匀离散扩_UDM-GRPO]] | arXiv | 2026 | UDM-GRPO的核心直觉是：**让RL训练的状态-动作分布尽可能贴近预训练时的分布**。动作重定义（$\hat{x}_ |
| [[P__面向目标指令微调的影响力数据选择_LESS]] | ICML | 2024 | LESS (Low-rank gradiEnt Similarity Search) enables selecting |
| [[P__可执行视觉工作流生成的评测基准_Chat2Workflow]] | Unknown | 2026 | 现有LLM评测体系以「任务完成率」为核心，忽视了工业部署中工作流的结构规范性与可执行性约束。Chat2Workflow的 |
| [[P__基于Trace_Rewritin_PLMAUD]] | arXiv | 2026 | — |
| [[P__图条件扩散模型的关系数据库联合生_GRDM_(Graph-Cond]] | NeurIPS | 2025 | Jointly modeling all tables in a relational database without |
| [[P__大语言模型的贝叶斯低秩适应_Laplace-LoRA]] | ICLR | 2024 | — |
| [[P__基于KV共享的LLM无缝早退推理_River-LLM]] | ACL | 2026 | — |
