---
title: AFL
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Federated Learning
---

# AFL

AFL（Analytic Federated Learning）是一种针对预训练模型设计的联邦学习方法，其核心创新在于仅需单轮通信即可完成联邦聚合，彻底避免了传统联邦学习中多轮迭代通信带来的高昂开销。该方法在CVPR 2025上提出，特别适用于大规模预训练模型在隐私敏感场景下的分布式部署。

传统联邦学习（如FedAvg）需要数百至数千轮的客户端-服务器通信，在跨机构协作中面临网络带宽、同步延迟和隐私泄露累积等多重挑战。AFL通过解析式（analytic）的聚合策略，利用预训练模型的特性，从理论上推导出最优全局模型的闭式解，实现"一次上传、永久聚合"的高效范式。

**研究领域**: Federated Learning

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__单轮解析式联邦学习预训练模型适配_AFL_(Analytic_Fe]] | CVPR | 2025 |

