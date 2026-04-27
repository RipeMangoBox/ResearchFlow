---
title: 'ARIA: Training Language Agents with Intention-driven Reward Aggregation'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- ARIA
- ARIA (Aggregating Rewards in Intent
acceptance: Spotlight
cited_by: 5
code_url: https://aria-agent.github.io/
method: ARIA
modalities:
- Text
paradigm: Reinforcement Learning
---

# ARIA: Training Language Agents with Intention-driven Reward Aggregation

[Code](https://aria-agent.github.io/)

**Topics**: [[T__Reasoning]], [[T__Agent]] | **Method**: [[M__ARIA]]

> [!tip] 核心洞察
> ARIA (Aggregating Rewards in Intention space) projects natural language actions from high-dimensional joint token distributions into a low-dimensional intention space where semantically similar actions are clustered and share rewards, significantly reducing reward variance and improving policy optimization.


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f2e2475b-80f1-43f7-88d3-5cb670e6c5c3/figures/Figure_1.png)
*Figure 1 (pipeline): Illustration of ARIA. ARIA first lets agents interact to collect trajectories. Then it performs intention-aware reward aggregation based on the intention space and updates the policy using the aggregated rewards.*


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f2e2475b-80f1-43f7-88d3-5cb670e6c5c3/figures/Figure_2.png)
*Figure 2 (comparison): Diversity selection results comparing Best-of-N, Greedy, and ARIA on balancing diversity and performance.*


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f2e2475b-80f1-43f7-88d3-5cb670e6c5c3/figures/Table_2.png)
*Table 2 (quantitative): Main results on single-agent games.*


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f2e2475b-80f1-43f7-88d3-5cb670e6c5c3/figures/Figure_3.png)
*Figure 3 (result): (a) and (b) show the reward curves of ARIA and other online methods over iterations on the Twenty Questions and State My City respectively.*


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f2e2475b-80f1-43f7-88d3-5cb670e6c5c3/figures/Figure_4.png)
*Figure 4 (result): (a) Illustrates the distribution of rewards. (b) portrays the change in reward variance.*


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f2e2475b-80f1-43f7-88d3-5cb670e6c5c3/figures/Figure_5.png)
*Figure 5 (ablation): Ablation of ARIA. (a) shows win rates on referential games and (b) shows training loss curves with or without (non-smoothed).*


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f2e2475b-80f1-43f7-88d3-5cb670e6c5c3/figures/Table_3.png)
*Table 3 (quantitative): ARIA on Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct.*

