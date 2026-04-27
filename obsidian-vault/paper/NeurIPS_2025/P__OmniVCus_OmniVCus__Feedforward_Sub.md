---
title: 'OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- OmniVCus
- OmniVCus achieves feedforward multi
acceptance: Poster
cited_by: 12
code_url: https://caiyuanhao1998.github.io/project/OmniVCus/
method: OmniVCus
modalities:
- Image
- Video
- Text
paradigm: supervised
baselines:
- 大语言模型的贝叶斯低秩适应_Laplace-LoRA
---

# OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions

[Code](https://caiyuanhao1998.github.io/project/OmniVCus/)

**Topics**: [[T__Video_Generation]], [[T__Image_Editing]] | **Method**: [[M__OmniVCus]]

> [!tip] 核心洞察
> OmniVCus achieves feedforward multi-subject video customization with multimodal control through a novel data construction pipeline (VideoCus-Factory), Image-Video Transfer Mixed training, and two embedding mechanisms (Lottery Embedding and Temporally Aligned Embedding).


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_001.jpeg)
*Figure: (a) and (c) show that our method can change the pose and action of the subject. (b) The instructive*


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_002.jpeg)
*Figure: Our method can flexibly compose different conditions to control multi-subject video customization. 2*


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_003.jpeg)
*Figure: VideoCus-Factory can*


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_004.jpeg)
*Figure: OmniVCus is DiT architecture that can compose different input signals to customize a video. (a) LE*


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_005.jpeg)
*Figure: Visual comparison of single-subject video customization with state-of-the-art algorithms. Our method*


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_006.jpeg)
*Figure: Visual comparison of instructive editing for subject-driven video customization with SOTA methods.*


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_007.jpeg)
*Figure: In a nutshell, our contributions can be summarized as: (i) We design a data construction pipeline, VideoCus-Factory, to produce training data pairs and*


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_008.jpeg)
*Figure: Comparison of cameral-controlled subject-driven video customization. Motionctrl, Cameractrl, and*


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_009.jpeg)
*Figure: Visual analysis. (a) Using the data augmentation in our VideoCus-Factory can vary the scale, pose,*


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_010.jpeg)
*Figure: Comparison of cameral-controlled subject-driven video customization. Motionctrl, Cameractrl, and*


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_011.jpeg)
*Figure: Visual analysis. (a) Using the data augmentation in our VideoCus-Factory can vary the scale, pose,*


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_012.jpeg)


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_013.jpeg)


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_014.jpeg)


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_015.jpeg)


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_016.jpeg)


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_017.jpeg)


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_018.jpeg)


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_019.jpeg)


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e34c657a-cd10-4a27-94ee-4027629b9ab4/figures/fig_020.jpeg)

## 引用网络

### 直接 baseline（本文基于）

- [[P__大语言模型的贝叶斯低秩适应_Laplace-LoRA]] _(方法来源)_: LoRA is the standard parameter-efficient fine-tuning method; essential for subje

