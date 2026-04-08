#!/usr/bin/env python3
"""Batch-merge highly similar tags in paperAnalysis frontmatter.

Rules applied (in order):
1. Exact case-insensitive + hyphen/underscore normalisation merges
2. Semantic duplicates (manually curated)

Canonical form: snake_case (underscores, lowercase first letter of each word
except well-known acronyms like CLIP, SMPL, VQ_VAE, DiT, GCN, etc.)
"""

import os, re, yaml, sys
from pathlib import Path

# ── merge map: old_tag -> canonical_tag ──────────────────────────────
# Every key will be replaced by its value wherever it appears in tags.
MERGE_MAP = {
    # ── case / punctuation variants ──
    # diffusion
    "Diffusion": "diffusion",
    "Diffusion_Model": "diffusion_model",
    "conditional-diffusion": "diffusion_model",
    "conditional_diffusion": "diffusion_model",
    "text-conditioned-diffusion": "diffusion_model",
    "diffusion-transformer": "diffusion_transformer",
    "discrete-diffusion": "diffusion_model",
    # diffusion_model casing
    # autoregressive
    "auto-regressive": "autoregressive",
    "ar": "autoregressive",
    "Autoregressive_Generation": "autoregressive",
    "autoregressive_diffusion": "Autoregressive_Diffusion",
    # VQ_VAE
    "vq-vae": "VQ_VAE",
    "VQ-VAE": "VQ_VAE",
    # benchmark / dataset
    "Benchmark": "benchmark",
    "Dataset": "dataset",
    "dataset_benchmark": "benchmark",
    # text_to_motion
    "text-to-motion": "text_to_motion",
    "text-to-pose": "text_to_pose",
    "Text_Driven": "text_driven",
    # controllable_generation
    "controllable-generation": "controllable_generation",
    "controllable-diffusion": "controllable_generation",
    "Controllable_Generation": "controllable_generation",
    "Controllable_Motion": "controllable_generation",
    "controllable-attributes": "controllable_generation",
    # ControlNet
    "controlnet": "ControlNet",
    # reinforcement_learning
    "reinforcement-learning": "reinforcement_learning",
    "Reinforcement_Learning": "reinforcement_learning",
    "RL": "reinforcement_learning",
    # real_time
    "real-time": "real_time",
    "Real_Time": "real_time",
    "real-time-control": "real_time",
    "Real_Time_Reaction": "real_time",
    # physics_based
    "physics-based": "physics_based",
    "Physics_Based": "physics_based",
    "physics-based-animation": "physics_based",
    "physics_based_animation": "physics_based",
    "physics_based_control": "physics_based",
    "physics_informed": "physics_based",
    "physical_plausibility": "physics_plausibility",
    "Physics_Plausibility": "physics_plausibility",
    # training_free
    "Training_Free": "training_free",
    # zero_shot
    "zero-shot": "zero_shot",
    "Zero_Shot": "zero_shot",
    # few_shot
    "few-shot": "few_shot",
    # contrastive_learning
    "contrastive-learning": "contrastive_learning",
    # co_speech_gesture
    "co-speech-gesture": "co_speech_gesture",
    "Co_speech_Gesture": "co_speech_gesture",
    # audio_driven
    "audio-driven": "audio_driven",
    "Audio_Driven": "audio_driven",
    "audio_to_motion": "audio_driven",
    # dance_generation
    "dance-generation": "dance_generation",
    "Dance_Generation": "dance_generation",
    # music_to_dance / music_to_motion
    "music-driven-generation": "music_to_dance",
    "music_to_motion": "music_to_dance",
    "Music_Driven": "music_to_dance",
    "music_conditioned": "music_to_dance",
    # transformer
    "Transformer": "transformer",
    # GAN
    "gan": "GAN",
    # VAE
    "vae": "VAE",
    "cvae": "VAE",
    "cVAE": "VAE",
    # flow_matching
    "flow-matching": "flow_matching",
    "Flow_Matching": "flow_matching",
    # SMPL_X
    "SMPL-X": "SMPL_X",
    "SMPL-H": "SMPL_H",
    # skeleton_agnostic
    "Skeleton_Agnostic": "skeleton_agnostic",
    # fine_tuning
    "Fine_Tuning": "fine_tuning",
    "finetuning": "fine_tuning",
    # motion_editing
    "Motion_Editing": "motion_editing",
    # multi_task
    "Multi_Task": "multi_task",
    "multitask": "multi_task",
    # unified_model
    "Unified_Model": "unified_model",
    # instruction_tuning
    "Instruction_Tuning": "instruction_tuning",
    # RLHF / preference
    "preference-alignment": "RLHF",
    "preference_learning": "RLHF",
    "Preference_Optimization": "RLHF",
    # DPO is specific enough to keep, but link to RLHF? Keep DPO separate.
    # human_feedback
    "human-feedback": "human_feedback",
    # retrieval_augmented
    "retrieval-augmented-generation": "retrieval_augmented",
    # scene_aware
    "scene-aware": "scene_aware",
    "scene-conditioned-motion": "scene_aware",
    "scene_aware_motion": "scene_aware",
    # human_human_interaction
    "Human_Human_Interaction": "human_human_interaction",
    "human_interaction": "human_human_interaction",
    # human_object_interaction
    "Human_Object_Interaction": "human_object_interaction",
    "hoi": "human_object_interaction",
    "HOI_Generation": "human_object_interaction",
    "HOI_Imitation": "human_object_interaction",
    "HOI_Reconstruction": "human_object_interaction",
    # Human_Scene_Interaction
    "human-scene": "Human_Scene_Interaction",
    # point_cloud
    "point-cloud": "point_cloud",
    "Point_Cloud": "point_cloud",
    # 3D_reconstruction
    "4D_Reconstruction": "3D_reconstruction",
    # articulated_objects
    "articulated-objects": "articulated_objects",
    "Articulated_Object": "articulated_objects",
    # indoor_scenes
    "indoor-scenes": "indoor_scenes",
    # video_generation
    "video-generation": "video_generation",
    "controllable-video-generation": "video_generation",
    # text_to_video
    "text-to-3d": "text_to_3d",
    # image_generation
    "image-generation": "image_generation",
    "human-image-generation": "image_generation",
    # egocentric
    "egocentric-perception": "egocentric",
    "egocentric-pose": "egocentric",
    "Egocentric_Motion": "egocentric",
    # pose_estimation
    "skeleton-pose": "pose_estimation",
    # style_transfer / style_control
    # keep both, they are different
    # multimodal
    "Multimodal": "multimodal",
    "multi-modal": "multimodal",
    "multi_modal": "multimodal",
    "multimodal-LLM": "multimodal_LLM",
    "multimodal-directives": "multimodal",
    # sparse_tracking
    "sparse-tracking": "sparse_tracking",
    "sparse-sensors": "sparse_tracking",
    "sparse-sensing": "sparse_tracking",
    # adversarial
    "adversarial-learning": "adversarial_training",
    # coarse_to_fine
    "coarse-to-fine": "coarse_to_fine",
    # compositional_generation
    "compositional": "compositional_generation",
    "composition": "compositional_generation",
    "complex-text": "compositional_generation",
    # body_part_control
    "body-part-level": "body_part_control",
    "body_part_editing": "body_part_control",
    "part_level_control": "body_part_control",
    # fine_grained_control
    "fine-grained-control": "fine_grained_control",
    "Fine_Grained_Control": "fine_grained_control",
    "fine-grained-actions": "fine_grained_control",
    "fine-grained-semantic": "fine_grained_control",
    "fine-grained-text": "fine_grained_control",
    "Fine_Grained_Text": "fine_grained_control",
    # classifier_free_guidance
    "Classifier_Free_Guidance": "classifier_free_guidance",
    # Efficiency
    "Efficiency": "efficiency",
    "efficiency/lightweight": "efficiency",
    "efficient_generation": "efficiency",
    "efficient_inference": "efficiency",
    # data_augmentation (keep)
    # plug_and_play
    "plug_in": "plug_and_play",
    # Sliding_Window
    "Sliding_Window": "sliding_window",
    # Online_Generation
    "Online_Generation": "online_generation",
    "streaming": "online_generation",
    "streaming_inference": "online_generation",
    "Online_Reaction": "online_generation",
    # Cascaded_Diffusion
    "Cascaded_Diffusion": "cascaded_diffusion",
    "cascaded": "cascaded_diffusion",
    # two_person
    "two-character": "two_person_interaction",
    "two_person_generation": "two_person_interaction",
    "Two_Person_Motion": "two_person_interaction",
    "multi-human": "multi_person",
    "multi-person": "multi_person",
    "Multi_Person_Generation": "multi_person",
    # motion_generation (too generic, keep but normalise)
    "Motion_Generation": "motion_generation",
    # motion_representation
    "motion_representation": "motion_representation",
    "continuous_motion_representation": "motion_representation",
    # motion_tokenizer
    "motion_tokenization": "motion_tokenizer",
    "tokenization": "motion_tokenizer",
    # Temporal_Modeling
    "Temporal_Modeling": "temporal_modeling",
    "spatial_temporal_modeling": "temporal_modeling",
    "spatial_temporal": "temporal_modeling",
    # graph
    "graph-networks": "graph_neural_network",
    "graph-reasoning": "graph_neural_network",
    "graph_convolution": "GCN",
    "graph_topology": "GCN",
    "topology_graph": "GCN",
    # hierarchical
    "hierarchical-generation": "hierarchical",
    "hierarchical_control": "hierarchical",
    # storyboard
    "Storyboard": "storyboard",
    # SDS
    "sds": "SDS",
    "Score_Distillation": "SDS",
    # rectified_flow
    "Rectified_Flow": "rectified_flow",
    # consistency_distillation
    # keep
    # Noise_Optimization
    "noise_optimization": "Noise_Optimization",
    # Optimization_Based
    "optimization_based": "Optimization_Based",
    # Skill_Learning
    "Skill_Learning": "skill_learning",
    # Cross_Attention
    "cross_attention": "Cross_Attention",
    # Dual_Encoding
    # keep
    # Causal_Modeling
    "causal_generation": "Causal_Modeling",
    "causal_latent": "Causal_Modeling",
    # chain_of_thought
    # keep
    # Inference_Time_Optimization
    "test_time_optimization": "Inference_Time_Optimization",
    "test_time_scaling": "Inference_Time_Optimization",
    # weakly_supervised
    "weakly-supervised": "weakly_supervised",
    # self_supervised
    "Self_Supervised_Learning": "self_supervised",
    # teacher_student
    "teacher-student": "teacher_student",
    # representation_learning
    "representation-learning": "representation_learning",
    # Identity_Preservation
    "identity-embedding": "Identity_Preservation",
    "Subject_Consistency": "Identity_Preservation",
    "subject_fidelity": "Identity_Preservation",
    # personalization
    # keep
    # Emotion_Control
    "emotion-control": "Emotion_Control",
    "Dynamic_Emotion": "Emotion_Control",
    # frequency_domain
    "frequency-decomposition": "frequency_domain",
    "Frequency_Decomposition": "frequency_domain",
    # bidirectional
    "bidirectional_generation": "bidirectional_attention",
    # camera_control
    "camera-control": "camera_control",
    # action_to_motion
    "action-conditioned": "action_to_motion",
    # atomic_actions
    "atomic-actions": "atomic_actions",
    # motion_inbetweening
    "generative_inbetweening": "motion_inbetweening",
    "in_betweening": "motion_inbetweening",
    # motion_completion -> motion_inbetweening? They overlap but keep separate
    # Conformer
    "conformer": "Conformer",
    # 3d_scene
    "3d-scene": "3d_scene",
    "3d-scenes": "3d_scene",
    # 3d_pose
    "3d-pose": "3d_pose",
    "global-3d-pose": "3d_pose",
    # 3d_portrait
    "3d-portrait": "3d_portrait",
    # humanoid_control
    "humanoid-control": "humanoid_control",
    "humanoid": "humanoid_control",
    # sim_to_real
    "sim-to-real": "sim_to_real",
    # trajectory_control
    "trajectory-optimization": "trajectory_control",
    "trajectory_conditioning": "trajectory_control",
    "trajectory_guidance": "trajectory_control",
    # grasping
    "dexterous_grasping": "grasping",
    "whole-body-grasp": "grasping",
    # navigation
    # keep
    # instruction_following
    "instruction-following": "instruction_following",
    # open_vocabulary
    "open-vocabulary": "open_vocabulary",
    "open_set": "open_vocabulary",
    # Catastrophic_Forgetting
    # keep
    # apex_free casing
    "Apex_Free": "apex_free",
    # discrete_tokens
    "discrete-tokens": "discrete_tokens",
    "Discrete_Tokens": "discrete_tokens",
    "discrete_motion": "discrete_tokens",
    "discrete_representation": "discrete_tokens",
    # motion_text_retrieval
    "motion_retrieval": "motion_text_retrieval",
    "text-aligned-retrieval": "motion_text_retrieval",
    # motion_understanding
    "motion_language": "motion_understanding",
    "motion_language_alignment": "motion_understanding",
    # GPT
    "gpt": "GPT",
    # BERT
    # keep
    # state_space_model
    "architecture/mamba-ssm": "state_space_model",
    # VLM
    "vision-language": "VLM",
    "vision-language-action": "VLA",
    # Motion_VLM -> VLM? keep separate, it's specific
    # multi_condition
    "multimodal_control": "multi_condition",
    # real_time_generation -> real_time
    "real_time_generation": "real_time",
    # holistic_body -> whole_body_motion
    "holistic_body": "whole_body_motion",
    "full_body": "whole_body_motion",
    "full-body-appearance": "whole_body_motion",
    "full_body_synthesis": "whole_body_motion",
    "whole_body_gesture": "whole_body_motion",
    # hand_motion
    "Hand_Motion": "hand_motion",
    "hand-pose": "hand_motion",
    "hand_pose": "hand_motion",
    "hand_gesture": "hand_motion",
    "hand-pressure": "hand_motion",
    # pose_control
    "pose-control": "pose_control",
    "pose-guidance": "pose_control",
    "pose-guided-t2i": "pose_control",
    "pose-to-video": "pose_control",
    # spatial_control
    # keep
    # diffusion_GAN -> GAN
    "diffusion_GAN": "GAN",
    "dd-gan": "GAN",
    # data_scaling
    "scaling": "data_scaling",
    # Contact_Modeling
    "contact_modeling": "Contact_Modeling",
    "contact-aware-loss": "Contact_Modeling",
    "contact-constraints": "Contact_Modeling",
    "contact-estimation": "Contact_Modeling",
    "contact-fields": "Contact_Modeling",
    "Contact_Annotation": "Contact_Modeling",
    "contact_consistency": "Contact_Modeling",
    "contact_estimation": "Contact_Modeling",
    "Contact_Graph": "Contact_Modeling",
    "Contact_Guidance": "Contact_Modeling",
    "Contact_Loss": "Contact_Modeling",
    "contact_map": "Contact_Modeling",
    "Contact_Scheduling": "Contact_Modeling",
    # interaction
    "interaction-field": "interaction_modeling",
    "interaction-modeling": "interaction_modeling",
    "interaction-prior": "interaction_modeling",
    "interaction-synthesis": "interaction_modeling",
    "Interaction_Generation": "interaction_modeling",
    "Interaction_Loss": "interaction_modeling",
    "Interaction_Order": "interaction_modeling",
    "interaction_prediction": "interaction_modeling",
    "interactive": "interaction_modeling",
    # Masked_Modeling
    "masked_language_modeling": "Masked_Modeling",
    # Mixture_of_Experts
    "Multi_Expert_Architecture": "Mixture_of_Experts",
    # Long_Term_Generation
    "arbitrary_length": "Long_Term_Generation",
    # motion_prior_distillation -> distillation
    "motion_prior_distillation": "distillation",
    "Flow_Distillation": "distillation",
    # generative_prior -> diffusion_model? no, keep
    # world_model -> keep
    # foundation_model -> keep
    # single_stage / two_stage -> keep
    # Tencent_Hunyuan -> keep
    # example_project_tag -> keep (project-specific tag)
    # GRPO -> keep (specific RL method)
    # DPO -> keep
    # RLHF -> keep
    # DiT -> keep
    # FSQ -> keep
    # DINOv2 -> keep
    # RWKV -> keep
    # DCT -> keep
    # LLM_annotation
    "LLM_decomposition": "LLM_annotation",
    # motion_grounding
    "temporal_grounding": "motion_grounding",
    "grounding": "motion_grounding",
    "motion_localization": "motion_grounding",
    # test_time_training -> Inference_Time_Optimization
    "test_time_training": "Inference_Time_Optimization",
    # soft_masking -> keep (specific to ZOMG)
    # BABEL_Grounding -> keep (specific dataset)
    # text_controlled_selection -> keep (specific to TM-Mamba)
    # temporal_segmentation -> motion_grounding
    "temporal_segmentation": "motion_grounding",
    # LLM_decomposition already mapped above
    # open_vocabulary already mapped
    # Masked_Modeling
    "Masked_Modeling": "masked_modeling",
    "masked_language_modeling": "masked_modeling",
    # Mixture_of_Experts
    "Mixture_of_Experts": "mixture_of_experts",
    # Long_Term_Generation
    "Long_Term_Generation": "long_term_generation",
    # Noise_Optimization already mapped
    # Optimization_Based already mapped
    # Skill_Learning already mapped
    # Cross_Attention already mapped -> keep as cross_attention
    "Cross_Attention": "cross_attention",
    # Causal_Modeling already mapped -> keep as causal_modeling
    "Causal_Modeling": "causal_modeling",
    # Inference_Time_Optimization -> inference_time_optimization
    "Inference_Time_Optimization": "inference_time_optimization",
    # Emotion_Control -> emotion_control
    "Emotion_Control": "emotion_control",
    # Contact_Modeling -> contact_modeling (already have contact_modeling -> Contact_Modeling, fix)
    # Let me fix the chain: everything -> contact_modeling
    "Contact_Modeling": "contact_modeling",
    # SDS -> keep uppercase
    # GAN -> keep uppercase
    # GPT -> keep uppercase
    # RLHF -> keep uppercase
    # DPO -> keep uppercase
    # GRPO -> keep uppercase
    # VQ_VAE -> keep uppercase
    # ControlNet -> keep
    # SMPL -> keep
    # SMPL_X -> keep
    # SMPL_H -> keep
    # CLIP -> keep
    # DiT -> keep
    # FSQ -> keep
    # GCN -> keep
    # BERT -> keep
    # DINOv2 -> keep
    # RWKV -> keep
    # DCT -> keep
    # Conformer -> keep
    # Autoregressive_Diffusion -> keep
    # Action_Reaction -> keep
    # Action_Unit / Action_Unit_Detection
    "Action_Unit_Detection": "Action_Unit",
    # Head_Centric_Representation -> keep
    # Personality_Conditioning -> keep
    # Waypoint_Conditioning -> keep
    # Start_End_Frame_Pair -> keep
    # Pseudo_Last_Frame -> keep
    # Consistent_Self_Attention -> keep
    # Decoupled_Cross_Attention -> keep
    # Semantic_Motion_Predictor -> keep
    # Component_Process_Model -> keep
    # Dual_System_Cognition -> keep
    # Intention_Features -> keep
    # Motion_Pre_Training -> keep
    # Reward_Model -> keep
    # MotionCritic -> keep
    # MotionFix -> keep
    # MotionNFT -> keep
    # MotionPercept -> keep
    # motionverse -> keep
    # ICAdapt -> keep
    # PULSE -> keep
    # GORS -> keep
    # TMR -> keep
    # ImageBind -> keep
    # BLIP2_inspired -> keep
    # BLIP_VQA -> keep
    # Gemma2 -> keep
    # Qwen2_5_VL -> keep
    # Qwen3 -> keep
    # EDM2 -> keep
    # UniDiffuser -> keep
    # 3DMM -> keep
    # HuBERT -> keep
    # wav2vec -> keep
    # RQVAE -> keep
    # RVQ_VAE -> keep
    # Focal_Loss -> keep
    # AdaIN -> keep
    # AdaLN -> keep
    # PEFT -> keep
    # DSTformer -> keep
    # BiLSTM -> keep
    # CTMC -> keep
    # EMG -> keep
    # semg -> keep
    # imu -> keep
    # HMD -> keep
    # NPR
    "npr": "NPR",
}

# ── resolve transitive chains ────────────────────────────────────────
def resolve(m):
    changed = True
    while changed:
        changed = False
        for k in list(m):
            v = m[k]
            if v in m and m[v] != v:
                m[k] = m[v]
                changed = True
    return m

MERGE_MAP = resolve(MERGE_MAP)

# ── apply ─────────────────────────────────────────────────────────────
def process_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    m = re.match(r'^(---\s*\n)(.*?\n)(---)', content, re.DOTALL)
    if not m:
        return False
    pre, fm_str, post = m.group(1), m.group(2), m.group(3)
    try:
        fm = yaml.safe_load(fm_str)
    except:
        return False
    if not fm or 'tags' not in fm or not isinstance(fm['tags'], list):
        return False

    old_tags = fm['tags']
    new_tags = []
    seen = set()
    changed = False
    for t in old_tags:
        t_str = str(t)
        canonical = MERGE_MAP.get(t_str, t_str)
        if canonical != t_str:
            changed = True
        if canonical not in seen:
            seen.add(canonical)
            new_tags.append(canonical)

    if not changed:
        return False

    # Rebuild the tags block in the frontmatter
    # Find the tags section and replace it
    tag_pattern = re.compile(r'(tags:\s*\n)((?:\s+-\s+.*\n)*)', re.MULTILINE)
    tag_match = tag_pattern.search(fm_str)
    if not tag_match:
        return False

    new_tag_block = "tags:\n"
    for t in new_tags:
        new_tag_block += f"  - {t}\n"

    new_fm_str = fm_str[:tag_match.start()] + new_tag_block + fm_str[tag_match.end():]
    new_content = pre + new_fm_str + post + content[m.end():]

    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return True


def main():
    root = Path('paperAnalysis')
    modified = 0
    total = 0
    for md in root.rglob('*.md'):
        total += 1
        if process_file(str(md)):
            modified += 1
            print(f"  [MOD] {md}")
    print(f"\nDone: {modified}/{total} files modified.")

if __name__ == '__main__':
    main()
