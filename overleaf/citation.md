# Gaze Estimation Papers

[1] Detecting people looking at each other in videos (2014)

[2] Where are they looking? (2015)

[3] Following gaze in video (2017)

[4] Connecting gaze, scene, and attention: Generalized attention estimation via joint modeling of gaze and scene saliency. (2018)
[5] Inferring shared attention in social scene videos. (2018)
[6] Human gaze following for human-robot interaction. (2018)
[7] Where and why are they looking? jointly inferring human attention and intentions in complex tasks. (2018)
[8] Recurrent CNN for 3d gaze estimation using appearance and shape cues. (2018)

[9] Believe it or not, we know what you are looking at! (2019)
[10] Understanding human gaze communication by spatio-temporal graph reasoning. (2019)
[11] Enhanced gaze following via object detection and human pose estimation. (2019)
[12] Gaze360: Physically unconstrained gaze estimation in the wild. (2019)
[13] Laeo-net: revisiting people looking at each other in videos. (2019)

[14] Detecting attended visual targets in video. (2020)
[15] Attention flow: End-to-end joint attention estimation. (2020)

[16] Dual Attention Guided Gaze Target Detection in the Wild (2021)
[17] Boosting image-based mutual gaze detection using pseudo 3d gaze. (2021)
[18] LAEO-Net++: revisiting people Looking At Each Other in videos. (2021)
[19] Goo: A dataset for gaze object prediction in retail environments. (2021)
[20] Multi-person gaze-following with numerical coordinate regression. (2021)

[21] Escnet: Gaze target detection with the understanding of 3d scenes. (2022)
[22] Gaze estimation via the joint modeling of multiple cues. (2022)
[23] A modular multimodal architecture for gaze target prediction: Application to privacy-sensitive settings. (2022)
[24] Gaze target estimation inspired by interactive attention. (2022)
[25] Depth-aware gaze-following via auxiliary networks for robotics. (2022)
[26] End-to-end human-gaze-target detection with transformers. (2022)
[27] Multimodal across domains gaze target detection. (2022)
[28] Gatector: A unified framework for gaze object prediction. (2022)
[29] We know where they are looking at from the rgb-d camera: Gaze following in 3d. (2022)
[30] Dynamic 3D Gaze from Afar: Deep Gaze Estimation from Temporal Eye-Head-Body Coordination (2022)
[31] HHP-Net: A light Heteroscedastic neural network for Head Pose estimation with uncertainty (2022)
[32] MGTR: End-to-end mutual gaze detection with transformer. (2022)
[33] Gazeonce: Real-time multi-person gaze estimation (2022)

[34] Temporal understanding of gaze communication with gazetransformer. (2023)
[35] Where are they looking in the 3d space? (2023)
[36] Automated detection of joint attention and mutual gaze in free play parent-child interactions. (2023)
[37] Patch-level gaze distribution prediction for gaze following (2023)
[38] Interaction-aware Joint Attention Estimation Using People Attributes (2023)
[39] Childplay: A new benchmark for understanding children’s gaze behaviour. (2023)
[40] Object-aware gaze target detection. (2023)
[41] Joint gaze-location and gaze-object detection. (2023)
[42] Gaze pattern recognition in dyadic communication. (2023)

[43] Vitgaze: gaze following with interaction features in vision transformers. (2024)
[44] Sharingan: A transformer architecture for multi-person gaze following. (2024)
[45] Toward semantic gaze target detection (2024)
[46] Gaze target detection based on head-local-global coordination. (2024)
[47] Gaze Target Detection by Merging Human Attention and Activity Cues (2024)
[48] Diffusion-refined vqa annotations for semi-supervised gaze following. (2024)
[49] MTGS: A Novel Framework for Multi-Person Temporal Gaze Following and Social Gaze Prediction (2024)
[50] A unified model for gaze following and social gaze prediction (2024)
[51] Exploring the Zero-Shot Capabilities of Vision-Language Models for Improving Gaze Following (2024)
[52] Multi-modal gaze following in conversational scenarios (2024)

[53] GazeLLM: a plug-and-play zero-shot LLM reasoning framework for boosting gaze target detection (2025)
[54] VL4Gaze: Unleashing Vision-Language Models for Gaze Following (2025)
[55] Gaze-Guided Multimodal LLMs for Social Scene Understanding (2025)
[56] GazeHTA: End-to-end gaze target detection with headtarget association. (2025)
[57] GazeDETR: Gaze Detection using Disentangled Head and Gaze Representations (2025)
[58] Multi-view gaze target estimation. (2025)
[59] CSGaze: Multi-view gaze target estimation. (2025)
[60] GazeVLM: A vision-language model for multi-task gaze understanding (2025)
[61] Gaze-lle: Gaze target estimation via large-scale learned encoders (2025)

[62] Toward gaze target detection of young autistic children (2026)
[63] GazeCoT: Unleashing Social Intelligence in Multimodal LLMs With Gaze-Informed Chain-of-Thought Reasoning (2026)
[64] End-to-End Shared Attention Estimation via Group Detection with Feedback Refinement (2026)
[65] Eyes on VLM: Benchmarking Gaze Following and Social Gaze Prediction in Vision Language Models (2026)
[66] Enhancing Gaze Reasoning in Vision Foundation Models for Gaze Following (2026)
[67] GazeMoE: Perception of Gaze Target with Mixture-of-Experts (2026)


# Categories

## 1. Gaze Estimation / Gaze Following 초기·기본 연구들
[2, 3, 4, 6, 7, 9, 12, 14, 19, 20, 33, 37, 62]

foundation model이나 Transformer가 핵심이 아닌, gaze following / gaze target estimation의 기본 formulation을 만든 계열
- [2] Where are they looking? 은 GazeFollow와 two-branch gaze following의 출발점.
- [3] Following gaze in video, [14] Detecting attended visual targets in video 는 video gaze following 계열.
- [4] Connecting gaze, scene, and attention 은 out-of-frame까지 포함한 generalized attention estimation 계열.
- [9] Believe it or not... 은 gaze direction field 기반의 전통 gaze target estimation 흐름.
- [19] Goo, [37] Patch-level gaze distribution, [62] young autistic children은 각각 object/domain/output-formulation 확장이지만, 방법론 축에서는 기본 gaze target detection 계열에 가까움.

## 2. Gaze Estimation에 여러 모달리티·보조 cue 추가
[7, 8, 11, 16, 17, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 35, 38, 40, 41, 45, 46, 47, 52, 58, 59, 60, 67]

depth, pose, object, 3D geometry, RGB-D, multi-view, audio, activity, head/eye/body coordination, semantic/object cue
- Depth / 3D / RGB-D: [16, 21, 25, 29, 35]
- Pose / head / eye / body cue: [8, 11, 17, 22, 23, 30, 31]
- Object / semantic / interaction cue: [24, 28, 40, 41, 45, 46, 47]
- People attribute / social cue: [38]
- Audio / conversational modality: [52]
- Multi-view: [58, 59]
- RGB+depth+text VLM-style input: [60]
- MoE로 eyes, head pose, gesture, context cue를 선택적으로 활용: [67]

## 3. Transformer-based Architecture
[26, 32, 34, 40, 41, 43, 44, 49, 57, 61]

Transformer/DETR/ViT-style token interaction이 architecture의 핵심
- [26] HGTTR: end-to-end human-gaze-target detection with transformers.
- [32] MGTR: mutual gaze detection with transformer.
- [34] GazeTransformer: gaze communication의 temporal understanding.
- [40] Object-aware gaze target detection, [41] Joint gaze-location and gaze-object detection: object/gaze joint reasoning에서 Transformer/DETR 계열로 보는 것이 자연스럽습니다.
- [43] ViTGaze: Vision Transformer interaction feature 기반.
- [44] Sharingan: multi-person gaze following Transformer.
- [49] MTGS: temporal multi-person Transformer + person-specific tokens.
- [57] GazeDETR: DETR 계열.
- [61] Gaze-LLE: frozen DINOv2 위에 small Transformer decoder를 올린 구조.

## 4. Foundation Vision Model
[44, 48, 56, 61, 66, 67]

pretrained/frozen large visual model, diffusion model, vision foundation model, foundation VLM hidden state를 gaze estimation 성능 향상에 직접 활용
- [48] Diffusion-refined VQA annotations: pretrained VQA model과 diffusion prior로 pseudo annotation을 생성합니다.
- [56] GazeHTA: pretrained diffusion model feature를 scene feature로 활용합니다.
- [61] Gaze-LLE: frozen DINOv2 encoder + lightweight decoder.
- [66] Enhancing Gaze Reasoning in Vision Foundation Models: VFM 기반 gaze following의 한계를 분석하고 local LoRA와 out-of-cone penalty로 gaze reasoning을 강화합니다.
- [67] GazeMoE: frozen foundation model에서 gaze-target-related cue를 MoE로 선택합니다.

## 5. LLM / VLM 활용 연구
[48, 51, 53, 54, 55, 60, 63, 65]

language reasoning, VQA, VLM prompting, LLM/VLM fine-tuning, gaze-as-VQA formulation이 핵심
- [48] Diffusion-refined VQA annotations: VQA model을 pseudo annotation prior로 사용.
- [51] Exploring the Zero-Shot Capabilities of VLMs: VLM을 zero-shot contextual cue extractor로 사용.
- [53] GazeLLM: LLM reasoning을 gaze target detection 보조 모듈로 사용.
- [54] VL4Gaze: gaze following을 VQA-style 문제로 구성.
- [55] Gaze-Guided Multimodal LLMs: gaze-guided MLLM 기반 social scene understanding.
- [60] GazeVLM: RGB, depth, textual prompt를 사용하는 multi-task gaze VLM.
- [63] GazeCoT: 제목 기준으로 gaze-informed CoT reasoning을 활용하는 MLLM/LLM 계열로 분류.
- [65] Eyes on VLM: VLM이 gaze following과 social gaze prediction을 얼마나 잘 수행하는지 benchmark.

## 6. Social Gaze Prediction
### 6.1. LAEO - Looking-At-Each-Other / Mutual Gaze
[1, 13, 17, 18, 32, 36]

전용 LAEO 또는 mutual gaze detection 중심
- [1] Detecting people looking at each other in videos
- [13] LAEO-Net
- [17] Boosting image-based mutual gaze detection using pseudo 3D gaze
- [18] LAEO-Net++
- [32] MGTR
- [36] Automated detection of joint attention and mutual gaze

### 6.2. SA - Shared Attention / Joint Attention
[5, 15, 36, 38, 64]

SA 또는 joint attention 자체를 직접 다루는 연구
- [5] Inferring shared attention in social scene videos
- [15] Attention Flow
- [36] Automated detection of joint attention and mutual gaze
- [38] Interaction-aware Joint Attention Estimation Using People Attributes
- [64] End-to-End Shared Attention Estimation via Group Detection with Feedback Refinement

### 6.3. LAH - Looking At Head / Looking At Humans
[39, 62]

LAH는 LAEO/SA처럼 오래된 독립 task로 발전했다기보다는, VSGaze/MTGS 이후 social gaze 통합 task 안에서 명확히 정식화된 성격
- [39] ChildPlay: LAH 전용 모델은 아니지만, gaze-to-person/head 관련 annotation을 social gaze 확장에 활용할 수 있는 benchmark 성격.
- [62] Toward gaze target detection of young autistic children: young autistic children의 gaze target detection이라는 점에서 social/person-directed gaze 분석과 연결 가능. 단, LAH 전용 연구로 단정하기보다는 LAH 인접 clinical gaze target 연구로 두는 것이 안전합니다.

### 6.4. LAH/LAEO/SA 통합
[49, 50, 65]
- [49] MTGS
- [50] A unified model for gaze following and social gaze prediction
- [65] Eyes on VLM

넓은 의미의 social gaze communication까지 포함하면 다음도 보조적으로 넣을 수 있음. 단, LAH/LAEO/SA 세 task를 MTGS처럼 명시적으로 통합한 논문이라기보다는, broader social gaze / gaze communication understanding에 가까움.
- [10] Understanding human gaze communication by spatio-temporal graph reasoning
- [34] Temporal understanding of gaze communication with GazeTransformer
- [42] Gaze pattern recognition in dyadic communication


# Related Work 작성 흐름 : 
## 2.1. Gaze Following
- 초기 two-branch / CNN gaze following : [2, 3, 4, 6, 7, 9, 12, 14, 19, 20, 33, 37, 62]
- depth/pose/object/3D/audio 등 auxiliary cue 확장 : [7, 8, 11, 16, 17, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 35, 38, 40, 41, 45, 46, 47, 52, 58, 59, 60, 67]
- Transformer / DETR / multi-person token architecture : [26, 32, 34, 40, 41, 43, 44, 49, 57, 61]
- frozen visual foundation model 기반 gaze estimator : [44, 48, 56, 61, 66, 67]
## 2.2. Social Gaze Prediction
- LAEO task : [1, 13, 17, 18, 32, 36]
- SA task : [5, 15, 36, 38, 64]
- LAH task : [39, 62]
- LAH/LAEO/SA 통합 social gaze prediction : [49, 50, 65]
## 2.3. VLM
- VLM 발전 방향 (조사 예정) : 
별도 일반 VLM 문헌 사용 권장: CLIP, BLIP/BLIP-2, Flamingo, LLaVA, GPT-4V, Qwen-VL/Qwen2.5-VL/Qwen3-VL 등.
현재 gaze 논문 리스트에서는 직접 매칭하지 않는 편이 좋음.
- VLM/LLM 기반 semantic·social gaze reasoning : [48, 51, 53, 54, 55, 60, 63, 65]