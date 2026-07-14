# Introduction

## 1. Task Definition and Importance

Human gaze is one of the most informative and powerful nonverbal cues. It reveals a person's attention, conveys intent, and carries social interaction and relationship. In multi-person scenes, understanding the overall interaction requires not only low-level gaze following -- where each person is looking -- but also high-level social relations: who is looking at whom (LAH), who is looking at each other (LAEO), and who is jointly attending to the same target (SA). Modeling these two levels jointly is important for developmental and behavioral analysis, human-robot interaction, and fine-grained video understanding.

## 2. Existing Methods

Gaze following was first formulated as a deep learning problem by Recasens et al. [#]. Their two-pathway architecture processed the global scene and the target person's head information separately, then combined them to predict a gaze heatmap. A wide range of CNN-based models have been proposed, including methods that incorporate additional modalities such as depth [#], pose [#], and 3D head direction [#]. Transformer-based end-to-end models [#] later advanced gaze target estimation further, and recent work has shown that vision foundation models (VFMs) such as DINO series[#] can provide strong visual representations for this task [#]. In parallel, vision-language approaches [#] have begun to frame gaze understanding as an LLM-conditioned [#] or multi-task multimodal problem [#].

High-level social gaze prediction has also been studied through relation-specific tasks. LAEO methods [#] focus on detecting whether two people are looking at each other, while shared attention methods [#] predict whether multiple people attend to the same person or location in a scene. [ChildPlay] provides a dataset for children's gaze behavior with LAH annotations. More recently, [MTGS] integrated these three prediction tasks into a unified architecture, jointly addressing gaze target estimation and social gaze reasoning across LAH, LAEO, and SA.

## 3. Limitations of Existing Methods

Despite these advances, many social gaze methods share a structural limitation: they predict relations, but do not represent the relation itself as an explicit modeling unit. For example, [MTGS] predicts social relations by concatenating pairs of person tokens and passing them through shallow MLP decoders. In this design, the directed act "person i looks at person j" does not exist as an explicit representation; instead, the decoder must implicitly recover it from two person embeddings for every pair. Existing graph-based approaches have not fully resolved this issue. Fan et al. [#] performs message passing over a spatio-temporal graph whose nodes are people, but edge information is used mainly as scalar connectivity weights or messages for node updates, and the final gaze communication labels are predicted at the node-level. As a result, relation information collapses into node states before prediction, and the output does not explicitly specify the counterpart of the relation. Gupta et al. [#] uses a GAT [#] to model person-person interactions, but the graph edges do not contain learned edge features beyond attention weights, and social relations are still decoded implicitly from concatenated updated node embeddings.

This node-centric design introduces two main costs. First, the embedding of a person aggregates interactions with many other people in the scene, so when judging a specific pair (i, j), signals from unrelated relations can act as noise. Second, because directionality is not built into the representation--except through the order of concatenated node embeddings--a lightweight pairwise decoder must infer directed facts such as "i looks at j" independently for each pair. In addition, existing methods usually do not treat non-person gaze as an explicit target for relational reasoning. When a person looks at an object in the scene or outside the frame, this evidence is often discarded or handled separately, although it provides important negative evidence for social relation prediction. Finally, ambiguous gaze directions, occlusions, and context-dependent targets may require visual commonsense beyond what purely geometric or person-pair pipelines can provide.

## 4. Proposed Method

In this paper, we propose [FRAMEWORK], an edge-centric graph framework that jointly addresses gaze following and social gaze prediction. The key idea is to align the modeling unit with the prediction target: instead of reconstructing social relations after the fact from person embeddings, [FRAMEWORK] represents each directed gaze hypothesis, such as "person i looks at person j", as a dedicated edge state from a source person to a target entity. LAH, LAEO, and SA are then read out directly from these edge states. Directionality is therefore not something a decoder must recover implicitly; it is built into the representation itself. Each person is represented with separate source and target roles, so the edges E[i->j] and E[j->i] exist as distinct states from the beginning. We further augment the target space with two explicit null targets: Null_in, which represents gaze toward an in-frame non-person target, and Null_out, which represents gaze outside the frame. This allows non-social gaze, which previous methods often discard or handle separately, to serve as both relational supervision and negative evidence for social relation prediction.

The edge states are neither isolated nor indiscriminately mixed. [FRAMEWORK] iteratively refines them through attention among outgoing edges from the same source and incoming edges to the same target. The refined edges are then aggregated to update source and target node states, which are injected back into the edge representations. In this way, evidence such as "person i is already looking at person k" can influence the decision for E[i->j] only through structured relational paths, rather than being collapsed into a single node vector where unrelated relations may become noise. Finally, to handle cases that are difficult to resolve from geometry alone, we complement the graph with vision-language reasoning. After training the graph model, we freeze it and fine-tune Qwen3-VL-8B [#] with LoRA [#], conditioning the VLM on edge-based graph evidence through text prompts and soft graph tokens. The final prediction softly blends the graph and VLM estimates, preserving the explicit relational structure of the graph while allowing the VLM to correct ambiguous cases involving occlusion, subtle head pose, or context-dependent targets.

## 5. Contribution Summary

In summary, our contributions are as follows:

- We formulate [FRAMEWORK], an edge-centric graph-based framework that represents candidate gaze relations as edge-level states with source/target role separation. We also introduce `Null_in` and `Null_out` targets for relational supervision of non-person and out-of-frame gaze, together with iterative node-edge refinement over outgoing and incoming gaze context.
- Beyond the graph-based architecture, we further fine-tune a VLM to incorporate visual-linguistic reasoning for cases that remain ambiguous from relational geometry alone. This refinement complements the explicit graph evidence by revising uncertain predictions under occlusion, subtle head pose, and context-dependent gaze targets.
- We achieve strong performance on VSGaze across `AP_SA`, `F1_LAH(PP)`, `F1_LAEO(PP)`, and `Dist` ([XX.X], [XX.X], [XX.X], [X.XXX]).

Edit version : In summary, our contributions are as follows:

- Edge-centric gaze graph. We propose [FRAMEWORK], which aligns the modeling unit of social gaze prediction with the relation being predicted. Each directed gaze hypothesis is represented as a dedicated edge state between seperate source and target roles, and LAH, LAEO, and SA are read out directly from these edge states. The edges are iteratively refined through attention among outgoing and incoming edges with node-edge mutual updates. We furthur introduce explicit Null_in and Null_out targets, allowing in-frame non-person and out-of-frame gaze, which previous methods often discard, to be used as relational supervision and negative evidence.
- VLM-based complementary refinement. We freeze the trained [FRAMEWORK] and fine-tune a VLM conditioned on edge-based graph evidence through text prompts and soft graph tokens, softly blending graph and VLM predictions at inference time. This refinement preserves the explicit relational structure of the graph while using vision-language reasoning to correct cases that remain ambiguous from geometry alone, such as occlusion, subtle head pose, and context-dependent targets.
- Unified validation on VSGaze. On the comprehensive VSGaze benchmark, [FRAMEWORK] improves social gaze prediction -- reaching AP_SA [XX.X], F1_LAH(PP) [XX.X], and F1_LAEO(PP) [XX.X] -- while maintaining competitive gaze following performance with Dist [X.XXX].

# Citations

[1] Emery, Nathan J. "The eyes have it: the neuroethology, function and evolution of social gaze." *Neuroscience & Biobehavioral Reviews* 24.6 (2000): 581-604.

[2] Admoni, Henny, and Brian Scassellati. "Social eye gaze in human-robot interaction: a review." *Journal of Human-Robot Interaction* 6.1 (2017): 25-63.

[3] Recasens, Adria, et al. "Where are they looking?." *Advances in neural information processing systems* 28 (2015).

[4] Lian, Dongze, Zehao Yu, and Shenghua Gao. "Believe it or not, we know what you are looking at!." *Asian Conference on Computer Vision*. Cham: Springer International Publishing, 2018.

[5] Chong, Eunji, et al. "Connecting gaze, scene, and attention: Generalized attention estimation via joint modeling of gaze and scene saliency." *Proceedings of the European conference on computer vision (ECCV)*. 2018.

[6] Chong, Eunji, et al. "Detecting attended visual targets in video." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2020.

[7] Fang, Yi, et al. "Dual attention guided gaze target detection in the wild." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2021.

[8] Bao, Jun, Buyu Liu, and Jun Yu. "Escnet: Gaze target detection with the understanding of 3d scenes." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2022.

[9] Jin, Tianlei, et al. "Depth-aware gaze-following via auxiliary networks for robotics." *Engineering Applications of Artificial Intelligence* 113 (2022): 104924.

[10] Miao, Qiaomu, Minh Hoai, and Dimitris Samaras. "Patch-level gaze distribution prediction for gaze following." *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*. 2023.

[11] Gupta, Anshul, Samy Tafasca, and Jean-Marc Odobez. "A modular multimodal architecture for gaze target prediction: Application to privacy-sensitive settings." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*. 2022.

[12] Horanyi, Nora, et al. "Where are they looking in the 3D space?." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*. 2023.

[13] Hu, Zhengxi, et al. "Gaze target estimation inspired by interactive attention." *IEEE Transactions on Circuits and Systems for Video Technology* 32.12 (2022): 8524-8536.

[14] Tu, Danyang, et al. "End-to-end human-gaze-target detection with transformers." *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. IEEE, 2022.

[15] Tonini, Francesco, et al. "Object-aware gaze target detection." *Proceedings of the IEEE/CVF international conference on computer vision*. 2023.

[16] Tafasca, Samy, Anshul Gupta, and Jean-Marc Odobez. "Sharingan: A transformer architecture for multi-person gaze following." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2024.

[17] Ryan, Fiona, et al. "Gaze-lle: Gaze target estimation via large-scale learned encoders." *Proceedings of the Computer Vision and Pattern Recognition Conference*. 2025.

[18] Wang, Shijing, et al. "VL4Gaze: Unleashing Vision-Language Models for Gaze Following." *arXiv preprint arXiv:2512.20735* (2025).

[19] Cao, Xu, et al. "Gaze Target Estimation Anywhere with Concepts." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2026.

[20] Mathew, Athul M., et al. "Gazevlm: A vision-language model for multi-task gaze understanding." *arXiv preprint arXiv:2511.06348* (2025).

[21] Marin-Jimenez, Manuel Jesús, et al. "Detecting people looking at each other in videos." *International Journal of Computer Vision* 106.3 (2014): 282-296.

[22] Marin-Jimenez, Manuel J., et al. "Laeo-net: revisiting people looking at each other in videos." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2019.

[23] Marin-Jimenez, Manuel J., et al. "LAEO-Net++: revisiting people Looking At Each Other in videos." *IEEE Transactions on Pattern Analysis and Machine Intelligence* 44.6 (2022): 3069-3081.

[24] Fan, Lifeng, et al. "Inferring shared attention in social scene videos." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018.

[25] Tafasca, Samy, Anshul Gupta, and Jean-Marc Odobez. "Childplay: A new benchmark for understanding children's gaze behaviour." *Proceedings of the IEEE/CVF international conference on computer vision*. 2023.

[26] Gupta, Anshul, et al. "Mtgs: A novel framework for multi-person temporal gaze following and social gaze prediction." *Advances in Neural Information Processing Systems* 37 (2024): 15646-15673.

[27] Hu, Edward J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." *International Conference on Learning Representations*. 2022.

[28] Bai, Shuai, et al. "Qwen3-VL Technical Report." *arXiv preprint arXiv:2511.21631* (2025).
