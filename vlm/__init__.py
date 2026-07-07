"""VLM Stage-2: graph-frozen social-gaze specialist (Qwen3-VL LoRA) + soft-blend.

mtgs 의 gaze_graph 출력을 offline 캐시로 소비하는 형제 패키지.
mtgs 는 이 패키지를 import 하지 않는다 (의존 방향 vlm -> mtgs 단방향).
"""
