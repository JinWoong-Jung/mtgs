# VLM Unified-Eval + Scale-Up 진행 원장
Plan: docs/superpowers/plans/2026-07-14-vlm-unified-eval-and-scale.md
Branch: feat/vlm-text-graph-evidence (base: main)
Base HEAD at plan start: f05fb79

## Tasks
- Task 1: reframe text prompt (evidence-for-final-judgment) — complete (f05fb79..90e2652, review clean, 11 passed). Plan/test 모순 해소: _text_evidence_block의 "uncertain" 3곳도 함께 제거.
- Task 2: remove confidence-gated routing (config + locking tests) — NOT STARTED
- Task 3: revert text mode to plain images — NOT STARTED
- Task 4: extract _VisionReuseMixin (pure refactor) — NOT STARTED
- Task 5: wire TextGenerativeVLM + reuse-aware SFT collate — NOT STARTED
- Task 6: reuse-aware text eval collate — NOT STARTED
- Task 7: wire main()/eval_pair.py + config — NOT STARTED
- Task 8: equivalence proof + hit-rate smoke + full-suite gate — NOT STARTED

## 실행 모드
사용자 지시: SDD이되 한 태스크씩만 실행 후 한국어 중간보고 → 사용자 승인 대기 → 다음 태스크.
(skill 기본의 continuous execution은 사용자 지시로 override됨)

## 건드리지 않을 미커밋 변경
- mtgs/config/config.yaml (laeo_derive: decoder) — 사용자 변경, 계획 태스크가 stage하지 않음
- scripts/.claude/scheduled_tasks.lock 삭제 — 세션 이전부터 존재
- config_vlm_pair.yaml 사용자 편집(exp name v2, routing threshold 0.8)은 Task 2/3/7이 함께 커밋 (정합)

## 이전 계획 (완료, 참고용)
Plan: docs/superpowers/plans/2026-07-14-vlm-text-graph-evidence.md — Task 1-7 완료, HEAD f05fb79.
