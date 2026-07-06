# VLM Stage-2 진행 원장
Plan: docs/superpowers/plans/2026-07-04-vlm-stage2.md
Branch: vlm-stage2 (base: main)

## Tasks
- Task 0: 환경+스켈레톤 — complete (commits d607952..8a4eedb, review clean)
- Task 1: export 훅 — complete (commits 8a4eedb..3bca96e, review clean)
  - Minor(비이슈): reviewer가 gaze_vecs 'detached' 주석 지적 → 실제로는 line586에서 detach됨 + export는 no_grad라 그래프 미보유. 수정 불요.
- Task 2: vlm/cfg.py — complete (commits 3bca96e..3da41b1, review clean)
- Task 3: vlm/graph_export.py — complete (commits 3da41b1..6c6d066, review clean; GPU smoke OK, 27키/De=256, ckpt missing=0)
  - Minor: 미사용 변수 De (무해).
  - 잔여리스크: train split transform-override 분기는 val smoke에서 미실행 → 첫 실제 train export 때 검증 필요(peer render_overlays 동일 패턴, Task5에서 재실행됨).
- Task 4: overlay/prompt — complete (commits 6c6d066..f40f216, review clean, port faithful)
  - Minor(배치정리 대상): overlay.py 미사용 `Path` import; `from __future__` before docstring → __doc__ orphan (peer 상속 패턴, 이식 파일 전반 재발). 최종리뷰에서 일괄 정리.
- Task 5: data_prep — complete (commits f40f216..da6de54: impl 576906a + ALIGN FIX da6de54, review clean; nw0 graph↔gtmeta diff=0.0)
  - Minor(배치정리): 미사용 `--workers` arg(nw=0 하드코딩); graph_export의 make_cfg/seed 순서가 data_prep과 반대(무해).
- Task 6: injection — complete (commits da6de54..ed79ea6, review clean, verbatim port, 17키 호환 확인)
  - Minor(inherited): graph_text_block의 미사용 `nin` 변수(peer 원본, 이식 유지).
- Task 7: dataset — complete (commits ed79ea6..dd25369, review clean, faithful port)
  - Minor(inherited): make_collate 미사용 zip unpack; _BLIND import-time 평가 (peer 원본).
  - note: _smoke fixture nw0로 재생성(정렬 0.0) → Task8/9는 정렬 데이터 사용.
- Task 8: eval — complete (commits dd25369..cfc96af, review clean; sgg import 0, evaluate 7키, graph-only smoke OK)
  - Minor(배치정리): build_results 내부 numpy import.
  - 한계(문서화): dsn(idx) peer val 경계 하드코딩 → blend의 per-dataset/gazefollow/inout 분기만 영향, headline social F1/AP 무관. 정식 수치는 nograph/token(gtmeta기반) 경로 사용.
  - note: _smoke graph cache는 LAEO 양성 확보 위해 504샘플(nw0)로 재생성됨(30프레임 overlays superset).
- Task 9: train+wandb — complete (commits cfc96af..29ed79c, review clean; nograph smoke 실제 실행 성공, W&B 7키 gated, LoRA config 보존)
  - Minor(inherited/cosmetic): float(loss) requires_grad UserWarning(.item()로 수정 권장); __future__/docstring 순서.
  - 미검증: token 경로 런타임(hook 주입) — 실제 실행 시 검증 필요(이식 충실).
- Task 10: train_vlm.sh — complete (commits 29ed79c..ac2c5f7, review clean; CLI flags 실모듈 대조 일치, header+cd-to-root 정확)
  - Minor: `set -e` 없음 → eval MODE에서 nograph 실패해도 blend 실행(train_vsgaze.sh 관례와 동일하나 위험).

## FINAL REVIEW: READY-AFTER-FIXES (opus) → MUST-FIX 1건(set -e) 적용 commit b790f8e. 나머지 Minor 전부 DEFER. cross-cutting(캐시키/sid정렬/import DAG/7키/sa_gt↔coatt_gt) 일관 확인.
##   post-merge 런타임 검증 필요: 첫 val 전량 export 후 graph-only F1_LAH ≈ baseline(~0.837) 확인(orientation), token 경로 실행.

## Minor findings roll-up (최종리뷰 triage 대상)
- overlay.py 미사용 `Path` import
- `from __future__` before docstring → __doc__ orphan (이식 파일 전반)
- data_prep.py 미사용 `--workers` arg
- graph_export.py 미사용 `De` 변수; make_cfg/seed 순서가 data_prep과 반대
- injection.py 미사용 `nin` 변수(peer 상속)
- dataset.py make_collate 미사용 zip unpack; `_BLIND` import-time 평가(peer 상속)
- eval.py build_results 내부 numpy import
- train.py float(loss) requires_grad UserWarning(.item() 권장)
- train_vlm.sh `set -e` 없음(eval MODE 다단계 위험)
- 문서화 한계: eval.py dsn(idx) peer val 경계 하드코딩(blend per-dataset/inout만 영향, headline 무관)

## ===== 2026-07-06 RUN: token-mode consolidation (plan 2026-07-06-vlm-token-consolidation.md) =====
(controller ledger for this run's Tasks 1–5 below)
- Task 1: complete (commits b790f8e..8dade2f, review clean; Spec ✅ Approved)
  - Minor(defer→final): injection.py 부유 docstring(구 graph-token 헤더 문자열, no-op); dataset.py `# Collate: graph-text / vision-only` stale 주석. 둘 다 Task2/3 재작성 영역과 겹쳐 자연 정리 가능.
- Task 2: complete (commits 8dade2f..54af3f6, review clean; Spec ✅ Approved)
  - Minor(defer→final): test_projector_shape_and_role_conditioning의 role 단언이 role_emb zero-init 때문에 init 시점 vacuous(plan-mandated 테스트 설계 약점). 구현은 정상(feats+role_emb[role_ids]). 최종 triage 대상.
- Task 3: complete (commits 54af3f6..47f83ff, review clean; Spec ✅ Approved)
  - Minor(defer→final): tests/test_vlm_token_prompt.py 중간 위치 import(torch, gather_feats) — PEP8 스타일만, 계획서 append 구조에서 유래.
- Task 4: complete (commits 47f83ff..96371fb, review clean; Spec ✅ Approved)
  - Minor(defer→final): tests/test_vlm_token_injection.py append 블록 import 위치(중간). 스타일만.

## 2026-07-06 Consolidation (token-only)
- Removed B/D/E graph-text path (train `nograph`, eval `nograph`, LoRADatasetNoGraph, graph_text_block) + 4 dead prompt helpers.
- Token mode redesigned: task-specific role-aware injection (LAH 3 / LAEO 4 / SA 6 tokens), inline prompt with head-box text, role-keyed projector, variable-length collate.
- Launcher: MODE ∈ {export, overlays, token, eval}; eval → `vlm.eval token` + blend.
- Plan: docs/superpowers/plans/2026-07-06-vlm-token-consolidation.md
- Orientation INVARIANT baked in: EDGE_FWD("i→j")=edge_pp[j,i]. Verify at first val eval (F1_LAH sanity).
