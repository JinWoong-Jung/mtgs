# AAAI 2026 Paper Plan
**생성일:** 2026-06-14 (VLM 부분 2026-06-29 제거·deferred)  
**데드라인:** 2026년 8월 초 (AAAI 2026)  
**상태:** edge-centric graph(V14.6) 정리 완료 / VLM 보강은 동료 코드 통합 후 별도 진행

---

## 제목 (가안)
*Edge-Centric Directed Graph Reasoning for Multi-Person Social Gaze Prediction*

## 한 줄 주장
Person token pair 방식 대신 directed edge feature를 직접 social prediction에 활용하면 "i가 j를 본다"는 행위를 구조적으로 인코딩할 수 있으며, LAH/LAEO/SA 전 태스크에서 유의미한 성능 향상을 달성한다. (VLM reasoning 보강은 추후 별도)

---

## Chapter Plan

### Chapter 1 — Introduction
- **Urgency (B형):** Person token pair는 "i→j" 방향성을 암묵적으로만 다루는 구조적 한계
- **Research gap:** 기존 방법은 잘못된 표현 단위(person token)로 social relation을 디코딩했다
- **Contribution preview:**
  1. Directed edge-centric graph로 social gaze를 명시적으로 모델링
  2. E[i→j] 기반 head 설계로 LAH/LAEO/SA 개선
  3. null_in/out 노드로 scene·out-of-frame 응시를 명시적 target으로 처리
  - (VLM 보강은 동료 코드 통합 후 추가 contribution 후보 — 추후)
- **Word count 목표:** ~600-800 words

### Chapter 2 — Related Work
| 계열 | 한계 정리 |
|------|-----------|
| Gaze following | Low-level "어디를 보는가"는 잘 다루나 person-to-person relation 없음 |
| Social gaze | Relation modeling 시도했으나 person token 기반 → 행위 자체를 구조적으로 인코딩 못함 |

- **결론 문단 목적:** edge-centric directed graph로 social relation을 구조적으로 인코딩하는 접근이 필요

### Chapter 3 — Method

#### (A) GazeGraphBlock
**Stage 1 — Node Initialization (Dual-role)**
- Source node = "이 사람이 어디를 보는가": `sourcei = LN(hi + XAttn(hi, hmi)) → vsrc_i`
- Target node = "이 사람이 얼마나 보여지는가": overlap-weighted msg aggregation → `vtgt_j`
- Null_in, Null_out: learnable parameter

**Stage 2 — Edge Initialization**
- Prior (soft distribution over targets, all in [0,1]):
  - p2p: `overlap(H_i, bbox_j)` — i의 normalized heatmap mass 중 j의 bbox 안에 들어가는 비율
  - null_in: `1 - Σ_j overlap(H_i, bbox_j)` — 어떤 인물 bbox에도 걸리지 않는 mass (장면/객체 응시 확률)
  - null_out: `1 - σ(inout_logit_i)` — frame 밖일 확률
- Type embedding: person=0, null_in=1, null_out=2
- `E_init[i→t] = MLP([vsrc_i ‖ vtgt_t ‖ prior ‖ type_emb])`

**Stage 3 — Dual-Role Edge Refinement (×L=2)**
1. Row attention (outgoing): source_i 고정, i에서 나가는 N+2개 edge attention
2. Column attention (incoming): target_t 고정, t로 들어오는 N개 edge attention
3. Edge refresh: `E ← LN(E + MLP([E, contextROW, contextCOL]))`
4. Node update: vsrc_i, vtgt_j attention pooling
5. Re-inject nodes: `E ← LN(E + MLP([E, vsrc_i, vtgt_j]))`

**Stage 4 — Readout Heads**
| Task | 입력 | 비고 |
|------|------|------|
| LAH | `E[i→j] ‖ E[i→null_in] ‖ E[i→null_out]` | 방향성 + i의 전체 시선 분포 (j / 장면 / frame밖) |
| LAEO | `E[i→j] ‖ E[j→i]` | 양방향 concat (또는 min(LAH_ij, LAH_ji)) |
| SA | `E[i→null_in] ‖ E[j→null_in] ‖ \|diff\|` | 3-component, 순수 장면 응시 패턴 비교 |
| null_in/out | `E[i→null]` | Aux loss |

#### (B) Null Node 설계
- **null_in:** 장면/객체 응시 (비-인물 타겟) 커버
- **null_out:** frame 밖 응시 커버
- 없으면 해당 케이스가 다른 person edge로 흘러들어가 그래프 전체 오염

#### (C) VLM 보강 — **추후 재작성 예정 (TBD)**
> 기존 VLM(Stage B) 구현은 전부 제거됨. VLM 보강은 **동료 코드 통합 후 새로 설계·기술**할 예정. 옛 설계(GraphEvidenceTokenizer/EvidenceAugmentedVLM)는 폐기.

### Chapter 4 — Experiments

**데이터셋:** VSGaze (vat + childplay + videocoatt + uco_laeo)

**현재 실험 결과 (2026-06-14 기준):**
| 모델 | Dist↓ | LAH AP | LAEO AP | SA AP |
|------|--------|--------|---------|-------|
| Transformer (baseline) | 0.0881 | 0.8982 | 0.8024 | 0.6114 |
| gaze_graph v3 (lah_min) | 0.0876 | 0.8959 | 0.7803 | 0.6639 |
| gaze_graph v3 (monitor=social_ap) | 0.0881 | **0.8992** | **0.8122** | **0.6652** |

- **Primary metric:** LAH AP (핵심 주장 검증)
- **SA AP:** +5.4p 개선 이미 확보
- **Dist:** near-SOTA, 유지 대상

**Ablation 항목 (필수):**
1. null_in/out 제거 → null 노드의 기여도 검증
2. null_in column-attention 제거 → SA의 edge-기반 cross-person scene-gaze 인코딩 검증
3. dual-role refinement / re-injection 제거 → refinement 설계의 중요성
4. node→edge readout (gaze_graph.use=false vs true) → edge-centric 핵심 주장 검증
5. SA head 입력 구성 (ni only vs +directed edge)
- (VLM 관련 ablation은 동료 코드 통합 후 추가 예정)

### Chapter 5 — Discussion & Conclusion
- **Take-home:** Directed graph로 사회적 상호작용을 모델링하면 person token보다 직관적·구조적 예측 가능 (교수님과 정교화 예정)
- **한계 1:** Graph inference overhead — gaze following 단독 대비 무거움
- **한계 2:** 학습 N≤4 고정 / 테스트 가변 N = batch_size=1 제약
- **Future work:** 더 큰 N 스케일링, VLM 보강(동료 코드 통합 후)

---

## INSIGHT Collection

| # | INSIGHT | 출처 |
|---|---------|------|
| I-1 | **Thesis:** Directed edge E[i→j]가 "i가 j를 본다"는 행위를 person token pair보다 직접적으로 인코딩한다. SA는 null_in 기반으로 이미 증명, LAH는 피처 설계 실험으로 확인 예정 | Step 1 |
| I-2 | **SA 메커니즘:** null_in edge가 "이 사람이 장면의 어디를 보는가"를 집약 → SA head가 장면 응시 패턴 비교를 직접 수행 가능 | Q2 |
| I-3 | **Null node 역할:** null_in=비-인물 응시, null_out=frame 밖. 없으면 해당 케이스가 다른 edge로 흘러들어가 그래프 오염 | Q9 |
| I-4 | **Dual-role node:** Source="어디를 보는가"(gaze output), Target="얼마나 보여지는가"(gaze input) — 동일 person이지만 역할이 다르므로 분리 초기화 필요 | Q8 |
| I-5 | **Contribution claim:** Edge-centric directed graph로 social relation 디코딩 단위를 node→edge로 이전 (VLM 보강은 추후 별도) | L5-W1 |
| I-6 | **Priority:** edge-centric graph가 핵심. (VLM은 동료 코드 통합 후 추가 contribution 후보) | Q4 |
| I-7 | **가장 약한 고리:** LAH AP 개선 미확인. 실험 결과가 나와야 핵심 주장 완성. LAEO도 파생이므로 동일 리스크 | Q15 |
| I-8 | **Open question (L5-W2):** "이 논문이 없으면 문헌에서 무엇이 빠지는가" — 교수님·팀원과 상의 후 contribution 서술 정교화 필요 | Step 2.5 |

---

## 리스크 & 다음 액션

**논문 성립 필수 조건:**
1. **LAH AP 유의미한 개선** — head 피처 설계 실험 결과 확인 (※ 현재 V14.6에서 LAH AP 개선 확보됨)
2. (VLM 보강은 동료 코드 통합 후 별도 진행 — 논문 성립 필수 요건 아님)

**현재 확보된 것:**
- SA AP: +5.4p 개선 (0.6114 → 0.6652)
- LAEO AP: transformer 수준 회복 (monitor=social_ap 기준)
- Dist: near-SOTA 유지

**재개 시 다음 단계:**
- LAH 개선 확인 후 → `/ars-outline`으로 상세 outline + evidence map
- 전체 실험 완료 후 → `/ars-full`로 초고 작성
- e.g. "LAH 개선됐어. 저번 논문 계획 불러와서 /ars-outline 진행하자"