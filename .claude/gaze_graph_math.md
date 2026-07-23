# Gaze Graph — 수식 정리

> **2026-06-13 리팩토링 공지**: `interaction_type` (`transformer`/`graph`/`hypergraph`) 스위치와
> 그에 딸린 `SocialGraphBlock`/`UndirectedSocialGraphBlock` 코드는 **완전히 삭제**되고 `gaze_graph`
> (`GazeGraphBlock`) 단일 아키텍처로 통합됐다. 아래 **§1, §2는 더 이상 존재하지 않는 코드에 대한
> 역사적 기록**이며 현재 `train_vsgaze.sh` 파이프라인과 무관하다. 현재 활성 아키텍처는 **§3
> GazeGraphBlock**만 참조하면 된다 (그마저도 V14 이후 상당 부분 바뀌었으므로 아래 §3 본문은
> 현재 코드 기준으로 다시 작성됨 — 과거 iteration은 [version.md](version.md) 참조).

## 모드 구조 (히스토리 — §1·§2는 삭제된 코드)

```
interaction_type = "graph"  [삭제됨, 코드 없음]
  ViT-Adaptor 루프 안:
    SocialGraphBlock × 4      (directed, LAH/LAEO trunk)
    UndirectedSocialGraphBlock × 4  (undirected, SA 전용 분기)
  루프 후:
    decoder_lah(h_i, h_j)  → LAH
    decoder_sa(s_i, s_j)   → SA
    min(LAH_ij, LAH_ji)    → LAEO

interaction_type = "gaze_graph"  [현재 유일한 아키텍처 — config에서는 그냥 gaze_graph.use=true/false]
  ViT-Adaptor 루프 안:
    people_interaction × 4   (self-attn)
    people_temporal × 4
  루프 후:
    gaze_graph.use=true  → GazeGraphBlock × 1 (concat된 proj_tokens 위에 1회만)
      → LAH, LAEO, SA, null_in, null_out 모두 이 블록이 직접 출력
    gaze_graph.use=false → decoder_lah(h_i,h_j) / decoder_sa(s_i,s_j) (person-token pair 직접 예측)
      → LAEO = min(LAH_ij, LAH_ji)

interaction_type = "transformer"  [삭제됨, 코드 없음 — gaze_graph.use=false가 사실상의 대체]
```

아래 §1·§2는 삭제된 코드의 수식 정리(참고용), §3은 현재 코드의 수식 정리.

---

## 공통 기호

| 기호 | 의미 |
|------|------|
| N | person 수 (padding 포함) |
| T | temporal window |
| D | person token dim (768) |
| De | edge embedding dim |
| Tl = N+2 | target slot 수 (N person + null_in + null_out) |
| h_i | person token i ∈ R^D |
| g_i | unit gaze direction ∈ R^2 |
| b_i | head bbox [x1,y1,x2,y2] |
| c_i | head center = (b_i[:2] + b_i[2:]) / 2 |
| H_i | per-person gaze heatmap |
| d_ij | normalize(c_j - c_i) |

---

## 1. SocialGraphBlock  (Directed, Graph-mode trunk) — ⚠️ DEPRECATED, 코드 삭제됨

그래프: V = {h_1,...,h_N, v_in, v_out}
집계 방향: outgoing — i가 자신이 바라보는 노드로부터 수집

### 반복 (l = 1,...,L):

(a) Edge score

    e(i→j)   = MLP_dir(h_i, h_j) + λ · w_l · (g_i · d_ij)   ← LAH gaze prior
    e(i→in)  = MLP_null_in(h_i, v_in)
    e(i→out) = MLP_null_out(h_i, v_out)

    λ     = prior_w_attn  (learnable scalar)
    w_l   = softmax(γ)[l]  (learnable per-layer decay)

(b) Dual-null softmax over N+2 targets

    α(i,*)  = softmax( [e(i→1), ..., e(i→N),  e(i→in),  e(i→out)] )

(c) Message aggregation

    m_i = Σ_j α(i→j) · W_msg(h_j)
        + α(i→in)  · W_msg(v_in)
        + α(i→out) · W_msg(v_out)

(d) Gated node update

    gate_i = sigmoid( W_gate(h_i) )
    delta_i = W_upd( concat(h_i, m_i) )
    h_i ← LN( h_i + gate_i ⊙ delta_i )

출력: (h, α_null_in, α_null_out)  ← α_null은 auxiliary null loss 전용, 예측 미사용

---

## 2. UndirectedSocialGraphBlock  (Undirected, SA branch) — ⚠️ DEPRECATED, 코드 삭제됨

그래프: V = {h_1,...,h_N}, 대칭 edge
핵심 불변식: gate(i,j) == gate(j,i)

### 반복 (l = 1,...,L):

(a) 대칭 edge score

    e_ij = MLP_edge( concat(h_i + h_j,  |h_i - h_j|) )
         + λ_SA · w_l · (g_i · g_j)    ← SA cosine prior

(b) 독립 sigmoid gate + degree-normalized mean

    gate_ij = sigmoid(e_ij) · valid(i,j)
    deg_i   = |{ j : valid(i,j) }|
    m_i     = Σ_j gate_ij · W_msg(h_j)  /  deg_i

(c) Node update  (SocialGraphBlock과 동일 형식)

    h_i ← LN( h_i + sigmoid(W_gate(h_i)) ⊙ W_upd( concat(h_i, m_i) ) )

---

## 3. GazeGraphBlock  (Unified Directed, gaze_graph mode) — 현재 코드 기준 (V14+)

> **아래 내용이 코드 기준 최신판이다.** 과거(≤V13) 설계는 소스 노드에 heatmap cross-attention,
> 타깃 노드에 overlap-가중 message pre-init을 따로 두고 edge prior가 1D scalar였다. V14에서
> node init이 `person_token + face + geom` 단일 통합식으로 단순화됐고, edge prior는 4채널로
> 확장됐으며, row/col/temporal/null_in/null_out/face_proj/node_geom/type_embed에 대한
> capacity-controlled ablation 스위치가 추가됐다. 버전별 변경 이력은 [version.md](version.md) 참조.

그래프: V = {v_1,...,v_N, v_null_in, v_null_out},  directed edges
Edge tensor: E ∈ R^{B × T × N × Tl × De},  Tl = N+2

---

### Step 1 — 통합 Node 초기화 (src/tgt 공유, heatmap XAttn 없음)

    face_i = use_face_proj ? W_face( stopgrad(gaze_feat_i) ) : 0     ← raw GazeEncoder token 재주입
    geom_i = use_node_geom ? MLP_geom( [c_i, wh_i, g_i] ) : 0        ← [cx,cy,w,h,gaze_vec]
    node_i = LN( person_token_i + face_i ) + geom_i                 ∈ R^D

    v_tgt  = W_node( cat(node_1..N, null_in_param, null_out_param) )  ← shape: (Tl, De)
    v_src  = v_tgt[:N]                                                ← persons as sources (src=tgt 공유)

`face_proj`/`node_geom_mlp`는 zero-init(끄면 forward에서 항등적으로 0을 더하는 것과 동일한 시작점),
즉 ablation on/off 모두 안전한 no-op에서 학습을 시작한다.

---

### Step 2 — Edge 초기화 (4채널 기하 prior + type embedding)

    dir_ij  = normalize(c_j - c_i),  c = head bbox center
    overlap(H_i, b_j) = H_i를 정규화 후 bbox b_j 영역 내 적분값 ∈ [0,1]
    y_io_i  = inout logit (detached)

    prior(i→Pj)       = [ cos(g_i, dir_ij),  overlap(H_i,b_j),  dir_ij_x,  dir_ij_y ]   ← 4채널
    prior(i→null_in)  = [ 1 - Σ_j overlap(H_i,b_j),  0, 0, 0 ]     ← heatmap이 person에 안 걸리는 비율
    prior(i→null_out) = [ 1 - sigmoid(y_io_i),        0, 0, 0 ]     ← out-of-frame 확률

    edge_feat(i→t) = use_prior ? prior_w(zero-init 학습 스칼라) · W_prior( prior(i→t) ) : 0   ← 4D → De
    type_exp(i→t)  = use_type_embed ? type_emb[ person=0 | null_in=1 | null_out=2 ] : 0

    E(i→t)^0 = MLP_init( [v_src_i ‖ v_tgt_t ‖ edge_feat(i→t) ‖ type_exp(i→t)] ) · ev(i,t)

    ev(i,t) ∈ {0,1}: validity mask (i≠t, 둘 다 valid person; null_in/null_out은 use_null_in/out=false면
                     항상 0 — 마스킹이 edge-init 이전 단일 지점에서 이뤄져 attention key로도 제외됨)

---

### Step 3 — Edge Refinement  ×L layers (`_UnifiedRefiner`, 레이어마다 독립 가중치)

각 레이어는 6단계 (①②는 row/col attention ablation, ⑥은 module-skip ablation):

    ① Row-Attn (use_row_attn):
         E[i, :] → TransformerLayer_row → row_ctx[i, :]
         source i가 자신의 Tl(=N+2) outgoing edges를 attend
         off면 row_ctx = 0 (모듈은 그대로 존재, 파라미터 수 불변)

    ② Col-Attn (use_col_attn):
         E[:, k] → TransformerLayer_col → col_ctx[:, k]   (k = person 또는 null_in; null_out 제외)
         off면 col_ctx = 0

    ③ Edge Refresh:
         E ← LN( E_in + MLP_refresh( [row_ctx ‖ col_ctx] ) ) · ev
         (row/col 둘 다 off면 refresh(cat(0,0))의 bias 항만 반영)

    ④ Node Update (attention pooling, row/col 방향별 게이팅):
         # use_row_attn: source i, 자신의 outgoing edges 집계
         α_out(i,t)  = softmax_t( W_pool_out(E[i→t]) )   (invalid → -inf)
         out_agg_i   = Σ_t α_out(i,t) · E[i→t]
         v_src_i    ← LN( v_src_i + MLP_src( [v_src_i ‖ out_agg_i] ) )

         # use_col_attn: target j(person)와 null_in, incoming edges 집계 (별도 MLP)
         α_in(i,j)   = softmax_i( W_pool_in(E[i→j]) )   (invalid → -inf)
         in_agg_j    = Σ_i α_in(i,j) · E[i→j]
         v_tgt_j    ← LN( v_tgt_j + MLP_tgt( [v_tgt_j ‖ in_agg_j] ) )     ← person j
         v_null_in  ← LN( v_null_in + MLP_nullin( [v_null_in ‖ in_agg_null_in] ) )
         # null_out: 업데이트 없음 (고정)

    ⑤ Node Re-injection into Edges (row 또는 col 중 하나라도 on이면 실행):
         E ← LN( E + MLP_inj( [E ‖ v_src ‖ v_tgt] ) ) · ev

    ⑥ Temporal Edge-Attn (use_temporal_attn, T>1일 때만; off면 모듈 자체 미생성):
         각 edge (i,t)가 자신의 T개 프레임에 대해 self-attention
         — MTGS의 people_temporal/gaze_encoder_temporal과는 별개의, 그래프 고유 temporal consistency

---

### Step 4 — Readout  (E_pp = E[:,:,:,:N,:], person-to-person 부분)

    LAH(i→j)   = Head_LAH( E[i→j] )
                  directed readout

    laeo_derive="decoder"(기본):
      LAEO(i,j) = ( Head_LAEO( [E[i→j] ‖ E[j→i]] ) + Head_LAEO( [E[j→i] ‖ E[i→j]] ) ) / 2
                   symmetrized — 전용 MLP (head_laeo forward 실행)
    laeo_derive="lah_min":
      head_laeo forward 자체를 skip (낭비 연산 제거), mtgs_net.py에서
      LAEO(i,j) = min(LAH(i→j), LAH(j→i))로 derive

    # SA: 각 사람의 null_in edge (장면 응시 패턴)와 cross edge를 합쳐 판단
    ni_i       = E[i→null_in]                                  ← person i의 null_in edge
    ni_j       = E[j→null_in]
    SA(i,j)    = Head_SA( [ni_i ‖ ni_j ‖ |ni_i - ni_j| ‖ E[i→j] ‖ E[j→i]] )
                 (mat + mat.T) / 2  symmetrize

    null_in_i  = Head_null_in(  E[i→null_in]  )
    null_out_i = Head_null_out( E[i→null_out] )

BCE loss (0/1 label) 각 출력에 적용. invalid edge는 -1e4 masking 후 loss 제외.
`use_null_in`/`use_null_out`이 꺼진 null head는 상수(0) edge만 읽으므로 해당 BCE 항이 학습 loss와
W&B 로깅에서 자동 제외된다 (`compute_dual_null_loss` 가중치 0).

---

## 모드별 수식 요약

| 태스크 | Graph (directed trunk) — ⚠️ 삭제됨 | Graph (undirected SA) — ⚠️ 삭제됨 | gaze_graph — 현재 유일한 아키텍처 |
|--------|----------------------|----------------------|------------|
| 집계   | softmax over N+2 | sigmoid gate, mean/deg | row/col/temporal attn (전부 ablation 가능) |
| LAH    | decoder_lah(h_i, h_j) | — | Head_LAH(E[i→j]) |
| LAEO   | min(LAH_ij, LAH_ji) | — | Head_LAEO([E[i→j]; E[j→i]]) sym. (또는 lah_min derive) |
| SA     | decoder_sa(h_i, h_j) undirected branch | gated mean → decoder | Head_SA([ni_i; ni_j; \|diff\|; E[i→j]; E[j→i]]) |
| Prior  | g_i · d_ij (LAH) | g_i · g_j (SA) | 4채널: [align, heatmap-overlap, rel_pos_x, rel_pos_y] (p2p) / heatmap mass (null_in) / inout prob (null_out) |
| Null   | dual-null softmax → aux loss | — | explicit null edge heads (ablation 가능) |

`gaze_graph.use=false`로 설정하면 `GazeGraphBlock` 대신 person-token pair에서 직접 예측하는
`decoder_lah`/`decoder_sa`(`LinearDecoderSocialGraph`) 경로가 활성화된다 — LAEO는 이때
`min(LAH_ij, LAH_ji)`로 derive. 이는 위 "Graph" 열과는 다른, gaze_graph 모듈 내부의 A/B 스위치다.
