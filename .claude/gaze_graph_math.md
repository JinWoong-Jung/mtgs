# Gaze Graph — 수식 정리

## 모드 구조 (세 블록은 동시에 쓰이지 않는다)

```
interaction_type = "graph"
  ViT-Adaptor 루프 안:
    SocialGraphBlock × 4      (directed, LAH/LAEO trunk)
    UndirectedSocialGraphBlock × 4  (undirected, SA 전용 분기)
  루프 후:
    decoder_lah(h_i, h_j)  → LAH
    decoder_sa(s_i, s_j)   → SA
    min(LAH_ij, LAH_ji)    → LAEO

interaction_type = "gaze_graph"
  ViT-Adaptor 루프 안:
    people_interaction × 4   (transformer와 완전히 동일한 self-attn)
    people_temporal × 4
  루프 후:
    GazeGraphBlock × 1       (concat된 proj_tokens 위에 1회만)
      → LAH, LAEO, SA, null 모두 이 블록이 직접 출력
      → decoder_lah / decoder_sa 미생성

interaction_type = "transformer"
  ViT-Adaptor 루프 안:
    people_interaction × 4
    people_temporal × 4
  루프 후:
    decoder_lah(h_i, h_j)  → LAH
    decoder_sa(s_i, s_j)   → SA
    min(LAH_ij, LAH_ji)    → LAEO
```

아래 수식은 각 모드에서 쓰이는 블록별 정리.

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

## 1. SocialGraphBlock  (Directed, Graph-mode trunk)

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

## 2. UndirectedSocialGraphBlock  (Undirected, SA branch)

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

## 3. GazeGraphBlock  (Unified Directed, gaze_graph mode)

> **PDF 슬라이드와 달라진 부분 (현재 코드 기준)**
> - Region Node 없음 (K개 gaze target 위치 노드 제거됨)
> - Edge prior: 3D 기하 feature → **edge type별 1D scalar**
> - Refiner: Temporal-Attn 없음 (5단계: Row→Col→Refresh→NodeUpdate→Reinject)
> - Target node pre-init 추가 (heatmap overlap weighted message)
> - Node update: source/target 별도 MLP, source는 outgoing만, target은 incoming만
> - SA readout: `dot(p_i, p_j)` → **head_sa MLP (null_in edge 활용)**

그래프: V = {v_1,...,v_N, v_null_in, v_null_out},  directed edges
Edge tensor: E ∈ R^{B × T × N × Tl × De},  Tl = N+2

---

### Step 1a — Source Node 초기화 (heatmap cross-attention)

    hm_feat_i = W_hm( pool(H_i) ) + pos_hm         ← H_i를 8×8 grid pooling, 1→D proj
    src_prime_i = LN( h_i + XAttn(h_i, hm_feat_i) )   "나는 어디를 보는가"
    v_src_i     = W_src( src_prime_i )               ∈ R^De

---

### Step 1b — Target Node 초기화 (incoming gaze pre-init)

    # "나는 누구에게 보여지는가"를 미리 반영
    overlap(H_i, b_j) = H_i를 정규화 후 bbox b_j 영역 내 적분값  ∈ [0,1]

    tgt_w_ij      = softmax_i( overlap(H_i, b_j) )         ← j를 향하는 source 가중치
    tgt_msg_j     = Σ_i tgt_w_ij · h_i                     ← weighted message
    tgt_gate_j    = max_i( overlap(H_i, b_j) )             ← gate (0~1)
    tgt_person_j  = h_j + tgt_gate_j · MLP([h_j ‖ tgt_msg_j])

    null_in, null_out → learnable parameter
    v_tgt = W_tgt( cat(tgt_person, null_in, null_out) )    ← shape: (Tl, De)

---

### Step 2 — Edge 초기화

**Edge type별 1D scalar prior:**

    e(i→Pj)      : scalar = cos(g_i, dir_ij)                ← gaze 방향과 j bbox 방향의 cosine
    e(i→null_in) : scalar = 1 - Σ_j overlap(H_i, b_j)      ← heatmap이 person에 걸리지 않는 비율
    e(i→null_out): scalar = 1 - sigmoid(y_io_i)             ← out-of-frame 확률

    y_io_i  = inout logit (detached)
    dir_ij  = normalize(c_j - c_i),  c = head bbox center

**Type embedding:** person=0, null_in=1, null_out=2 (learnable)

    prior_e(i→t)  = prior_w · W_scalar( scalar )            ← 1D → De
    E(i→t)^0      = MLP_init( [v_src_i ‖ v_tgt_t ‖ prior_e ‖ type_emb_t] ) · ev(i,t)

    ev(i,t) ∈ {0,1}: validity mask (i≠t, 둘 다 valid person)

---

### Step 3 — Edge Refinement  ×L layers (_UnifiedRefiner)

**Temporal attention 없음.** 각 레이어는 5단계:

    (i)  Row-Attn:
         E[i, :] → TransformerLayer_row → row_ctx[i, :]
         source i가 자신의 Tl outgoing edges를 attend
         E ← row_ctx · ev

    (ii) Col-Attn:
         E[:, k] → TransformerLayer_col → col_ctx[:, k]   (k = person 또는 null_in)
         null_out은 col-attn 제외 (row_ctx 그대로 유지)
         E ← col_ctx · ev

    (iii) Edge Refresh:
         E ← LN( E + MLP_refresh( [E ‖ row_ctx ‖ col_ctx] ) ) · ev

    (iv) Node Update (attention pooling, source/target 분리):
         # Source i: outgoing edges만 집계
         α_out(i,t)  = softmax_t( W_pool_out(E[i→t]) )   (invalid → -inf)
         out_agg_i   = Σ_t α_out(i,t) · E[i→t]
         v_src_i    ← LN( v_src_i + MLP_src( [v_src_i ‖ out_agg_i] ) )

         # Target j (person): incoming edges만 집계
         α_in(i,j)   = softmax_i( W_pool_in(E[i→j]) )   (invalid → -inf)
         in_agg_j    = Σ_i α_in(i,j) · E[i→j]
         v_tgt_j    ← LN( v_tgt_j + MLP_tgt( [v_tgt_j ‖ in_agg_j] ) )

         # null_in: 별도 MLP로 incoming 집계 동일 방식 적용
         # null_out: 업데이트 없음 (고정)

    (v)  Node Re-injection into Edges:
         E ← LN( E + MLP_inj( [E ‖ v_src_i ‖ v_tgt_t] ) ) · ev

---

### Step 4 — Readout

    LAH(i→j)   = Head_LAH( E[i→j] )
                  directed readout

    LAEO(i,j)  = ( Head_LAEO( [E[i→j] ‖ E[j→i]] )
                +  Head_LAEO( [E[j→i] ‖ E[i→j]] ) ) / 2
                  symmetrized — min() 아닌 전용 MLP

    # SA: 각 사람의 null_in edge (장면 응시 패턴)와 cross edge를 합쳐 판단
    ni_i       = E[i→null_in]                                  ← person i의 null_in edge
    ni_j       = E[j→null_in]
    SA(i,j)    = Head_SA( [ni_i ‖ ni_j ‖ |ni_i - ni_j| ‖ E[i→j] ‖ E[j→i]] )
                 (mat + mat.T) / 2  symmetrize

    null_in_i  = Head_null_in(  E[i→null_in]  )
    null_out_i = Head_null_out( E[i→null_out] )

BCE loss (0/1 label) 각 출력에 적용. invalid edge는 -1e4 masking 후 loss 제외.

---

## 모드별 수식 요약

| 태스크 | Graph (directed trunk) | Graph (undirected SA) | gaze_graph |
|--------|----------------------|----------------------|------------|
| 집계   | softmax over N+2 | sigmoid gate, mean/deg | row/col attn (temporal 없음) |
| LAH    | decoder_lah(h_i, h_j) | — | Head_LAH(E[i→j]) |
| LAEO   | min(LAH_ij, LAH_ji) | — | Head_LAEO([E[i→j]; E[j→i]]) sym. |
| SA     | decoder_sa(h_i, h_j) undirected branch | gated mean → decoder | Head_SA([ni_i; ni_j; \|diff\|; E[i→j]; E[j→i]]) |
| Prior  | g_i · d_ij (LAH) | g_i · g_j (SA) | align 1D scalar (p2p) / heatmap mass (null_in) / inout prob (null_out) |
| Null   | dual-null softmax → aux loss | — | explicit null edge heads |
