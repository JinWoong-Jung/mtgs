# GCN Interaction Module 구현 계획

## 목표

기존 `people_interaction` (TransformerBlock, spatial) + `people_temporal` (TransformerBlock, temporal)을  
**Directed Graph 기반 SocialGraphBlock**으로 대체한다.  
`config.yaml`의 `interaction.type` 설정 하나로 baseline(Transformer)과 신규(Graph)를 전환 가능하게 한다.  
**baseline 코드는 일절 변경하지 않는다 — `interaction.type: transformer` 설정 시 기존과 완전히 동일하게 동작해야 한다.**

---

## 결정 완료 항목 요약

| 항목 | 결정 |
|------|------|
| A1. SocialGraphBlock 구현 방식 | ✅ PyG sparse (MessagePassing 기반) |
| A2. TemporalGraphBlock 구현 방식 | ✅ GATv2Conv (PyG) |
| A3. Geometric prior 적용 시점 | ✅ SocialGraphBlock 내 첫 번째 iteration에만 |
| A4. SocialGraphBlock num_layers | ✅ 2 (초기값) |
| B1. Null node GT supervision | ✅ hard label: w_null GT(i) = 1 if 모든 LAH_gt(i→j)==0, else 0 |
| B2. LAEO 유도 방식 | ✅ min(lah(i→j), lah(j→i)) 유지 |
| B3. Edge normalization | ✅ LAH: softmax, SA: sigmoid (기존 결정) |
| C1. Stage 1 학습 | ✅ 기존 transformer ckpt 그대로 Stage 2 초기값 |
| C2. Freeze 전략 | ✅ 기존과 동일 (vit_encoder, image_tokenizer freeze) |
| C3. LR 그룹 | ✅ social_graph_blocks, temporal_graph_blocks → 기존 lr × 3 그룹 |
| D1. inout decoder | ✅ 별도 inout_decoder_graph (token_dim=768 입력) 신규 생성 |
| D2. Null node 초기화 | ✅ nn.Parameter (learnable) |
| D3. gaze_vec 전달 경로 | ✅ 루프 안에서 gaze_vec.view(b*t, n, -1) inline, num_valid 재사용 |
| D4. LAH logit 출력 위치 | ✅ 마지막 블록(i=3)의 edge만 final prediction |
| E1. 비교 실험 | ✅ baseline / graph(prior X) / graph(prior O) 전부 열어둠 |
| E2. Null node ablation | ✅ use_null_node true/false 전부 열어둠 |

---

## 아키텍처 요약

### Baseline (현재, 변경 없음)
```
ViT Adaptor block i (InteractionBlock):
  Injector (scene → person) → DINOv2 ViT layers → Extractor (person → scene)
  → people_interaction[i]  (TransformerBlock, person-person self-attention)
  → people_temporal[i]     (TransformerBlock, time-axis self-attention, t>1일 때만)
```

### Graph 모드 (신규)
```
ViT Adaptor block i (InteractionBlock):  ← 변경 없음
  Injector → DINOv2 ViT layers → Extractor
  → SocialGraphBlock[i]   (PyG MessagePassing, directed, person-person spatial)
  → TemporalGraphBlock[i] (GATv2Conv, undirected, t>1일 때만)
```

### SocialGraphBlock 내부 설계 (Scene Graph 방식, 반복 정제)
```
입력 (PyG sparse 형식):
  x            (total_nodes, D)      — 배치 내 모든 유효 노드 features
  edge_index   (2, total_edges)      — 유효 노드 간 directed 완전 연결
  batch        (total_nodes,)        — 어느 그래프 소속인지
  gaze_vec     (total_nodes, 2)      — prior 계산용 (첫 iteration에만 사용)
  head_bboxes  (total_nodes, 4)      — bbox center 계산용
  null_node_h  nn.Parameter (D,)     — learnable null node feature

  ※ 변환: MTGS.forward() 내 루프에서
    person_tokens (b*t, N, D) + num_valid (b*t,) → PyG 변환은 SocialGraphBlock.forward() 내부에서 수행
    - num_valid[i]개 유효 노드만 추출하여 concat → x, batch 구성
    - edge_index: 각 그래프의 유효 노드 간 모든 directed pair (N_i*(N_i-1)개) + i→null edge
    - 출력 후 다시 (b*t, N, D) dense로 복원 (패딩 위치는 0)

[초기화 — 외부에서 전달된 gaze_vec 기반]
  e_dir(i→j)^prior = cosine(gaze_vec_i, normalize(center_j - center_i))

[반복: num_layers=2 회]
  1. edge logit 계산:
     - iteration 0:
       e_dir(i→j)   = MLP_dir(concat(h_i, h_j))  +  prior_weight * e_dir(i→j)^prior
     - iteration 1+:
       e_dir(i→j)   = MLP_dir(concat(h_i, h_j))          # prior 미적용

  2. null node edge:
     e_null(i) = MLP_null(h_i)  # person i → null (directed)

  3. edge weight 정규화:
     w_dir(i→j) = softmax_j( [e_dir(i→1), ..., e_dir(i→N-1), e_null(i)] )
                  # LAH: 배타적 gaze → softmax (null 포함)

  4. message passing (directed):
     msg_i = sum_j [ w_dir(i→j) * W_msg * h_j ]  +  w_null(i) * W_msg(null_node)

  5. node 업데이트:
     h_i = LayerNorm(h_i + Linear(concat(h_i, msg_i)))

[루프 종료 후 SA 계산]
  e_sa(i,j) = MLP_sa(msg_i + msg_j)   # msg = 마지막 iteration의 directed outgoing aggregation
  # 기존 e_undir(i,j) = MLP_sa(h_i + h_j) 무방향 엣지 방식에서 변경됨
  # msg_i가 "i가 바라보는 장면 표현"을 담고 있어 SA 판단이 시선 방향 정보를 반영

출력:
  - person_tokens_updated  (b*t, N, D)   — dense 복원 (null 제외, 패딩 위치 0)
  - lah_logits  (b*t, N*(N-1))           — 마지막 블록(i=3)만 최종 예측으로 사용
  - sa_logits   (b*t, N*(N-1))           — 마지막 블록(i=3)만 최종 예측으로 사용
```

### TemporalGraphBlock 설계 (GATv2Conv, PyG)
```
입력: person_tokens (b*t, N, D)
  → (b*N, t, D) reshape
  → PyG 변환: 동일 인물의 프레임 간 완전 연결 (complete graph over t)
     x: (b*N*t, D), edge_index: 각 인물의 t 프레임 간 undirected 완전 연결
     batch: (b*N*t,)

GATv2Conv (undirected, 같은 사람의 프레임 간)
  ← 기존 people_temporal self-attention (complete graph)과 동일한 연결 방식

출력: (b*t, N, D)  — dense 복원

※ train: N 고정(4) → edge_index 고정 가능 (캐싱 최적화 가능)
   test:  N 가변 → 매 forward마다 edge_index 재구성 (O(N²), 경량)
```

---

## Null Node GT Supervision 설계 (B1 ✅)

```
hard label:
  w_null GT(i) = 1   if  모든 LAH_gt(i→j) == 0  (또는 -1: 어노테이션 없음)
  w_null GT(i) = 0   if  어떤 j에 대해 LAH_gt(i→j) == 1

losses.py 반영:
  - LAH softmax 출력에서 null에 해당하는 logit을 별도 추출
  - GT null label과 CrossEntropyLoss 또는 BCELoss로 supervision
  - LAH_gt가 전부 -1 (어노테이션 없음)인 경우 null supervision도 skip
```

---

## D3. gaze_vec 전달 경로 (✅)

`mtgs_net.py` forward 기준:
- `gaze_vec`: line 244에서 `(b*t, n, 2)` 생성 → line 247에서 `(b, t, n, -1)` reshape
- `num_valid`: line 266에서 이미 `(b*t,)`로 준비됨
- `head_bboxes`: `x["head_bboxes"].view(b*t, n, -1)` — 루프 외부에서 미리 선언하거나 inline

루프 내 SocialGraphBlock 호출:
```python
# 루프 진입 전 (line ~272 부근)
head_bboxes_bt = x["head_bboxes"].view(b * t, n, -1)  # (b*t, n, 4) — 루프 밖에서 한 번만

# 루프 내 (기존 people_interaction 호출 자리)
if self.use_graph:
    person_tokens, lah_logits_i, sa_logits_i = self.social_graph_blocks[i](
        person_tokens,                    # (b*t, N, D)
        gaze_vec.view(b * t, n, -1),      # (b*t, N, 2)  — reshape inline
        head_bboxes_bt,                   # (b*t, N, 4)
        num_valid,                        # (b*t,)
    )
    if i == len(self.vit_adaptor) - 1:
        lah_from_graph = lah_logits_i     # 마지막 블록만 저장
        sa_from_graph  = sa_logits_i
```

---

## 수정할 파일 목록

```
mtgs/config/config.yaml          ← interaction 설정 추가
mtgs/networks/adaptor_modules.py ← SocialGraphBlock, TemporalGraphBlock 추가
mtgs/networks/mtgs_net.py        ← __init__ 및 forward 조건 분기 추가
mtgs/networks/models.py          ← LAH/SA prediction 소스 조건 분기, configure_optimizers 수정
mtgs/train/losses.py             ← null node supervision 추가
```

---

## 파일별 구체적 작업

---

### 1. `config.yaml` 수정

`model:` 블록 아래에 다음을 추가한다:

```yaml
interaction:
  type: transformer   # "transformer" (baseline) | "graph" (신규)
  graph:
    num_layers: 2         # SocialGraphBlock 내 반복 정제 횟수 ✅
    hidden_channels: 96   # GCN hidden dim
    heads: 8              # GATv2Conv attention heads
    use_null_node: true   # null node 사용 여부 (ablation E2용, 설정으로 on/off)
    use_gaze_prior: true  # gaze_vec으로 directed edge 초기화 (ablation E1용)
    prior_weight: 0.5     # geometric prior 반영 강도
```

> **주의**: `interaction.type: transformer`가 default.

---

### 2. `adaptor_modules.py` 수정

기존 코드는 **일절 건드리지 않는다.** 파일 맨 아래에 새 클래스 두 개만 추가한다.

#### 2-1. `SocialGraphBlock` 추가 (PyG MessagePassing 기반)

```python
from torch_geometric.nn import MessagePassing
import torch_geometric.utils as pyg_utils

class SocialGraphBlock(MessagePassing):
    """
    Directed graph block for spatial social gaze interaction. (PyG sparse)
    Replaces people_interaction (TransformerBlock) in graph mode.

    - Directed edges (i→j): LAH  — softmax over {j1,...,jN-1, null}
    - Undirected edges (i,j): SA — pair별 독립 sigmoid, null 미참여
    - Null node: learnable nn.Parameter, directed sink only
    - Prior: gaze_vec geometric prior, 첫 번째 iteration에만 적용 ✅
    - num_layers=2 반복 정제 ✅
    """
    def __init__(self, token_dim, hidden_channels, heads,
                 num_layers, use_null_node, use_gaze_prior, prior_weight):
        super().__init__(aggr='add', flow='source_to_target')
        self.num_layers = num_layers
        self.use_null_node = use_null_node
        self.use_gaze_prior = use_gaze_prior
        self.prior_weight = prior_weight

        # null node: learnable parameter ✅
        if use_null_node:
            self.null_node_h = nn.Parameter(torch.zeros(token_dim))

        # MLPs for edge logits (shared across num_layers)
        self.mlp_dir  = MLP(token_dim * 2, hidden_channels, 1)
        self.mlp_sa   = MLP(token_dim,     hidden_channels, 1)
        if use_null_node:
            self.mlp_null = MLP(token_dim, hidden_channels, 1)

        # message projection
        self.W_msg = nn.Linear(token_dim, token_dim, bias=False)

        # node update
        self.update_proj = nn.Linear(token_dim * 2, token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, person_tokens, gaze_vec, head_bboxes, num_valid):
        """
        person_tokens: (b*t, N, D)
        gaze_vec:      (b*t, N, 2)
        head_bboxes:   (b*t, N, 4)
        num_valid:     (b*t,)  — 유효 인물 수 per frame
        반환:
          person_tokens_out: (b*t, N, D)
          lah_logits:        (b*t, N*(N-1))
          sa_logits:         (b*t, N*(N-1))
        """
        bt, N, D = person_tokens.shape

        # --- Dense → PyG 변환 ---
        # 유효 노드만 추출하여 batch, x, edge_index 구성
        x_list, batch_list, gv_list, bb_list = [], [], [], []
        edge_index_list, edge_offset = [], 0
        for i in range(bt):
            nv = int(num_valid[i].item())
            x_list.append(person_tokens[i, :nv])          # (nv, D)
            batch_list.append(torch.full((nv,), i, device=person_tokens.device))
            gv_list.append(gaze_vec[i, :nv])
            bb_list.append(head_bboxes[i, :nv])
            # directed 완전 연결 (nv*(nv-1)개)
            src, dst = [], []
            for s in range(nv):
                for d in range(nv):
                    if s != d:
                        src.append(s + edge_offset)
                        dst.append(d + edge_offset)
            edge_index_list.append(torch.tensor([src, dst], device=person_tokens.device))
            edge_offset += nv

        x          = torch.cat(x_list, dim=0)          # (total_nodes, D)
        batch      = torch.cat(batch_list, dim=0)
        gv         = torch.cat(gv_list, dim=0)
        bb         = torch.cat(bb_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)  # (2, total_edges)

        # geometric prior (첫 번째 iteration에만 적용) ✅
        if self.use_gaze_prior:
            centers = (bb[:, :2] + bb[:, 2:]) / 2       # (total_nodes, 2)
            src_idx, dst_idx = edge_index
            dir_vec = centers[dst_idx] - centers[src_idx]
            dir_vec = F.normalize(dir_vec, dim=-1)
            prior = (gv[src_idx] * dir_vec).sum(dim=-1)  # cosine sim (total_edges,)

        # --- 반복 정제 ---
        for layer_idx in range(self.num_layers):
            src_idx, dst_idx = edge_index
            hi, hj = x[src_idx], x[dst_idx]

            # directed edge logit
            e_dir = self.mlp_dir(torch.cat([hi, hj], dim=-1)).squeeze(-1)  # (total_edges,)
            if self.use_gaze_prior and layer_idx == 0:   # 첫 번째 iteration에만 ✅
                e_dir = e_dir + self.prior_weight * prior

            # SA edge logit
            e_sa = self.mlp_sa(hi + hj).squeeze(-1)     # (total_edges,)

            # null node logit
            if self.use_null_node:
                e_null = self.mlp_null(x).squeeze(-1)   # (total_nodes,)

            # softmax (directed: per-source, null 포함)
            # pyg_utils.softmax: scatter softmax over dst → src 기준으로 해야 하므로 src_idx 기준
            if self.use_null_node:
                # e_dir과 e_null을 합쳐 src 기준 softmax
                # [구현 상세는 scatter_softmax 커스텀 필요]
                w_dir = self._softmax_with_null(e_dir, e_null, src_idx, x.shape[0])
            else:
                w_dir = pyg_utils.softmax(e_dir, src_idx, num_nodes=x.shape[0])

            # SA: 독립 sigmoid
            w_sa = torch.sigmoid(e_sa)

            # message passing
            msg = self.propagate(edge_index, x=x, w=w_dir)  # (total_nodes, D)
            if self.use_null_node:
                null_contrib = (w_dir  # null weight per node, 별도 scatter 필요)
                # [구현 상세 TODO]
                msg = msg + null_contrib * self.null_node_h

            # node update
            x = self.norm(x + self.update_proj(torch.cat([x, msg], dim=-1)))

        # --- PyG → Dense 복원 ---
        person_tokens_out = torch.zeros_like(person_tokens)
        lah_logits  = torch.full((bt, N * (N - 1)), float('-inf'), device=person_tokens.device)
        sa_logits   = torch.full((bt, N * (N - 1)), 0.0, device=person_tokens.device)

        node_offset, edge_offset = 0, 0
        for i in range(bt):
            nv = int(num_valid[i].item())
            person_tokens_out[i, :nv] = x[node_offset:node_offset + nv]
            # lah/sa logits: nv*(nv-1)개의 directed pair → N*(N-1) 슬롯에 매핑
            ne = nv * (nv - 1)
            lah_logits[i, :ne] = e_dir[edge_offset:edge_offset + ne]  # 마지막 iter 기준
            sa_logits[i, :ne]  = e_sa[edge_offset:edge_offset + ne]
            node_offset += nv
            edge_offset += ne

        return person_tokens_out, lah_logits, sa_logits

    def message(self, x_j, w):
        return w.unsqueeze(-1) * self.W_msg(x_j)
```

> **구현 시 주의**: null node softmax 통합 부분(`_softmax_with_null`)은 `scatter_softmax`를 커스텀하거나  
> null logit을 가상의 edge로 추가하는 방식으로 구현. 상세는 구현 단계에서 결정.

#### 2-2. `TemporalGraphBlock` 추가 (GATv2Conv, PyG)

```python
from torch_geometric.nn import GATv2Conv

class TemporalGraphBlock(nn.Module):
    """
    Temporal graph block for intra-person cross-frame interaction. (GATv2Conv)
    Replaces people_temporal (TransformerBlock) in graph mode.

    동일 인물의 t개 프레임 노드를 undirected 완전 연결 → GATv2Conv
    기존 people_temporal self-attention (complete graph)과 동일한 연결 방식.
    """
    def __init__(self, token_dim, hidden_channels, heads):
        super().__init__()
        self.conv = GATv2Conv(token_dim, token_dim // heads, heads=heads,
                              concat=True, add_self_loops=True)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, person_tokens):
        """
        person_tokens: (b*t, N, D)
        반환:          (b*t, N, D)
        """
        b_t, N, D = person_tokens.shape
        # (b*t, N, D) → (b*N, t, D): 같은 사람, 다른 프레임 묶기
        # b*t = b × t → b, t 분리가 필요하나 b를 모르면 불가 → forward에서 b, t 전달 필요
        # 또는 호출 시 person_tokens.view(b, t, N, D).permute(0,2,1,3).reshape(b*N, t, D)로 변환 후 입력
        # [호출부에서 변환 후 전달 방식으로 통일]

        x = person_tokens  # (b*N, t, D) 로 입력받는다고 가정
        b_n, t, D = x.shape

        # complete graph over t: 모든 프레임 쌍 연결
        src, dst = [], []
        for ti in range(t):
            for tj in range(t):
                if ti != tj:
                    src.append(ti)
                    dst.append(tj)
        # 각 person에 대해 같은 edge_index 반복 (batch offset 추가)
        edge_index_single = torch.tensor([src, dst], device=x.device)
        # [b_n개 그래프 → PyG batch 구성 후 GATv2Conv 적용]
        # 구현 상세: scatter 기반 batching (torch_geometric.data.Batch 활용)

        x_flat = x.reshape(b_n * t, D)
        # ... PyG batching ...
        out = self.conv(x_flat, edge_index)  # (b_n*t, D)
        out = self.norm(x_flat + out)
        return out.view(b_n, t, D)
```

---

### 3. `mtgs_net.py` 수정

#### 3-1. import 추가

```python
from mtgs.networks.adaptor_modules import InteractionBlock, SocialGraphBlock, TemporalGraphBlock
```

#### 3-2. `MTGS.__init__()` 수정

```python
# 기존 코드 (유지)
self.people_interaction = nn.Sequential(...)
self.people_temporal    = nn.Sequential(...)

# 추가: graph 모드 초기화
self.use_graph = (cfg.interaction.type == "graph")
if self.use_graph:
    g = cfg.interaction.graph
    self.social_graph_blocks = nn.ModuleList([
        SocialGraphBlock(
            token_dim=token_dim,
            hidden_channels=g.hidden_channels,
            heads=g.heads,
            num_layers=g.num_layers,          # 2 ✅
            use_null_node=g.use_null_node,
            use_gaze_prior=g.use_gaze_prior,
            prior_weight=g.prior_weight,
        )
        for _ in range(len(self.interaction_indexes))
    ])
    self.temporal_graph_blocks = nn.ModuleList([
        TemporalGraphBlock(
            token_dim=token_dim,
            hidden_channels=g.hidden_channels,
            heads=g.heads,
        )
        for _ in range(len(self.interaction_indexes))
    ])
    self.inout_decoder_graph = InOutDecoder(token_dim)   # 별도 decoder ✅ (D1)
```

#### 3-3. `MTGS.forward()` 수정

```python
# 루프 진입 전
head_bboxes_bt = x["head_bboxes"].view(b * t, n, -1)   # (b*t, N, 4) — 한 번만 계산 (D3)

for i, layer in enumerate(self.vit_adaptor):
    ...
    image_tokens, person_tokens = layer(...)

    if self.use_graph:
        person_tokens, lah_logits_i, sa_logits_i = self.social_graph_blocks[i](
            person_tokens,
            gaze_vec.view(b * t, n, -1),      # inline reshape (D3) ✅
            head_bboxes_bt,
            num_valid,
        )
        if t > 1:
            pt_reshaped = person_tokens.view(b, t, n, -1).permute(0,2,1,3).reshape(b*n, t, -1)
            pt_reshaped = self.temporal_graph_blocks[i](pt_reshaped)
            person_tokens = pt_reshaped.view(b, n, t, -1).permute(0,2,1,3).reshape(b*t, n, -1)
        if i == len(self.vit_adaptor) - 1:       # 마지막 블록만 저장 (D4) ✅
            lah_from_graph = lah_logits_i
            sa_from_graph  = sa_logits_i
    else:
        # 기존 코드 (변경 없음)
        person_tokens = self.people_interaction[i](person_tokens, key_padding_mask=person_pad_mask)
        if t > 1:
            person_tokens = self.people_temporal[i](...)

    img_layers.append(image_tokens[:, self.num_prefix_tokens:])
    gaze_layers.append(person_tokens)
```

#### 3-4. `MTGS.forward()` 반환값 분기

```python
# inout 예측
if self.use_graph:
    inout = self.inout_decoder_graph(gaze_layers[-1].view(b * t * n, -1))  # D1 ✅
else:
    person_tokens_flat = torch.cat([self.gaze_projs[i](gl) for i, gl in enumerate(gaze_layers)], dim=-1)
    inout = self.inout_decoder(person_tokens_flat.view(b * t * n, -1))

# social gaze 예측
if self.use_graph:
    lah   = lah_from_graph    # (b*t, N*(N-1)) — raw logit, sigmoid는 models.py에서
    coatt = sa_from_graph     # (b*t, N*(N-1))
    # LAEO: min(lah(i→j), lah(j→i)) 유지 (B2) ✅
    lah_sig = torch.sigmoid(lah)
    laeo = ...  # harmonic mean 기존 코드 재사용
else:
    # 기존 Pairwise Instance Generator + decoder 경로 (변경 없음)
    ...
```

---

### 4. `models.py` 수정

#### configure_optimizers 분기 (C3 ✅)

```python
if self.model.use_graph:
    high_lr_params = [
        {"params": self.model.social_graph_blocks.parameters(), "lr": base_lr * 3},
        {"params": self.model.temporal_graph_blocks.parameters(), "lr": base_lr * 3},
        {"params": self.model.inout_decoder_graph.parameters(), "lr": base_lr * 3},
    ]
else:
    high_lr_params = [
        {"params": self.model.decoder_sa.parameters(), "lr": base_lr * 3},
        ...  # 기존 코드 (변경 없음)
    ]
```

---

### 5. `losses.py` 수정 (null node supervision, B1 ✅)

```python
# null node GT:
#   w_null GT(i) = 1  if 모든 LAH_gt(i→j) == 0 또는 -1
#   w_null GT(i) = 0  if 어떤 j에 대해 LAH_gt(i→j) == 1
#   LAH_gt 전부 -1 (어노테이션 없음)인 frame → null supervision skip

def compute_null_node_loss(null_logits, lah_gt):
    # null_logits: (b*t, N)  — e_null(i) per person
    # lah_gt:      (b*t, N*(N-1))
    ...
```

---

## 학습 전략 (2-Stage)

### Stage 1 — GazeFollow (Transformer mode): 스킵 ✅

기존 체크포인트 사용:
```
/home/jinwoongjung/MTGS/experiments/2026-05-16/MTGS-dinov2-vitb14-448/train/checkpoints/best.ckpt
```

### Stage 2 — VSGaze (Graph mode): Fine-tuning

```bash
python ./main.py \
  experiment.task=train+test+metrics \
  experiment.dataset=vsgaze \
  model.weights=<Stage1_ckpt_path> \
  interaction.type=graph
```

**체크포인트 로딩 (strict=False 기적용):**

| 모듈 | Stage 1 ckpt | Graph 모델 | 결과 |
|------|-------------|-----------|------|
| encoder, gaze_encoder, vit_adaptor, decoders | ✅ | ✅ | Stage 1 weights 로드 |
| people_interaction, people_temporal | ✅ | ✅ (유지됨) | 로드되나 forward 미사용 |
| social_graph_blocks | ❌ | ✅ | **random init** |
| temporal_graph_blocks | ❌ | ✅ | **random init** |
| inout_decoder_graph | ❌ | ✅ | **random init** |

```python
# 로딩 시 missing key 확인 권장
incompatible = self.model.load_state_dict(model_weights, strict=False)
if incompatible.missing_keys:
    logger.info(f"Randomly initialized: {incompatible.missing_keys}")
```

---

## 실험 설계 (E1, E2 — 전부 열어둠 ✅)

config 조합으로 제어 가능:

| 실험 | interaction.type | use_gaze_prior | use_null_node |
|------|-----------------|---------------|--------------|
| baseline | transformer | — | — |
| graph (prior X, null X) | graph | false | false |
| graph (prior O, null X) | graph | true | false |
| graph (prior X, null O) | graph | false | true |
| graph (prior O, null O) | graph | true | true |

---

## 구현 순서

```
Step 1  config.yaml에 interaction 블록 추가 (transformer default)
Step 2  adaptor_modules.py에 SocialGraphBlock 추가 (PyG MessagePassing)
Step 3  adaptor_modules.py에 TemporalGraphBlock 추가 (GATv2Conv)
Step 4  mtgs_net.py __init__ 분기 추가
Step 5  mtgs_net.py forward 분기 추가 (D3 gaze_vec 경로 포함)
Step 6  models.py configure_optimizers 분기 추가
Step 7  losses.py null node supervision 추가
Step 8  interaction.type=transformer 재실행 → 기존과 동일 결과 확인 (회귀 테스트)
Step 9  interaction.type=graph 로 VSGaze fine-tuning → 수렴 확인
Step 10 ablation (E1, E2 조합)
```

---

## 변경하면 안 되는 것

- `InteractionBlock`, `Extractor`, `Injector`, `CrossAttention`, `MLP` — 건드리지 않음
- `people_interaction`, `people_temporal` 선언 — 삭제하지 않고 유지 (transformer 모드에서 사용)
- `decoder_lah`, `decoder_sa`, `decoder_laeo` — 삭제하지 않음 (transformer 모드에서 사용)
- `compute_social_loss` — 변경 없음 (null node loss는 별도 추가)
- 기존 모든 데이터셋 / 학습 루프 / 메트릭 코드 — 변경 없음

---

## [복구용] 이전 SocialGraphBlock 구현 (커스텀 directed GAT)

> **복구 방법**: 아래 코드 블록 전체를 `adaptor_modules.py`의 현재 `SocialGraphBlock` 클래스와 교체하면 됨.
> `mtgs_net.py`의 instantiation 인자(`hidden_channels`, `heads`, `use_null_node`, `use_gaze_prior`, `prior_weight`, `aggr`)도 그대로 사용 가능.

**특징**: outgoing directed GAT + LAH cosine prior (iter 0에만) + null node + valid node masking

```python
class SocialGraphBlock(nn.Module):
    """
    Social interaction graph block replacing I^b_pp (Social Encoder).

    Supports three aggregation modes (controlled by `aggr`):

      "outgoing" (default):
          msg_i = Σ_{i→j} α[i→j] · W_msg(h_j)  +  α_null[i] · W_msg(null_node)
          softmax over destinations j (dim=-1).
          i collects info from nodes it is looking at.
          Null node: "i looks at no person" → null feature absorbed into msg_i.

      "ingoing":
          msg_i = Σ_{j→i} α[j→i] · W_msg(h_j)
          softmax over sources j (dim=1).
          i collects info from nodes looking at it.
          Null node disabled (null has no meaningful gaze direction as a source).

      "both":
          msg_i = W_out(msg_out_i) + W_in(msg_in_i)
          outgoing part includes null; ingoing part does not.

    Geometric LAH cosine prior is injected into attention weights on iteration 0 only.
    Social prediction (LAH/SA) is handled by the shared pair-wise decoder downstream.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_channels: int = 96,
        heads: int = 8,           # unused; kept for API compatibility
        num_layers: int = 2,      # internal message-passing iterations
        use_null_node: bool = True,
        use_gaze_prior: bool = True,
        prior_weight: float = 0.5,
        layer_idx: int = 0,       # unused; kept for API compatibility
        aggr: str = "outgoing",   # "outgoing" | "ingoing" | "both"
    ):
        super().__init__()
        assert aggr in ("outgoing", "ingoing", "both"), f"Unknown aggr: {aggr!r}"
        self.num_layers     = num_layers
        self.use_gaze_prior = use_gaze_prior
        self.aggr           = aggr

        # Null node is only meaningful in outgoing direction.
        self.use_null_node = use_null_node and (aggr in ("outgoing", "both"))

        # Learnable prior weight for attention routing only (single scalar).
        self.prior_w_attn = nn.Parameter(torch.tensor(prior_weight))

        # ── Attention scoring MLP (directed edge i→j) ───────────────────────
        self.mlp_dir = MLP(token_dim * 2, hidden_channels, 1)
        if self.use_null_node:
            self.null_node = nn.Parameter(torch.zeros(token_dim))
            self.mlp_null  = MLP(token_dim, hidden_channels, 1)

        # ── Message passing & node update ────────────────────────────────────
        self.W_msg       = nn.Linear(token_dim, token_dim, bias=False)
        if aggr == "both":
            # separate projections to combine outgoing and ingoing messages
            self.W_combine_out = nn.Linear(token_dim, token_dim, bias=False)
            self.W_combine_in  = nn.Linear(token_dim, token_dim, bias=False)
        self.update_proj = nn.Linear(token_dim * 2, token_dim)
        self.norm        = nn.LayerNorm(token_dim)

        self._edge_cache: dict = {}

    @staticmethod
    def _build_edges(nv: int, device: torch.device):
        """Directed edges in GT label order: [(s,d) for s in range(nv) for d in range(nv) if s!=d]."""
        src = torch.tensor(
            [s for s in range(nv) for d in range(nv) if s != d],
            dtype=torch.long, device=device,
        )
        dst = torch.tensor(
            [d for s in range(nv) for d in range(nv) if s != d],
            dtype=torch.long, device=device,
        )
        return src, dst

    def _get_edge_cache(self, N: int, device: torch.device) -> dict:
        if N not in self._edge_cache:
            src_N, dst_N = self._build_edges(N, device)
            self._edge_cache[N] = {"src_N": src_N, "dst_N": dst_N}
        return self._edge_cache[N]

    def forward(
        self,
        person_tokens,
        num_valid_people,
        gaze_vecs=None,
        head_bboxes=None,
        readout=False,   # unused; kept for call-site compatibility
    ):
        """
        Args:
            person_tokens:    (B, N, D)
            num_valid_people: (B,) int
            gaze_vecs:        (B, N, 2) unit gaze direction
            head_bboxes:      (B, N, 4) normalized [x1,y1,x2,y2]

        Returns:
            tokens_out: (B, N, D)  updated node features only.
                        Social prediction (LAH/SA) is handled by the shared
                        pair-wise decoder downstream, same as transformer mode.
        """
        B, N, D = person_tokens.shape
        device  = person_tokens.device
        dtype   = person_tokens.dtype

        cache  = self._get_edge_cache(N, device)
        src_N  = cache["src_N"]   # (E,)  E = N*(N-1)
        dst_N  = cache["dst_N"]   # (E,)

        # Valid nodes occupy the BACK slots [N-nv .. N-1]; front slots are padding.
        node_valid = (
            torch.arange(N, device=device).unsqueeze(0) >= (N - num_valid_people.unsqueeze(1))
        )  # (B, N)
        pair_valid = node_valid.unsqueeze(2) & node_valid.unsqueeze(1)   # (B, N, N)
        diag_mask  = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)

        # ── Pre-compute LAH cosine prior (iteration-independent) ─────────────
        lah_prior = None
        if self.use_gaze_prior and gaze_vecs is not None and head_bboxes is not None:
            centers = (head_bboxes[..., :2] + head_bboxes[..., 2:]) / 2   # (B, N, 2)
            dir_ij    = F.normalize(centers[:, dst_N] - centers[:, src_N], dim=-1)
            lah_prior = (gaze_vecs[:, src_N] * dir_ij).sum(-1)            # (B, E)

        h = person_tokens.clone()

        for iter_idx in range(self.num_layers):
            # ── Directed edge attention scores ───────────────────────────────
            h_i = h.unsqueeze(2).expand(B, N, N, D)
            h_j = h.unsqueeze(1).expand(B, N, N, D)

            e_dir_mat = self.mlp_dir(
                torch.cat([h_i, h_j], dim=-1).reshape(B * N * N, 2 * D)
            ).reshape(B, N, N)
            e_dir_mat = e_dir_mat.masked_fill(diag_mask,   float("-inf"))
            e_dir_mat = e_dir_mat.masked_fill(~pair_valid, float("-inf"))

            # Inject LAH cosine prior into attention on iteration 0 only
            if self.use_gaze_prior and lah_prior is not None and iter_idx == 0:
                lah_prior_mat = torch.zeros(B, N, N, device=device, dtype=dtype)
                lah_prior_mat[:, src_N, dst_N] = lah_prior.to(dtype)
                e_dir_mat = e_dir_mat + self.prior_w_attn * lah_prior_mat

            W_msg_h = self.W_msg(h)   # (B, N, D)

            # ── Outgoing aggregation: i collects from nodes it looks at ──────
            # α_out[i,j]: softmax over destinations j (dim=-1)
            # msg_out_i = Σ_j α_out[i→j] · W_msg(h_j)
            if self.aggr in ("outgoing", "both"):
                if self.use_null_node:
                    e_null = self.mlp_null(h.reshape(B * N, D)).reshape(B, N)
                    e_null = e_null.masked_fill(~node_valid, float("-inf"))
                    e_aug_out = torch.cat([e_dir_mat, e_null.unsqueeze(-1)], dim=-1)  # (B,N,N+1)
                else:
                    e_aug_out = e_dir_mat
                all_inf_out = e_aug_out.isinf().all(dim=-1, keepdim=True)
                e_aug_out   = e_aug_out.masked_fill(all_inf_out, 0.0)
                alpha_out   = torch.softmax(e_aug_out, dim=-1)   # (B, N, N[+1])
                msg_out = torch.einsum("bij,bjd->bid", alpha_out[:, :, :N], W_msg_h)
                if self.use_null_node:
                    alpha_null = alpha_out[:, :, N]   # (B, N)
                    msg_out = msg_out + alpha_null.unsqueeze(-1) * self.W_msg(self.null_node).to(dtype)

            # ── Ingoing aggregation: i collects from nodes looking at it ─────
            # α_in[i,j]: softmax over sources j (dim=1), treating e_dir_mat[j,i] as score of j→i
            # msg_in_i = Σ_j α_in[j→i] · W_msg(h_j)
            if self.aggr in ("ingoing", "both"):
                # mask invalid pairs before softmax (same pair_valid, diag already -inf)
                e_in = e_dir_mat.masked_fill(e_dir_mat.isinf().all(dim=1, keepdim=True), 0.0)
                alpha_in = torch.softmax(e_in, dim=1)   # (B, N, N): softmax over source dim
                # alpha_in[b, j, i] = how much j contributes to i
                msg_in = torch.einsum("bji,bjd->bid", alpha_in, W_msg_h)

            # ── Combine ───────────────────────────────────────────────────────
            if self.aggr == "outgoing":
                msg = msg_out
            elif self.aggr == "ingoing":
                msg = msg_in
            else:  # both
                msg = self.W_combine_out(msg_out) + self.W_combine_in(msg_in)

            # ── Node update ──────────────────────────────────────────────────
            h_new = self.update_proj(torch.cat([h, msg], dim=-1))
            h_new = self.norm(h + h_new).to(dtype)
            h = torch.where(node_valid.unsqueeze(-1), h_new, h)

        return h.float()
```
