1. SocialGraphBlock 수식 (aggr="outgoing", dual null nodes)

  notation: i, j ∈ valid nodes, D = token_dim = 768, 반복 횟수 L=2

  ---
  Step 1 — Directed Edge Score

  $$e_{i \to j} = \text{MLP}_{\text{dir}}\bigl([h_i;\, h_j]\bigr) \in \mathbb{R}$$

  MLP_dir: 2D → 96 → 1 (GELU 활성화)

  마스킹:

  $$e_{i \to j} = -\infty \quad \text{if } i = j \text{ or either node is padding}$$

  iteration 0에만 LAH cosine prior 주입:

  $$\hat{e}_{i \to j}^{(0)} = e_{i \to j} + w_{\text{prior}} \cdot \underbrace{\hat{g}_i \cdot \hat{d}_{i \to j}}_{\text{cosine prior}}, \quad w_{\text{prior}} \in \mathbb{R} \text{ learnable (init 0.5)}$$

  $$\hat{d}_{i \to j} = \frac{c_j - c_i}{\|c_j - c_i\|}, \quad c_k = \frac{\text{bbox}_k^{x_1 y_1} + \text{bbox}_k^{x_2 y_2}}{2}$$

  ---
  Step 2 — Dual Null Node Scores (node-dependent)

  $$e_{i \to \text{null\_in}}  = \text{MLP}_{\text{null\_in}}\bigl([h_i;\, v_{\text{in}}]\bigr)  \in \mathbb{R}, \quad v_{\text{in}}  \in \mathbb{R}^D \text{ learnable (init 0)}$$

  $$e_{i \to \text{null\_out}} = \text{MLP}_{\text{null\_out}}\bigl([h_i;\, v_{\text{out}}]\bigr) \in \mathbb{R}, \quad v_{\text{out}} \in \mathbb{R}^D \text{ learnable (init 0)}$$

  MLP_null_{in/out}: 2D → 96 → 1

  padding node는 −∞로 마스킹.

  ---
  Step 3 — Outgoing Attention & Message

  증강 score 벡터 (N+2차원, 마지막 2개가 null):

  $$\tilde{e}_i = \bigl[\hat{e}_{i \to 1},\; \ldots,\; \hat{e}_{i \to N},\; e_{i \to \text{null\_in}},\; e_{i \to \text{null\_out}}\bigr] \in \mathbb{R}^{N+2}$$

  $$\boldsymbol{\alpha}_i = \text{softmax}(\tilde{e}_i)$$

  분해:

  $$\alpha_{i \to j} = \boldsymbol{\alpha}_i[j], \quad \alpha_{i \to \text{null\_in}} = \boldsymbol{\alpha}_i[N], \quad \alpha_{i \to \text{null\_out}} = \boldsymbol{\alpha}_i[N+1]$$

  outgoing message:

  $$m_i = \sum_{j \neq i} \alpha_{i \to j} \cdot W_{\text{msg}} h_j \;+\; \alpha_{i \to \text{null\_in}} \cdot W_{\text{msg}} v_{\text{in}} \;+\; \alpha_{i \to \text{null\_out}} \cdot W_{\text{msg}} v_{\text{out}}$$

  $W_{\text{msg}} \in \mathbb{R}^{D \times D}$

  ---
  Step 4 — Node Update

  $$g_i = \sigma\bigl(W_{\text{gate}}\, h_i\bigr), \quad W_{\text{gate}} \in \mathbb{R}^{D \times D}$$

  $$h_i^{\text{new}} = \text{LayerNorm}\Bigl(h_i + g_i \odot W_{\text{upd}}\bigl([h_i;\, m_i]\bigr)\Bigr), \quad W_{\text{upd}} \in \mathbb{R}^{2D \times D}$$

  $$h_i \leftarrow \begin{cases} h_i^{\text{new}} & \text{if } i \text{ is valid} \\ h_i & \text{if } i \text{ is padding} \end{cases}$$

  Step 1~4를 L=2 반복. iteration 1부터는 cosine prior 없이 e_{i→j}만 사용.

---

2. TemporalGraphBlock 수식

  notation: B = batch, N = num_people, T = num_frames, D = token_dim = 768, n_h = 8 heads, d_h = D/n_h

  입력 reshape (mtgs_net.py 호출부):

  $$(B, T, N, D) \xrightarrow{\text{permute}} (B, N, T, D) \xrightarrow{\text{reshape}} (B \cdot N,\; T,\; D)$$

  이후 각 사람을 독립적인 길이-T 시퀀스로 처리.

  ---
  Step 1 — Multi-Head Self-Attention over T frames

  입력 $H \in \mathbb{R}^{M \times T \times D}$, $M = B \cdot N$.

  각 head $k$:

  $$Q_k = H W_k^Q, \quad K_k = H W_k^K, \quad V_k = H W_k^V, \quad W_k^{Q,K,V} \in \mathbb{R}^{D \times d_h}$$

  $$\text{head}_k = \text{softmax}\!\left(\frac{Q_k K_k^\top}{\sqrt{d_h}}\right) V_k \in \mathbb{R}^{M \times T \times d_h}$$

  $$\text{MHA}(H) = \bigl[\text{head}_1,\; \ldots,\; \text{head}_{n_h}\bigr] W^O, \quad W^O \in \mathbb{R}^{D \times D}$$

  ---
  Step 2 — Residual + LayerNorm 1

  $$H' = \text{LayerNorm}_1\bigl(H + \text{MHA}(H)\bigr)$$

  ---
  Step 3 — Feed-Forward Network

  $$\text{FFN}(x) = W_2\,\text{GELU}(W_1 x), \quad W_1, W_2 \in \mathbb{R}^{D \times D}$$

  ---
  Step 4 — Residual + LayerNorm 2

  $$H'' = \text{LayerNorm}_2\bigl(H' + \text{FFN}(H')\bigr)$$

  출력 reshape (mtgs_net.py 호출부):

  $$(B \cdot N,\; T,\; D) \xrightarrow{\text{reshape}} (B, N, T, D) \xrightarrow{\text{permute}} (B, T, N, D) \xrightarrow{\text{reshape}} (B \cdot T,\; N,\; D)$$

  T=1이면 self-attention을 건너뛰고 입력을 그대로 반환.
