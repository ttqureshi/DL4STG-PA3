# DL4STG — Programming Assignment 3

**Source code:** [https://github.com/ttqureshi/DL4STG-PA3](https://github.com/ttqureshi/DL4STG-PA3)

---

# Question 1 — What does a sum remember?

Notation matches the course graph layer

\[
h_v = \rho\!\left(\sum_{u \in \mathcal{N}(v)} \phi(x_u)\right),
\]

where $\mathcal{N}(v)$ excludes $v$, each neighbour $u$ reports a trend $x_u \in \{-1,0,+1\}$, and $\rho$ may be any nonlinearity applied **after** aggregation (as in a GCN without degree normalisation).

---

## (a) Affine messages $\phi(x) = wx + c$

For affine $\phi$,

\[
\sum_{u \in \mathcal{N}(v)} \phi(x_u)
= \sum_{u \in \mathcal{N}(v)} (wx_u + c)
= w \sum_{u \in \mathcal{N}(v)} x_u + |\mathcal{N}(v)|\,c
= w S_v + n_v c,
\]

with $n_v = |\mathcal{N}(v)|$ and $S_v = \sum_{u \in \mathcal{N}(v)} x_u$.

Therefore the input to $\rho$ is **completely determined by** $(n_v, S_v)$. For fixed trends in the neighbourhood, varying $w$, $c$, or the particular form of $\rho$ cannot change that scalar aggregate—it is always $wS_v + n_vc$. Two neighbourhoods with the same $n_v$ and $S_v$ produce the **same** pre-activation sum, hence the **same** $h_v$ for any fixed $\rho$.

They may still differ in how many neighbours are steady ($x_u=0$). Example with $n_v=5$, $S_v=0$:

| Neighbourhood | Trends | $n_{-}$ | $n_{0}$ (steady) | $n_{+}$ |
|---------------|--------|---------|------------------|---------|
| A | $(-1,-1,0,+1,+1)$ | 2 | **1** | 2 |
| B | $(-1,0,0,0,+1)$ | 1 | **3** | 1 |

Both have $n_v=5$, $S_v=0$, so for every $w,c$,

\[
\sum_{u} \phi(x_u) = 5c \quad\Rightarrow\quad h_v = \rho(5c),
\]

yet A has one steady neighbour and B has three. $\rho$ cannot tell them apart.

---

## (b) Nonlinear scalar encoding (known $K$)

Assume $1 \le n_v \le K$. We choose constant messages

\[
\phi(-1)=\beta^0=1,\quad \phi(0)=\beta,\quad \phi(+1)=\beta^2,\quad \beta = K+1.
\]

Summing over neighbours with counts $(n_-, n_0, n_+)$,

\[
\sum_{u \in \mathcal{N}(v)} \phi(x_u)
= n_- + \beta n_0 + \beta^2 n_+.
\]

Because each count is at most $K$ and $\beta = K+1$, this is a **mixed-radix** expansion in base $\beta$: no carries occur between the three “digits.” The encoded sum **uniquely** determines $(n_-, n_0, n_+)$.

**Recovery**

\[
\begin{aligned}
n_- &= \Big(\sum_u \phi(x_u)\Big) \bmod \beta,\\
t_1 &= \Big\lfloor \sum_u \phi(x_u) / \beta \Big\rfloor,\\
n_0 &= t_1 \bmod \beta,\\
n_+ &= \Big\lfloor t_1 / \beta \Big\rfloor.
\end{aligned}
\]

**Why $\beta$ must be large enough:** if $\beta \le K$, a carry from $n_-$ into $n_0$ would make two different count triples map to the same sum. We need $\beta > K$ (taking $\beta = K+1$ is sufficient).

**NumPy verification ($K=8$):** `src/q1_q2/q1_aggregation.py` enumerates every $(n_-, n_0, n_+)$ with $n_-+n_0+n_+ \in \{1,\ldots,8\}$; all encoded sums are distinct and decode correctly (`verify_place_value_encoding(8)` prints `True`).

**Simpler target — count only non-steady neighbours**

\[
\phi(-1)=1,\quad \phi(0)=0,\quad \phi(+1)=1
\quad\Rightarrow\quad
\sum_{u \in \mathcal{N}(v)} \phi(x_u) = n_- + n_+,
\]

the number of neighbours that are slowing or speeding (not steady). No maximum size is needed beyond non-negativity.

---

## (c) One-hot messages and sum / mean / max

Fixed messages (same for all three methods):

\[
\phi(-1)=(1,0,0),\quad
\phi(0)=(0,1,0),\quad
\phi(+1)=(0,0,1).
\]

Let $(n_-, n_0, n_+)$ be trend counts. Write the aggregated vector as $\mathbf{m}$.

| Aggregator | $\mathbf{m}$ |
|------------|----------------|
| **Sum** | $(n_-, n_0, n_+)$ |
| **Mean** | $\displaystyle\left(\frac{n_-}{n_v}, \frac{n_0}{n_v}, \frac{n_+}{n_v}\right)$ |
| **Max** | $\displaystyle\big(\mathbb{1}[n_->0],\ \mathbb{1}[n_0>0],\ \mathbb{1}[n_+>0]\big)$ |

$\rho$ sees only $\mathbf{m}$ (not $n_v$ unless it is implied).

### Tasks

| Task | Method | Possible? | Reason |
|------|--------|-----------|--------|
| (i) Count steady neighbours | **Sum** | **Yes** | Use coordinate 2: $n_0 = m_2$. |
| (i) Count steady neighbours | Mean | No | e.g. trends $(0,0)$ ($n_0=2$, $n_v=2$) and $(0,0,0,0)$ ($n_0=4$, $n_v=4$) both give $\mathbf{m}=(0,1,0)$ but steady counts 2 vs 4. |
| (i) Count steady neighbours | Max | No | e.g. one steady $(0,0,+1)$ and two steady $(0,0,0,+1,+1)$ both give $\mathbf{m}=(0,1,1)$ but $n_0\in\{1,2\}$. |
| (ii) More than half slowing | **Sum** | **Yes** | $n_-=m_1$, $n_v=m_1+m_2+m_3$; test $m_1 > \tfrac{1}{2}(m_1+m_2+m_3)$. |
| (ii) More than half slowing | Mean | No | $(-1,-1,-1,0,0)$ ($n_-=3$, majority yes) and $(-1,-1,-1,-1,-1)$ ($n_-=5$, majority yes) both give $\mathbf{m}=(1,0,0)$ with $n_v=5$ vs $5$… use $(-1,-1,-1,0,0)$ ($n_-=3>2.5$) vs $(-1,-1,0,0,0)$ ($n_-=2$, not majority): both have $\mathbf{m}=(0.6,0,0.4)$. |
| (ii) More than half slowing | Max | No | $(-1,-1,-1,0,0)$ (majority slowing) and $(-1,-1,0,+1,+1)$ (not majority) both give $\mathbf{m}=(1,1,1)$. |
| (iii) Any neighbour slowing | **Sum** | **Yes** | Test $m_1 > 0$. |
| (iii) Any neighbour slowing | **Mean** | **Yes** | Slowing present iff $m_1 > 0$; if $n_-=0$ then $m_1=0$, else $m_1 \ge 1/n_v > 0$. |
| (iii) Any neighbour slowing | **Max** | **Yes** | Slowing present iff $m_1 = 1$ (first coordinate of max). |

**Example reused for multiple “No” rows:** $(-1,-1,-1,0,0)$ vs $(-1,-1,0,0,0)$ share $\mathbf{m}=(0.6,0,0.4)$ under **Mean** but differ on task (ii) (3 vs 2 slowing; only the first has a slowing majority). For **Sum**, $(-1,-1,0,+1,+1)$ vs $(-1,0,0,0,+1)$ share $(n_v,S_v)=(5,0)$ but differ in $n_0$.

---

# Question 2 — Which way is the slowdown moving?

Forty sensors on a ring, indices $0,\ldots,39$ with wrap-around. Traffic flows so that **backward** motion of the pattern (assignment convention) is

\[
x_i(t+1) = x_{i+1}(t), \qquad \text{one step: }\texttt{np.roll(}x,\,-1\texttt{)}.
\]

A scalar $x_i(t)$ is slowdown at sensor $i$ ($0$ = none).

---

## (a) Symmetric neighbour layer

\[
\hat{x}_i(t+1) = a\, x_i(t) + b\,\big(x_{i-1}(t) + x_{i+1}(t)\big),
\]

same $(a,b)$ at every sensor.

### (a)(i) Isolated spike at sensor 20

Take $x_{20}(0)=1$, all other $x_i(0)=0$. True next snapshot (backward shift):

| Sensor | $x_i(0)$ | $x_i(1)$ true | $\hat{x}_i(1)=a x_i(0)+b(x_{i-1}+x_{i+1})$ |
|--------|----------|---------------|---------------------------------------------|
| 19 | 0 | **1** | $b(x_{18}+x_{20})=b$ |
| 20 | 1 | 0 | $a\cdot 1 = a$ |
| 21 | 0 | **1** | $b(x_{20}+x_{22})=b$ |
| other | 0 | 0 | 0 |

All other sensors: prediction and target are both $0$. Sum of squared errors

\[
\mathrm{SSE}(a,b) = (b-1)^2 + a^2 + (b-1)^2 = a^2 + 2(b-1)^2.
\]

Minimising: $\partial/\partial a = 2a = 0 \Rightarrow a=0$; $\partial/\partial b = 4(b-1)=0 \Rightarrow b=1$.

**Fitted layer:** $\hat{x}_i(t+1) = x_{i-1}(t) + x_{i+1}(t)$. This matches the two neighbours equally but **cannot** realise the one-sided shift $x_i(t+1)=x_{i+1}(t)$ (see symmetry below).

### (a)(ii) Bump rollout and heatmaps

Initial bump $x_i(0)=\exp\big(-(i-20)^2/8\big)$. Roll forward $t=0,\ldots,20$:

- **True dynamics:** $\mathbf{x}(t+1) = \mathrm{roll}(\mathbf{x}(t), -1)$.
- **Symmetric layer** with $a=0,b=1$: each step replaces $\mathbf{x}$ by left neighbour plus right neighbour.

Heatmaps (sensor index horizontal, time vertical, shared colour scale): `report/figures/q2a_bump_heatmaps.png`, generated by `src/q1_q2/q2_ring.py`.

**Comparison:** The true slowdown crest travels **one sensor per step against traffic** with **unchanged shape**. The symmetric layer **diffuses and splits** the bump: energy leaks to both sides of the ring, so the peak smears and does not translate coherently.

**Symmetry:** For an isolated spike at sensor 20, sensors 19 and 21 are mirror images; any layer with equal treatment of $x_{i-1}$ and $x_{i+1}$ and no position/direction ID gives $\hat{x}_{19}=\hat{x}_{21}$. True values are both $1$ here, but at sensor 20 the model predicts $b=1$ while truth is $0$—the model cannot represent directed transport.

---

## (b) Directional weights

\[
\hat{x}_i(t+1) = a\, x_i(t) + w_{-}\, x_{i-1}(t) + w_{+}\, x_{i+1}(t).
\]

Target: $\hat{x}_i = x_{i+1}$ for all snapshots on the ring.

**Choice:** $a=0$, $w_{-}=0$, $w_{+}=1$.

**Substitution:** $\hat{x}_i = x_{i+1}$ exactly, verified numerically in code (`verify_directional_exact()` $\Rightarrow$ max error $0$).

**Why this fixes (a):** The layer is told which neighbour is **ahead** in traffic ($i+1$) vs **behind** ($i-1$). The shift operator is **antisymmetric** in $(x_{i-1}, x_{i+1})$; equal weights only see the **symmetric** part $x_{i-1}+x_{i+1}$.

**1D convolution kernel** (order $[x_{i-1},\, x_i,\, x_{i+1}]$):

\[
\mathbf{k} = [\,w_{-},\ a,\ w_{+}\,] = [\,0,\ 0,\ 1\,].
\]

**Ring boundary:** circular padding / indices modulo 40 so $i\pm 1$ wrap ($0$ left of $0$ is $39$, etc.).

---

## (c) Temporal coefficient with equal spatial weights

\[
\hat{x}_i(t+1) = a\, x_i(t) + b\,\big(x_{i-1}(t) + x_{i+1}(t)\big) + c\, x_i(t-1).
\]

### (c)(i) Backward motion $x_i(t+1)=x_{i+1}(t)$

From one step of the true rule, $x_i(t) = x_{i+1}(t-1)$. Increment $t \mapsto t-1$:

\[
x_i(t-1) = x_{i+1}(t-2) = \cdots = x_{i-1}(t),
\]

using the chain $x_k(t) = x_{k+1}(t-1)$ with $k=i-1$. So **at time $t$ under backward motion**,

\[
x_i(t-1) = x_{i-1}(t).
\]

**Coefficients:** $a=0$, $b=1$, $c=-1$.

**Why they work:**

\[
\begin{aligned}
\hat{x}_i(t+1)
&= x_{i-1}(t) + x_{i+1}(t) - x_i(t-1) \\
&= x_{i-1}(t) + x_{i+1}(t) - x_{i-1}(t) \\
&= x_{i+1}(t),
\end{aligned}
\]

which is the backward-shift target. Code check: `verify_temporal_backward()` → exact (max error $0$).

*(Note: these $(a,b,c)$ differ from part (a)—here $b=1$, $c=-1$ use history to cancel the wrong symmetric part.)*

### (c)(ii) Forward motion $x_i(t+1)=x_{i-1}(t)$

Forward evolution gives $x_i(t)=x_{i-1}(t-1)$, hence

\[
x_i(t-1) = x_{i+1}(t)
\]

(not $x_{i-1}(t)$ as under backward motion).

With the **same** $(a,b,c)=(0,1,-1)$,

\[
\begin{aligned}
\hat{x}_i(t+1)
&= x_{i-1}(t) + x_{i+1}(t) - x_i(t-1) \\
&= x_{i-1}(t) + x_{i+1}(t) - x_{i+1}(t) \\
&= x_{i-1}(t),
\end{aligned}
\]

which is exactly the forward target $x_i(t+1)=x_{i-1}(t)$. The **same coefficients work** for both directions; only the **meaning** of $x_i(t-1)$ in terms of current neighbours changes. `verify_temporal_forward_same_coeffs()` on consistent forward rollouts reports max error $0$.

### (c)(iii) How history breaks directional ambiguity

Spatially, $x_{i-1}+x_{i+1}$ is unchanged if we swap the labels “ahead” and “behind.” The **previous snapshot** is not swapped that way under directed motion: under backward flow, what sensor $i$ saw at $t-1$ is what sensor $i-1$ shows at $t$, giving $x_i(t-1)=x_{i-1}(t)$. Under forward flow, the link is $x_i(t-1)=x_{i+1}(t)$ instead. The term $c\,x_i(t-1)$ therefore carries **which way time evolution ran** between $t-1$ and $t$, even when $b$ weights the two neighbours equally. With $c=-1$, the layer subtracts the “lagged” neighbour contribution and recovers a **one-sided** update.

---

## Code reference

| Script | Purpose |
|--------|---------|
| `src/q1_q2/q1_aggregation.py` | Q1 encoding and aggregators |
| `src/q1_q2/q2_ring.py` | Q2 layer fits, rollouts, heatmaps |
| `src/q1_q2/run_all.py` | One command to reproduce checks |

```bash
cd src/q1_q2 && python run_all.py
```

---

# Question 3 — Dataset Preparation and Graph Construction (LargeST San Diego)

Reference: [LargeST](https://github.com/liuxu77/LargeST) (NeurIPS 2023 DB Track). San Diego (SD) is **District 11** in `ca_meta.csv` (**716** sensors). Sensor order follows `sd_meta.csv` / `ID2` indexing into the CA adjacency matrix.

---

## Q3(a) — Data extraction and resampling

### Procedure

1. Download the CA bundle from [Kaggle — LargeST](https://www.kaggle.com/datasets/liuxu77/largest) into `data/raw/`.
2. Select SD rows with `District == 11`, keep **716** sensor IDs (same order as `ID2` for the graph).
3. Load `ca_his_raw_2020.h5` and `ca_his_raw_2021.h5` (5-minute PeMS flow).
4. Resample to **15 minutes** with `pandas` rule `15min`.
5. **Aggregation choice:** PeMS flow is a **vehicle count per 5-minute interval**. The assignment allows **mean** (average rate over the window) or **sum** (total vehicles in 15 min). We follow the official LargeST `process_ca_his.ipynb`: **mean** of the three 5-min bins, then `round(0)`. This matches the benchmark; **sum** would preserve total count per 15-min window.
6. **Missing values:** Resampling can introduce NaNs when a bin has no observations. We report counts **before** `fillna(0)` (LargeST uses zero imputation for absent flow). Details are in `data/processed/sd_flow_metadata.json` after running the pipeline.

### Outputs

| Artifact | Description |
|----------|-------------|
| `data/processed/flow_15min.npy` | Flow tensor \(X \in \mathbb{R}^{T \times 716}\) |
| `data/processed/manifest.json` | Start time and frequency for Q4 loaders |
| `data/processed/sd_meta.csv` | Sensor metadata (aligned column order) |

### Figure

![Three sensors, one week](../figures/q3/q3a_three_sensors_one_week.png)

*Generated by `python -m src.data.run_q3` after raw data is present. Week shown: 2020-01-06 — 2020-01-13; sensors at indices 0, 100, 200.*

---

## Q3(b) — Traffic graph \(A_{\mathrm{SD}} \in \mathbb{R}^{716 \times 716}\)

### Construction

1. Load `ca_rn_adj.npy` (road-network distances / connectivity for all CA sensors).
2. Submatrix: \(A_{\mathrm{SD}} = A_{\mathrm{CA}}[I, I]\) with \(I =\) `sd_meta.ID2` (same order as flow columns).

### Required statistics

After zeroing the diagonal for counting (LargeST convention), run the pipeline and read `data/processed/sd_adjacency_stats.json`:

- **Nodes:** 716
- **Nonzero entries (off-diagonal, directed count):** reported in JSON
- **Symmetry:** \(A_{\mathrm{SD}}\) should be symmetric because distances are mutual; verified via `np.allclose(A, A.T)`
- **Mean degree:** mean number of neighbors per sensor
- **Sparsity:** \(\frac{\#\{A_{ij} \neq 0\}}{716^2} \times 100\%\)

### Figures

![Adjacency heatmap](../figures/q3/q3b_adjacency_heatmap.png)

![Degree distribution](../figures/q3/q3b_degree_distribution.png)

### Meaning of \(A_{ij} \neq 0\)

A nonzero \(A_{ij}\) means sensors \(i\) and \(j\) are **linked on the road network** within the LargeST construction (distance-based adjacency from the CA matrix), not merely geographically close. The value encodes **road-network distance** (or derived weight), so graph convolution passes information along plausible traffic routes.

---

## Q3(c) — Graph neighborhoods (example sensor)

Example (default: **index 0**, first SD sensor in metadata order). Neighbor lists after running the pipeline: `data/processed/q3c_neighborhood_example.json`.

### One-hop vs two-hop

- **One-hop:** \(\mathcal{N}_1(i) = \{ j : A_{ij} \neq 0 \}\) (direct road-network neighbors).
- **Two-hop:** nodes reachable from \(i\) with a path of **at most two edges**, excluding \(i\) and one-hop nodes.

### Written answers

**Why might neighboring sensors help predict traffic at the selected sensor?**  
Traffic is spatially coupled: congestion, incidents, and demand propagate along highways. Neighbors on the same corridor or at merges often **lead or lag** the target sensor, so their recent flow is predictive.

**What is the difference between one-hop and two-hop information aggregation?**  
**One-hop** layers combine only **direct** neighbors (local corridor effects). **Two-hop** aggregation (one layer on neighbors, or a 2-layer GCN) also uses **indirect** influence (e.g., congestion two links away that will arrive later). Two-hop widens the spatial receptive field but mixes information along longer paths.

**Could too many graph convolution layers hurt performance?**  
Yes. Deep GCNs risk **over-smoothing** (node features become similar), add **noise** from distant irrelevant sensors, and increase parameters / training difficulty. For SD, most useful signal is often **local** (1–2 hops); deeper stacks can dilute temporal detail unless combined with strong temporal modules or skip connections.

---

## How to reproduce

```bash
pip install -r requirements.txt
python -m src.data.download          # checks files; use --kaggle to download
python -m src.data.run_q3
```

Raw files required in `data/raw/`: `ca_meta.csv`, `ca_rn_adj.npy`, `ca_his_raw_2020.h5`, `ca_his_raw_2021.h5`.

---

# Question 4 — Spatio-temporal forecasting on San Diego

We forecast 15-minute PeMS flow on the San Diego graph \(G=(V,E)\) with \(|V|=716\) sensors. Each time step records a scalar flow per node. The road-network adjacency \(A \in \mathbb{R}^{716\times 716}\) from Q3 encodes which pairs of sensors are linked on the network (nonzero off-diagonal entries); we work with nonnegative weights and add self-loops only inside the normalized operator \(\tilde{A}\).

---

## Q4(a) — Sliding windows, splits, and normalization

### Input–output windows

For end index \(t\), the model sees the past \(T_{\mathrm{in}}=8\) steps

\[
X_t \in \mathbb{R}^{T_{\mathrm{in}} \times N \times 1}, \qquad
X_t[\tau, i, 0] = x_i(t - T_{\mathrm{in}} + 1 + \tau),
\]

and predicts the next \(H=4\) steps

\[
Y_t \in \mathbb{R}^{H \times N \times 1}, \qquad
Y_t[h, i, 0] = x_i(t + h).
\]

Windows never cross split boundaries: samples are built inside contiguous time segments per split.

### Chronological splits

| Split | Period | Role |
|-------|--------|------|
| Train | Calendar year 2020 | Fit parameters and normalization |
| Validation | 2021-01 … 2021-03 | Model selection (best checkpoint) |
| Test | 2021-04 … 2021-12 | Held-out metrics for reporting |

### Normalization

Let \(\mu\) and \(\sigma\) be the mean and standard deviation of **all train-split flow values** (scalar, shared across sensors and time). Each value is transformed as \(\hat{x}=(x-\mu)/\sigma\) before batching; predictions are mapped back with \(x=\hat{x}\sigma+\mu\) for MAE and RMSE in original units. Validation and test use the **train** statistics only (no leakage).

Implementation: `src/data/forecast_windows.py` (`ForecastDataModule`, `NormStats`).

---

## Q4(b) — ST-GNN encoder and one graph convolution layer

### Notation

- \(N=716\) nodes, batch size \(B\).
- Normalized adjacency (symmetric, with self-loops):

\[
\tilde{A} = D^{-1/2}(A + I)\,D^{-1/2}, \qquad
D_{ii} = \sum_j (A+I)_{ij}.
\]

### Temporal encoder (per sensor)

Each sensor’s \(T_{\mathrm{in}}\) history is processed independently with 1D convolutions along time, then global average pooling:

\[
Z \in \mathbb{R}^{B \times N \times d}, \quad d=32.
\]

This yields a node embedding \(z_i\) that summarizes recent temporal patterns at sensor \(i\) before any spatial mixing.

### One message-passing layer

A single graph convolution aggregates neighbour embeddings:

\[
H = \mathrm{ReLU}\!\left(\tilde{A}\, Z\, W_g\right), \quad W_g \in \mathbb{R}^{d\times d}.
\]

In matrix form over nodes, row \(i\) of \(\tilde{A}\,Z\) is a weighted sum of messages from \(\mathcal{N}(i)\cup\{i\}\) — standard GCN message passing with fixed \(\tilde{A}\) from the road network.

### Forecast head

A shared linear map per node produces \(H\) future steps: output shape \((B, H, N, 1)\).

Code: `src/models/forecasting.py` (`ForecastingModel`, `TemporalConvEncoder`, `GraphConvLayer`).

---

## Q4(c) — Temporal-only baseline

The baseline uses the **same** temporal encoder and forecast head but **skips** graph convolution (\(H=Z\)). It tests whether spatial coupling on \(G\) improves multi-step flow prediction beyond per-sensor temporal patterns alone.

Config: `configs/temporal_only.yaml`; trained with `python -m src.train --config configs/temporal_only.yaml`.

---

## Training protocol (shared with Q5–Q7)

- Optimizer: Adam, learning rate \(10^{-3}\).
- Loss: L1 (MAE) on **normalized** targets.
- Early selection: checkpoint with lowest validation loss.
- Test metrics: MAE and RMSE on denormalized flow.
- Default seed: 42 (`src/train.py`).

---

# Question 5 — Adjacency variants for message passing

The ST-GNN in Q4(b) uses one layer \(H=\mathrm{ReLU}(\tilde{A} Z W_g)\). Here \(\tilde{A}\) may come from the fixed road graph, a learned graph, or a combination. Notation: \(G=(V,E)\), fixed raw adjacency \(A\) (zero diagonal for storage), degree matrix \(D\) of \(A+I\), and \(\tilde{A}=D^{-1/2}(A+I)D^{-1/2}\).

---

## Q5(a) — Fixed road-network adjacency

The fixed operator reuses Q3’s San Diego submatrix. Before normalization, off-diagonal zeros mean no direct network link. Self-loops are added in \(A+I\); symmetric normalization balances high- and low-degree sensors so message passing does not explode on hubs.

**Config:** `configs/stgnn_fixed.yaml` (`adjacency_mode: fixed`).

At forward time the model reads the buffer `fixed_norm` \(=\tilde{A}\) and never changes it during training (only \(W_g\) and temporal weights learn).

---

## Q5(b) — Adaptive (learned) adjacency

When the road graph is incomplete or non-stationary, we learn a row-stochastic matrix from sensor embeddings \(E_1, E_2 \in \mathbb{R}^{N\times d_e}\) (\(d_e=10\)):

\[
S = \mathrm{ReLU}(E_1 E_2^\top), \qquad
(A_{\mathrm{adaptive}})_{ij} = \mathrm{softmax}_j(S_{ij}).
\]

Each row is a distribution over possible receivers of messages from node \(i\). Optional **top-\(k\)** sparsification (here \(k=20\)) zeros all but the largest \(k\) entries per row and renormalizes the row — gradients flow through the kept entries.

The GCN layer uses \(A_{\mathrm{adaptive}}\) in place of \(\tilde{A}\) (no extra \(D^{-1/2}\) on the adaptive matrix; it is already row-normalized).

**Config:** `configs/stgnn_adaptive.yaml`.

After training, `outputs/stgnn_adaptive/learned_adjacency.npy` stores the final \(A_{\mathrm{adaptive}}\).

---

## Q5(c) — Hybrid adjacency

Hybrid message passing mixes physics-based and learned structure with \(\alpha \in [0,1]\) (default \(\alpha=0.5\)):

\[
A_{\mathrm{hybrid}} = \alpha\,\tilde{A} + (1-\alpha)\,A_{\mathrm{adaptive}},
\]

then each row of \(A_{\mathrm{hybrid}}\) is divided by its row sum so aggregation remains stable.

Intuition: \(\tilde{A}\) anchors propagation to the road network; \(A_{\mathrm{adaptive}}\) can add statistical shortcuts (e.g., correlated corridors not adjacent in \(A\)).

**Config:** `configs/stgnn_hybrid.yaml`.

---

## Model inventory (four runs)

| `run_name` | Spatial module | Adjacency used in \(\tilde{A} Z W_g\) |
|------------|----------------|----------------------------------------|
| `temporal_only` | None | — |
| `stgnn_fixed` | 1× GCN | \(\tilde{A}\) from \(A\) |
| `stgnn_adaptive` | 1× GCN | \(A_{\mathrm{adaptive}}\) |
| `stgnn_hybrid` | 1× GCN | Row-normalized \(A_{\mathrm{hybrid}}\) |

All four share \(T_{\mathrm{in}}=8\), \(H=4\), \(d=32\), batch size 32, 20 epochs, MAE loss, and the Q4 splits/normalization rules.

Implementation: `src/models/adjacency.py` (`GraphAdjacency`, `AdaptiveAdjacency`, `symmetric_normalized_adjacency`).

---

# Question 6 — Experimental comparison and forecast visuals

We compare the four models from Q5 on the **2021 test split** (April–December). Metrics are computed on denormalized flow (train-only \(\mu,\sigma\) from Q4). Lower MAE and RMSE are better.

---

## Q6(a) — Running all models and metrics table

### Procedure

1. Ensure Q3 artifacts exist in `data/processed/`.
2. Run the batch evaluator (trains any config missing `outputs/<run_name>/results.json`):

```bash
python -m src.evaluate_all
```

3. Inspect `outputs/comparison_table.md` and `outputs/comparison.json`.

Each training run writes `results.json` with `test_mae`, `test_rmse`, `train_seconds`, and `seed`. The aggregator sorts rows by test MAE and prints a Markdown table suitable for pasting here.

### Expected comparisons (fill from your run)

| Model | Adjacency | Test MAE | Test RMSE | Train time (s) | Seed |
|-------|-----------|----------|-----------|----------------|------|
| *(run `evaluate_all`)* | | | | | |

**Interpretation prompts**

- Does any ST-GNN beat `temporal_only`? If yes, spatial message passing on \(G\) adds signal beyond per-sensor convolutions.
- Among graph models, does adaptive or hybrid help over fixed \(\tilde{A}\)? Large gains suggest the road graph alone misaligns with flow coupling; small gaps suggest \(A\) already captures useful neighbours.

---

## Q6(b) — Predicted vs actual (three sensors)

We plot one-step-ahead predictions (\(h=1\)) against ground truth on a contiguous slice of the test set for sensor indices **0, 100, and 200** (same convention as Q3 plots).

```bash
python -m src.plot_predictions --config configs/stgnn_hybrid.yaml
```

Outputs:

- `report/figures/q6/predictions_<run_name>_three_sensors.png`
- `report/figures/q6/predictions_<run_name>_meta.json` (window length, sensor indices)

Use `--config` to overlay other checkpoints. The figure shows whether phase and peak timing are captured and where graph-aware models reduce systematic bias on congested sensors.

---

## Q6(c) — Learned vs fixed graph (quantitative)

Quantitative comparison of \(A_{\mathrm{adaptive}}\) against the road network \(A\) is produced by:

```bash
python -m src.analyze_graph --config configs/stgnn_adaptive.yaml --also-hybrid
```

`outputs/graph_analysis.json` reports, per run:

- Mean row entropy of \(A_{\mathrm{adaptive}}\) (concentration of messages).
- Pearson correlation between learned weights and fixed edge weights on pairs where \(A_{ij}>0\).
- Mean top-\(k\) overlap: fraction of each node’s learned top-\(k\) neighbours that are one-hop neighbours in \(A\) (default \(k=20\), matching training sparsity).

See Q7 for heatmaps and discussion of when learned edges diverge from \(\tilde{A}\).

---

# Question 7 — Analysis of learned spatial structure

This section interprets **what** the adaptive and hybrid models learn in the message-passing matrix and how that relates to the road-network graph \(G=(V,E)\) with adjacency \(A\).

---

## Q7(a) — Visualizing learned adjacency

`python -m src.analyze_graph` writes heatmaps under `report/figures/q7/`:

| Figure | Content |
|--------|---------|
| `learned_adjacency` | \(A_{\mathrm{adaptive}}\) (crop of first 128×128 sensors for readability) |
| `fixed_normalized` | \(\tilde{A}=D^{-1/2}(A+I)D^{-1/2}\) on the same crop |
| `difference` | Elementwise learned minus fixed on the crop |

**Reading the plots**

- Bright off-diagonal bands in \(\tilde{A}\) follow sparse road-network links.
- A diffuse or block-structured \(A_{\mathrm{adaptive}}\) suggests the model routes messages to statistically similar sensors, not only physical neighbours.
- The difference map highlights edges the model strengthens (positive) or suppresses (negative) relative to \(\tilde{A}\).

Hybrid models (`stgnn_hybrid`) save the adaptive component separately when present; analysis can be run with `--also-hybrid`.

---

## Q7(b) — Quantitative alignment with \(A\)

From `outputs/graph_analysis.json`:

1. **Correlation on fixed support** — compares \( (A_{\mathrm{adaptive}})_{ij} \) with raw \(A_{ij} \) only where the road graph has an edge. High correlation means learned weights respect distance-based magnitudes; low correlation means the model reweights links while keeping sparse road connectivity as a soft prior (hybrid) or not at all (pure adaptive).

2. **Top-\(k\) overlap with one-hop neighbours** — for each node \(i\), take the \(k\) largest entries in row \(i\) of \(A_{\mathrm{adaptive}}\) and measure how many coincide with \(\{j: A_{ij}>0\}\). Values near 1 indicate the learner mostly redistributes mass among true neighbours; values near 0 indicate **shortcut** edges to non-adjacent sensors (e.g., parallel corridors).

3. **Row entropy** — high entropy: messages spread across many receivers; low entropy: focused routing to a few nodes (often desirable with top-\(k\) sparsity).

Tie these numbers to Q6 test MAE: if adaptive overlap is low **and** MAE improves, statistical shortcuts are useful; if overlap is high but metrics do not improve, extra flexibility may overfit noise.

---

## Q7(c) — Methodology summary (full pipeline)

| Stage | Choice |
|-------|--------|
| Graph | \(G=(V,E)\), \(|V|=716\), \(A\) from LargeST SD submatrix |
| Time resolution | 15 minutes, 2020–2021 |
| Splits | 2020 train; 2021 Q1 val; 2021 Q2–Q4 test |
| Windows | \(T_{\mathrm{in}}=8\), horizon \(H=4\) |
| Normalization | Global train mean/std, applied to all splits |
| Temporal block | 2× Conv1d + time pooling → \(z_i \in \mathbb{R}^d\) |
| Spatial block | One layer \(H=\mathrm{ReLU}(A_{\mathrm{msg}} Z W_g)\) with \(A_{\mathrm{msg}} \in \{\tilde{A}, A_{\mathrm{adaptive}}, A_{\mathrm{hybrid}}\}\) |
| Baseline | Temporal encoder + head only |
| Training | Adam \(10^{-3}\), MAE loss, 20 epochs, seed 42 |

Reproduce end-to-end steps in `scripts/REPRODUCE.md`.

Build the master PDF-ready document:

```bash
python report/build_report.py
```

Output: `report/report.md` (includes GitHub URL placeholder at the top).

---
