# Portfolio Attention 專案邏輯文檔

## 1. 問題設定與基本記號

本專案以 **scenario parquet 檔** 為基本資料單位。每個 scenario 代表一條完整的市場路徑；模型不會把 scenario 維度與時間維度攤平成單一路徑，而是保留：

* batch 維度 $S$
* 時間維度 $T$
* 股票維度 $N$

在下文中，$S$ 表示模型前向傳遞中的 batch 維度：

* 在 train 階段，$S$ 是一個 batch 內的 rolling windows 數量
* 在 validation / test 階段，$S$ 通常也是一個 batch 內的 scenario 數量

### 1.1 原始 scenario 長度與 context 長度

對單一 scenario 而言，先定義：

* 原始時間長度：$T_{\mathrm{raw}} = \texttt{parsed\_t}$
* 股票數量：$N = \texttt{parsed\_n}$，再依 `DataConfig.num_stocks` 取固定 universe

本專案中的時間長度不是固定常數，而是由設定控制。訓練時常用的幾個量為：

* `lookback_days`
* `rolling_horizon_days`
* `rolling_stride_days`

對 train rolling window 而言，context 長度為：

$$
T_{\mathrm{ctx}}^{\mathrm{train}} = \texttt{lookback\_days} + \texttt{rolling\_horizon\_days}
$$

對 validation / test 而言，context 不再切成短 window，而是保留整段 scenario 的可對齊時間：

$$
T_{\mathrm{ctx}}^{\mathrm{val/test}} = T_{\mathrm{raw}} - 1
$$

這裡減去 $1$ 的原因是：模型在時間位置 $t$ 看特徵，對應的 target return 來自下一個時間位置。

### 1.2 目前預設設定

以目前 `config.py` 的預設值來看：

* `state = "neutral"`
* `num_train_scenarios = 74`
* `num_validation_scenarios = 8`
* `num_test_scenarios = 6`
* `lookback_days = 50`
* `rolling_horizon_days = 30`
* `rolling_stride_days = 2`
* `price_normalization_mode = "relative_to_anchor"`
* `num_stocks = 4860`

模型設定的預設值則包括：

* `stock_feature_dim = 4`
* `market_feature_dim = 3`
* `market_temporal_dim = 32`
* `stock_temporal_dim = 64`
* `cross_sectional_dim = 64`
* `stock_id_representation_type = "learning"`
* `stock_id_embedding_dim = 32`
* `stock_embedding_type = "concat"`
* `stock_temporal_encoder_type = "running_summary"`
* `stock_cross_sectional_encoder_type = "mlp"`
* `time_positional_encoding_type = "none"`
* `dropout = 0.0`

---

## 2. 原始資料與欄位定義

### 2.1 Scenario 檔名格式

scenario 檔名格式為：

$$
\{state\}\_\{N\}\_\{T\}\_\mathrm{PL}\_\{scenario\_index\}.parquet
$$

例如：

* `bull_4860_200_PL_17.parquet`
* `neutral_4860_81_PL_3.parquet`

專案會從檔名解析出：

* `state`
* `parsed_n`
* `parsed_t`
* `scenario_index`

### 2.2 必要欄位

每個 scenario 至少需要下列欄位：

* `stock_id`
* `t`
* `characteristic_1`
* `characteristic_2`
* `characteristic_3`
* `price`
* `MKT`
* `SMB`
* `HML`

若檔案中另外有 `return` 欄位，則會直接使用；若沒有，專案會由價格推回一期報酬。

### 2.3 個股特徵與市場特徵

對第 $s$ 個 batch item、時間位置 $t$、股票 $i$，原始個股特徵為：

$$
x^{\mathrm{raw}}_{s,t,i} \in \mathbb{R}^{4}
$$

其四維分別是：

$$
x^{\mathrm{raw}}_{s,t,i}
=
\big[
\text{characteristic}_1,\,
\text{characteristic}_2,\,
\text{characteristic}_3,\,
\text{price}
\big]_{s,t,i}
$$

市場特徵為：

$$
f^{\mathrm{raw}}_{s,t} \in \mathbb{R}^{3}
$$

其中：

$$
f^{\mathrm{raw}}_{s,t} = [MKT,\ SMB,\ HML]_{s,t}
$$

### 2.4 股票排序與時間索引的一致性

所有 scenario 必須共享：

* 相同的股票 universe
* 相同的股票排序
* 相同的時間索引排序

專案會先將 `stock_id` 與時間欄位排序，並將所有 scenario 對齊到同一組 reference grid。若不一致，資料載入會直接報錯。

若 scenario 原始股票數量大於 `DataConfig.num_stocks`，目前做法是保留排序後前 `num_stocks` 支股票，形成固定股票 universe。

### 2.5 一期報酬

若 `return` 欄位不存在，則由價格推回一期報酬：

$$
r^{\mathrm{raw}}_{t,i} = \frac{P_{t,i}}{P_{t-1,i}} - 1,
\qquad t \ge 1
$$

由於第一個原始時間點沒有前一期價格，因此在原始 return 陣列中，第一個位置會以 $0$ 補齊。

---

## 3. 資料切分與樣本構造

### 3.1 Scenario 層級切分

資料首先在 scenario 層級切成：

* train scenarios
* validation scenarios
* test scenarios

這個切分可依 `scenario_split_seed` 決定是否打亂，但不會在同一個 scenario 內切 train / validation / test。

### 3.2 Train rolling windows

對每個 train scenario，專案會在整段 scenario 上建立 rolling windows。

對單一 train window，定義：

* feature 起點：$a$
* feature 終點：$b$

其中：

$$
b-a = T_{\mathrm{ctx}}^{\mathrm{train}}
$$

對應的 feature 與 target 時間索引分別為：

$$
\texttt{feature\_time\_indices}
=
[\tau_a,\tau_{a+1},\dots,\tau_{b-1}]
$$

$$
\texttt{target\_time\_indices}
=
[\tau_{a+1},\tau_{a+2},\dots,\tau_b]
$$

也就是說，feature 與 target 在同一條時間 grid 上相差一個位置。模型在 feature 時間點看到特徵，對應的監督訊號來自下一個時間位置的股票報酬。

### 3.3 Train score mask

train window 的前半段是 warmup lookback，後半段才是實際計分區段。因此 train window 的 `score_mask` 可寫成：

$$
\underbrace{[0,\dots,0]}_{\texttt{lookback\_days}}
\;\Vert\;
\underbrace{[1,\dots,1]}_{\texttt{rolling\_horizon\_days}}
$$

因此 train 的 scored 長度為：

$$
T_{\mathrm{score}}^{\mathrm{train}} = \texttt{rolling\_horizon\_days}
$$

### 3.4 Validation / test full-scenario records

validation 與 test 不再切成短 rolling windows，而是保留整段 scenario 的可對齊區段：

* `x_stock` 使用前 $T_{\mathrm{raw}}-1$ 個特徵時間點
* `r_stock` 使用對齊後的後 $T_{\mathrm{raw}}-1$ 個 target return

此時 `score_mask` 的前 `lookback_days` 個位置為 `False`，其餘為 `True`。因此：

$$
T_{\mathrm{score}}^{\mathrm{val/test}}
=
T_{\mathrm{raw}} - 1 - \texttt{lookback\_days}
$$

這表示 validation / test 的前段只作為 warmup context，真正的 loss 與 backtest 只在 warmup 之後的路徑上計算。

---

## 4. 前處理與標準化

### 4.1 模型實際吃到的不是 raw feature

模型實際使用的不是原始特徵，而是先經過：

1. context 內價格轉換
2. train-only 標準化

之後得到的張量。

### 4.2 Train-only global standardization

本專案的 scaler 只使用 **train scenarios** 的資料來估計，不會用 validation / test rows fit scaler。

對 stock features 與 market features，分別估計：

$$
\mu_{\mathrm{stock}},\ \sigma_{\mathrm{stock}}
$$

$$
\mu_{\mathrm{market}},\ \sigma_{\mathrm{market}}
$$

然後做標準化：

$$
x_{s,t,i} = \frac{\tilde{x}_{s,t,i} - \mu_{\mathrm{stock}}}{\sigma_{\mathrm{stock}}}
$$

$$
f_{s,t} = \frac{f^{\mathrm{raw}}_{s,t} - \mu_{\mathrm{market}}}{\sigma_{\mathrm{market}}}
$$

其中 $\tilde{x}_{s,t,i}$ 表示價格轉換後的 stock features。

### 4.3 價格維度的 `relative_to_anchor` 轉換

當 `price_normalization_mode = "relative_to_anchor"` 時，price 維度先在每個 context 內做相對起點價格轉換。

若某個 context 的第一個價格為 $P^{\mathrm{anchor}}_i$，則該 context 中第 $t$ 個位置的價格特徵改寫為：

$$
\tilde{P}_{t,i}
=
\frac{P_{t,i}}{P^{\mathrm{anchor}}_i} - 1
$$

也就是說，模型不直接看絕對價格，而是看「相對於該 context 起點」的價格變化率。

### 4.4 price 統計量的估計方式

在 `relative_to_anchor` 模式下，price 這一維的均值與標準差，是從所有 train rolling windows 的相對價格值收集而來；而非直接用整段原始價格去估計。

這使得：

* characteristic 類特徵維持 train-only global standardization
* price 維度則先做 context-relative 轉換，再用 train windows 的統計量標準化

---

## 5. 模型輸入與輸出張量形狀

模型 `forward` 的主要輸入與輸出如下：

* `x_stock: [S, T, N, F_stock]`
* `x_market: [S, T, F_market]`
* `stock_indices: [S, N]`
* `target_returns: [S, T, N]`

其中：

* $F_{\mathrm{stock}} = 4$
* $F_{\mathrm{market}} = 3$

模型輸出為：

* `stock_weights: [S, T, N]`
* `cash_weight: [S, T]`
* `stock_logits: [S, T, N]`
* `cash_logit: [S, T]`
* `portfolio_return: [S, T]`，僅在提供 `target_returns` 時返回

`stock_indices` 是股票 universe 中的連續整數索引，用來查詢股票身份特徵。當
`stock_id_representation_type = "learning"` 時，這個查詢來自 `nn.Embedding`；
當 `stock_id_representation_type = "gaussian"` 時，這個查詢來自固定的 Gaussian
random code matrix。

---

## 6. 模型主體

### 6.1 個股輸入投影

先將每支股票的標準化後特徵投影到 hidden space。令：

* $d_t = \texttt{stock\_temporal\_dim}$
* $d_c = \texttt{cross\_sectional\_dim}$

則：

$$
h^{\mathrm{stock,cur}}_{s,t,i} = W_x x_{s,t,i} + b_x
$$

其中：

$$
W_x \in \mathbb{R}^{d_t \times 4},
\qquad
b_x \in \mathbb{R}^{d_t}
$$

因此：

$$
h^{\mathrm{stock,cur}}_{s,t,i} \in \mathbb{R}^{d_t}
$$

### 6.2 市場輸入投影

市場分支也先做線性投影。令：

* $d_m = \texttt{market\_temporal\_dim}$

則：

$$
h^{\mathrm{market,cur}}_{s,t} = W_f f_{s,t} + b_f
$$

其中：

$$
W_f \in \mathbb{R}^{d_m \times 3},
\qquad
b_f \in \mathbb{R}^{d_m}
$$

因此：

$$
h^{\mathrm{market,cur}}_{s,t} \in \mathbb{R}^{d_m}
$$

### 6.3 可選的時間位置編碼

目前 `time_positional_encoding_type` 支援：

* `none`
* `sinusoidal`

舊 checkpoint 中若出現 `running_mean`，目前會在載入時正規化為 `none`。

當模式為 `sinusoidal` 時，模型會顯式加入時間位置編碼：

這裡使用的是標準的 **正弦／餘弦時間編碼**。直觀上可理解為：

* 一部分維度使用 $\sin(\cdot)$
* 另一部分維度使用 $\cos(\cdot)$

因此它不是單一常數偏移，而是隨時間位置變化的 deterministic 編碼；你也可以把它理解成「包含餘弦項的時間位置編碼」。

$$
\tilde{h}^{\mathrm{stock,cur}}_{s,t,i}
=
h^{\mathrm{stock,cur}}_{s,t,i} + p^{\mathrm{stock}}_t
$$

$$
\tilde{h}^{\mathrm{market,cur}}_{s,t}
=
h^{\mathrm{market,cur}}_{s,t} + p^{\mathrm{market}}_t
$$

當模式為 `none` 時，模型不另外加顯式 sinusoidal 向量，而是直接使用線性投影後的表示。

為了讓符號簡潔，下面以 $h^{\mathrm{stock,cur}}_{s,t,i}$ 與 $h^{\mathrm{market,cur}}_{s,t}$ 代表進入下一步的「當前表示」。

### 6.4 個股時間摘要：causal running mean

模型不使用未來資訊。市場分支固定使用因果累積平均，而股票分支則由 `stock_temporal_encoder_type` 控制，可在 baseline 的因果累積摘要與 fixed-window causal self-attention 之間切換。

### 6.5 市場分支

市場分支不做 temporal attention。對市場 FF3 特徵做線性投影後，直接保留當前表示，並以因果累積平均形成市場摘要：

$$
h^{\mathrm{market,ctx}}_{s,t} = h^{\mathrm{market,cur}}_{s,t}
$$

$$
h^{\mathrm{market,sum}}_{s,t}
=
\frac{1}{t}\sum_{u=1}^{t} h^{\mathrm{market,cur}}_{s,u}
$$

因此 market 在目前版本中只扮演輔助上下文，不再作為 temporal encoder 的消融對象。

### 6.6 股票時間編碼器

股票時間分支由 `stock_temporal_encoder_type` 控制，目前支援兩種模式：

* `running_summary`
* `causal_self_attention`

以下令股票分支的基礎輸入序列為：

$$
H^{\mathrm{stock,base}}_{s,i}
=
\begin{bmatrix}
h^{\mathrm{stock,cur}}_{s,1,i} \\
h^{\mathrm{stock,cur}}_{s,2,i} \\
\vdots \\
h^{\mathrm{stock,cur}}_{s,T,i}
\end{bmatrix}
\in \mathbb{R}^{T \times d_t}
$$

其中 $W = \texttt{lookback\_days}$ 表示股票 temporal attention 的固定回看視窗。

實作上，temporal attention 是作用在目前 context 長度為 $T$ 的整段序列上；其中 local causal mask 的形狀為 $M \in \mathbb{R}^{T \times T}$，但每個 query 只能看到最近 $W$ 個位置，這裡的 $W=\texttt{lookback\_days}$ 是固定回看視窗長度。

#### 模式一：`running_summary`

當 `stock_temporal_encoder_type = "running_summary"` 時，股票當前表示保持為線性投影後的表示：

$$
h^{\mathrm{stock,tmp}}_{s,t,i} = h^{\mathrm{stock,cur}}_{s,t,i}
$$

股票時間摘要為：

$$
h^{\mathrm{stock,sum}}_{s,t,i}
=
\frac{1}{t}\sum_{u=1}^{t} h^{\mathrm{stock,cur}}_{s,u,i}
$$

它代表第 $(s,t,i)$ 個位置在當下可觀察到的股票歷史摘要。

#### 模式二：`causal_self_attention`

當 `stock_temporal_encoder_type = "causal_self_attention"` 時，模型會對每支股票各自沿時間維度做單頭 scaled dot-product causal self-attention。

先定義：

$$
Q_{s,i} = H^{\mathrm{stock,base}}_{s,i} W_Q,\qquad
K_{s,i} = H^{\mathrm{stock,base}}_{s,i} W_K,\qquad
V_{s,i} = H^{\mathrm{stock,base}}_{s,i} W_V
$$

其中：

$$
W_Q, W_K, W_V \in \mathbb{R}^{d_t \times d_t}
$$

再建立 local causal mask $M \in \mathbb{R}^{T \times T}$。對第 $t$ 個 query 與第 $u$ 個 key：

$$
M_{t,u}
=
\begin{cases}
0, & \max(1,\, t-W+1) \le u \le t \\
-\infty, & \text{otherwise}
\end{cases}
$$

因此注意力只允許看見「當前與最近 $W$ 天內」的股票歷史資訊，不會看未來位置。

股票 contextualized token 序列為：

$$
H^{\mathrm{stock,tmp}}_{s,i}
=
\mathrm{softmax}\!\left(
\frac{Q_{s,i} K_{s,i}^\top}{\sqrt{d_t}} + M
\right)V_{s,i}
$$

記第 $t$ 個時間位置的輸出為：

$$
h^{\mathrm{stock,tmp}}_{s,t,i} \in \mathbb{R}^{d_t}
$$

接著再對這條股票 contextualized 序列做固定視窗的因果平均，得到股票時間摘要：

$$
h^{\mathrm{stock,sum}}_{s,t,i}
=
\frac{1}{c_t}
\sum_{u=\max(1,\, t-W+1)}^{t}
h^{\mathrm{stock,tmp}}_{s,u,i}
$$

其中：

$$
c_t = t - \max(1,\, t-W+1) + 1
$$

因此，不論使用哪一種股票時間編碼器，後續都會得到兩個股票時間向量：

* 股票時間當前表示 $h^{\mathrm{stock,tmp}}_{s,t,i}$
* 股票時間摘要表示 $h^{\mathrm{stock,sum}}_{s,t,i}$

### 6.7 Stock ID Representation 與 Placement

本專案保留同一組公開介面：

* `stock_id_representation_type`
* `stock_id_embedding_dim`
* `stock_embedding_type`

但股票身份特徵的內部實作可分為兩種模式。以下令：

* 股票數量為 $N$
* 身份向量維度為 $d_{\mathrm{id}} = \texttt{stock\_id\_embedding\_dim}$

#### Representation 模式一：`learning`

當 `stock_id_representation_type = "learning"` 時，模型使用可訓練的 embedding table：

$$
E^{\mathrm{id}} \in \mathbb{R}^{N \times d_{\mathrm{id}}}
$$

對第 $i$ 支股票，其身份向量為：

$$
e^{\mathrm{id}}_i = E^{\mathrm{id}}_{\mathrm{id}(i)}
$$

其中：

$$
e^{\mathrm{id}}_i \in \mathbb{R}^{d_{\mathrm{id}}}
$$

這條路徑對應實作中的 `nn.Embedding`，因此 $E^{\mathrm{id}}$ 會隨訓練更新。

#### Representation 模式二：`gaussian`

當 `stock_id_representation_type = "gaussian"` 時，模型不使用 dense one-hot，
而是改成 **固定 Gaussian random code**。

先對每支股票抽一個 Gaussian 向量：

$$
g_i \sim \mathcal{N}(0, I_{d_{\mathrm{id}}})
$$

再對每一列做 L2 normalization：

$$
r_i = \frac{g_i}{\lVert g_i \rVert_2}
$$

將所有股票的固定 code 疊成矩陣：

$$
R =
\begin{bmatrix}
r_1 \\
r_2 \\
\vdots \\
r_N
\end{bmatrix}
\in \mathbb{R}^{N \times d_{\mathrm{id}}}
$$

forward 時，對第 $i$ 支股票直接以 `stock_id` 索引這個矩陣：

$$
e^{\mathrm{id}}_i = R_{\mathrm{id}(i)}
$$

其中：

$$
e^{\mathrm{id}}_i \in \mathbb{R}^{d_{\mathrm{id}}}
$$

在 PyTorch 實作上，$R$ 會以 `register_buffer(...)` 保存，因此它具有以下性質：

* 不可訓練
* 不會被 optimizer 更新
* 會跟著 checkpoint 一起保存

#### 沿時間維度展開

不論使用 `learning` 或 `gaussian`，最終都會先得到每支股票的身份向量
$e^{\mathrm{id}}_i$，再沿時間維度展開為：

$$
e^{\mathrm{id}}_{s,t,i} \in \mathbb{R}^{d_{\mathrm{id}}}
$$

因此每個時間點、每支股票都能拿到自己的固定身份特徵。

#### 為什麼不用 dense one-hot

若真的將股票身份展成 dense one-hot，對應的身份張量形狀會接近：

$$
[S, T, N, N]
$$

其記憶體量級約為：

$$
\mathcal{O}(S T N^2)
$$

改成固定 Gaussian code 之後，身份張量形狀為：

$$
[S, T, N, d_{\mathrm{id}}]
$$

其記憶體量級變成：

$$
\mathcal{O}(S T N d_{\mathrm{id}})
$$

因為通常 $d_{\mathrm{id}} \ll N$，所以這種做法可以明顯降低記憶體壓力並避免 OOM。

#### Placement 模式一：`concat`

當 `stock_embedding_type = "concat"` 時，股票身份向量不會進入 temporal encoder，
而是保留到後面的 cross-sectional scorer 再使用。

也就是說，股票 temporal encoder 的基礎輸入仍是：

$$
h^{\mathrm{stock,cur}}_{s,t,i} \in \mathbb{R}^{d_t}
$$

之後在股票打分階段，模型才會把 $e^{\mathrm{id}}_{s,t,i}$ 與內容訊號做 concat。

#### Placement 模式二：`pre_temporal`

當 `stock_embedding_type = "pre_temporal"` 時，模型會先把 stock embedding
加到 projected stock current 上，再送進股票 temporal encoder。

若先定義輸入投影後的股票表示為：

$$
h^{\mathrm{stock,cur}}_{s,t,i} \in \mathbb{R}^{d_t}
$$

則 `pre_temporal` 使用的 temporal encoder 輸入為：

$$
\hat{h}^{\mathrm{stock,cur}}_{s,t,i}
=
h^{\mathrm{stock,cur}}_{s,t,i}
+
e^{\mathrm{id}}_{s,t,i}
$$

這裡的加法是 **元素對應的逐元素相加**，因此需要：

$$
d_{\mathrm{id}} = d_t
$$

也就是：

* `stock_id_embedding_dim == stock_temporal_dim`

`pre_temporal` 下，不論身份表示來自 `learning` 或 `gaussian`，這個加法融合都是合法的。

### 6.8 股票內容訊號

stock 分支由 `stock_cross_sectional_encoder_type` 控制，目前支援兩種模式：

* `mlp`
* `self_attention`

不論使用哪一種模式，模型都會先保留目前的四塊內容訊號：

$$
c^{\mathrm{raw}}_{s,t,i}
=
\big[
h^{\mathrm{stock,cur}}_{s,t,i};
h^{\mathrm{stock,sum}}_{s,t,i};
h^{\mathrm{market,ctx}}_{s,t};
h^{\mathrm{market,sum}}_{s,t}
\big]
$$

這四塊訊號都是真正進入 cross-sectional 分支的基礎內容；但是否會先經過額外的線性投影，取決於 `stock_cross_sectional_encoder_type`：

* 當模式為 `mlp` 時，模型不會另外建立中間的 `stock_content`。
* 當模式為 `self_attention` 時，模型才會先將 `c^{\mathrm{raw}}_{s,t,i}` 投影成股票內容表示。

### 6.9 `self_attention` 模式下的股票內容表示

當 `stock_cross_sectional_encoder_type = "self_attention"` 時，模型先透過一個線性投影形成股票內容表示：

$$
u_{s,t,i} = W_u c^{\mathrm{raw}}_{s,t,i} + b_u
$$

其中：

$$
u_{s,t,i} \in \mathbb{R}^{d_c}
$$

### 6.10 Stock ID Representation 的用法

股票身份特徵永遠保留，但它進入模型的時機依 `stock_embedding_type` 而異。

先定義：

$$
e^{\mathrm{id}}_i \in \mathbb{R}^{d_{\mathrm{id}}}
$$

其中 $e^{\mathrm{id}}_i$ 可以來自：

* `learning` 模式下的 learnable embedding table
* `gaussian` 模式下的 fixed Gaussian random code matrix

當 `stock_embedding_type = "concat"` 且 `stock_cross_sectional_encoder_type = "mlp"` 時，
模型直接使用：

$$
\big[c^{\mathrm{raw}}_{s,t,i};e^{\mathrm{id}}_i\big]
$$

作為股票打分 MLP 的輸入。

當 `stock_embedding_type = "concat"` 且
`stock_cross_sectional_encoder_type = "self_attention"` 時，時間位置 $(s,t)$ 的股票表示會做 concat：

$$
\tilde{u}_{s,t,i} = [u_{s,t,i};e^{\mathrm{id}}_i]
$$

因此 attention 維度為：

$$
d_{\mathrm{attn}} = d_c + d_{\mathrm{id}}
$$

當 `stock_embedding_type = "pre_temporal"` 時，$e^{\mathrm{id}}_{s,t,i}$ 已經在 temporal
encoder 之前以逐元素相加的方式融入 $h^{\mathrm{stock,cur}}_{s,t,i}$，因此後續的
cross-sectional scorer 不再額外 concat 一份獨立的 stock identity。

### 6.11 Cross-Sectional Stock Encoder

#### 模式一：`mlp`

當 `stock_cross_sectional_encoder_type = "mlp"` 時，模型保留目前 baseline 做法，直接將

$$
\big[
h^{\mathrm{stock,cur}}_{s,t,i};
h^{\mathrm{stock,sum}}_{s,t,i};
h^{\mathrm{market,ctx}}_{s,t};
h^{\mathrm{market,sum}}_{s,t};
e^{\mathrm{id}}_{s,t,i}
\big]
$$

交給股票打分 MLP，現金部位則使用 pooled stock current / summary 與市場表示的 MLP head。這條路徑不會經過額外的 cross-sectional self-attention，也不會先建立 `u_{s,t,i}`。

#### 模式二：`self_attention`

當 `stock_cross_sectional_encoder_type = "self_attention"` 時，模型會先把 `c^{\mathrm{raw}}_{s,t,i}` 投影成 `u_{s,t,i}`，再與 stock ID representation concat，最後對每個 $(s,t)$ 的股票集合做單頭橫截面 self-attention。

先將所有股票堆疊成：

$$
\tilde{U}_{s,t}
=
\begin{bmatrix}
\tilde{u}_{s,t,1} \\
\tilde{u}_{s,t,2} \\
\vdots \\
\tilde{u}_{s,t,N}
\end{bmatrix}
\in \mathbb{R}^{N \times d_{\mathrm{attn}}}
$$

再計算：

$$
Q^{(c)} = \tilde{U}_{s,t} W_Q^{(c)},\qquad
K^{(c)} = \tilde{U}_{s,t} W_K^{(c)},\qquad
V^{(c)} = \tilde{U}_{s,t} W_V^{(c)}
$$

$$
\hat{U}_{s,t} = \mathrm{softmax}\left(\frac{Q^{(c)}(K^{(c)})^\top}{\sqrt{d_{\mathrm{attn}}}}\right)V^{(c)}
$$

其中第 $i$ 支股票的 attention 後表示記為：

$$
\hat{u}_{s,t,i} \in \mathbb{R}^{d_{\mathrm{attn}}}
$$

### 6.12 股票分數與現金分數

在 `self_attention` 模式下，股票分數由線性 head 產生：

$$
\ell^{\mathrm{stock}}_{s,t,i} = w_{\mathrm{stock}}^\top \hat{u}_{s,t,i} + b_{\mathrm{stock}}
$$

再對股票做橫截面平均：

$$
\bar{u}_{s,t} = \frac{1}{N}\sum_{i=1}^{N}\hat{u}_{s,t,i}
$$

現金分數為：

$$
\ell^{\mathrm{cash}}_{s,t} = w_{\mathrm{cash}}^\top\bar{u}_{s,t} + b_{\mathrm{cash}}
$$

不論使用 `mlp` 或 `self_attention`，最後都會得到：

* 所有股票的 logits
* 一個 cash logit

### 6.13 最終配置

在每個時間位置 $t$，將所有股票 logits 與 cash logit 串接：

$$
\tilde{\ell}_{s,t}
=
\big[
\ell^{\mathrm{stock}}_{s,t,1},\dots,\ell^{\mathrm{stock}}_{s,t,N},
\ell^{\mathrm{cash}}_{s,t}
\big]
$$

再做 softmax：

$$
\big[
w_{s,t,1},\dots,w_{s,t,N},w^{\mathrm{cash}}_{s,t}
\big]
=
\mathrm{softmax}(\tilde{\ell}_{s,t})
$$

因此滿足：

$$
\sum_{i=1}^{N} w_{s,t,i} + w^{\mathrm{cash}}_{s,t} = 1
$$

且：

$$
w_{s,t,i} \ge 0,
\qquad
w^{\mathrm{cash}}_{s,t} \ge 0
$$

這表示模型輸出的是 **long-only 且允許持有現金** 的配置。

---

## 7. 投組報酬與 scored path

### 7.1 逐時間步投組報酬

若提供 `target_returns`，模型會在每個時間位置計算投組報酬：

$$
R_{p,s,t}
=
\sum_{i=1}^{N} w_{s,t,i}\, r_{s,t,i}
$$

其中：

* $w_{s,t,i}$ 是股票權重
* $r_{s,t,i}$ 是對齊到 `target_time_indices` 的股票一期報酬

現金部位在此不額外加上報酬項，等價於採用：

$$
R^{\mathrm{cash}}_{s,t} = 0
$$

因此：

$$
R_{p,s,t}
=
\sum_{i=1}^{N} w_{s,t,i}\, r_{s,t,i}
+ w^{\mathrm{cash}}_{s,t}\cdot 0
$$

### 7.2 `score_mask` 的角色

模型會先產生完整的 `portfolio_return: [S, T]`，但 loss 與 backtest 不一定使用全部時間位置，而是先套用 `score_mask`：

$$
R^{\mathrm{score}}_{p,s,1:T_{\mathrm{score}}}
=
\mathrm{Mask}\!\left(
R_{p,s,1:T},
\texttt{score\_mask}_{s,1:T}
\right)
$$

也就是說：

* warmup 區段只提供 context，不參與計分
* scored 區段才是真正參與 loss 與 holdout backtest 的路徑

---

## 8. 損失函數

### 8.1 Return loss

令某個 batch item 的 scored path 為：

$$
R^{\mathrm{score}}_{p,s,1:T_{\mathrm{score}}}
$$

則其複利終值為：

$$
R^{\mathrm{term}}_s
=
\prod_{t=1}^{T_{\mathrm{score}}}
\big(1 + R^{\mathrm{score}}_{p,s,t}\big)
- 1
$$

目前 `return_loss` 的語意為：

$$
\mathcal{L}_{\mathrm{return}}
=
-\frac{1}{S}
\sum_{s=1}^{S}
R^{\mathrm{term}}_s
$$

也就是：先對每條 scored path 算複利終值，再取平均並加上負號。

### 8.2 Sharpe loss

對每條 scored path，先計算平均報酬與樣本標準差：

$$
\mu_s
=
\frac{1}{T_{\mathrm{score}}}
\sum_{t=1}^{T_{\mathrm{score}}}
R^{\mathrm{score}}_{p,s,t}
$$

$$
\sigma_s
=
\mathrm{Std}\!\left(
R^{\mathrm{score}}_{p,s,1:T_{\mathrm{score}}}
\right)
$$

其中實作使用的是 `torch.std(..., unbiased=True)`，也就是沿時間維度計算樣本標準差。

則 Sharpe-style loss 可寫成：

$$
\mathcal{L}_{\mathrm{sharpe}}
=
\frac{1}{S}
\sum_{s=1}^{S}
\left(
-\frac{\mu_s}{\sigma_s + \varepsilon}
\right)
$$

其中 $\varepsilon > 0$ 為數值穩定項。

若 scored path 太短，或波動太小而無法穩定計算，實作會退回到 terminal return 型目標。

### 8.3 Differential Sharpe Ratio

`dsr` 會沿 scored path 逐步更新一階與二階動差估計，形成 differential Sharpe 訊號，再將時間序列上的分數做聚合後取負值。

它的重點不在單一步的終值，而在於：

* 整條路徑上的動態風險調整報酬
* 對當前一步報酬如何改變長期 Sharpe 狀態的敏感度

### 8.4 Sortino / MDD / CVaR

目前另外支援：

* `sortino`：只懲罰下行波動
* `mdd`：最小化整條 scored path 的最大回撤風險
* `cvar`：聚焦報酬分布尾部的風險

因此目前專案支援的 loss 名稱為：

* `return`
* `sharpe`
* `dsr`
* `sortino`
* `mdd`
* `cvar`

訓練入口也支援 multi-loss workflow；以目前預設設定來看，預設訓練清單同樣是：

* `return`
* `sharpe`
* `dsr`
* `sortino`
* `mdd`
* `cvar`

---

## 9. 訓練、驗證與 Holdout Backtest

### 9.1 訓練流程

train 階段使用 DataLoader 讀取 rolling train dataset。每個 batch item 對應一個 train rolling window，並提供：

* `x_stock`
* `x_market`
* `r_stock`
* `stock_indices`
* `feature_time_indices`
* `target_time_indices`
* `score_mask`

模型先產生完整 `portfolio_return`，再用 `score_mask` 擷取 scored path，最後將其交給對應 loss function。

### 9.2 Validation 流程

validation dataset 會先保留 full-scenario record，但實際算 validation loss 時，並不是對整段 scenario 只 forward 一次；而是對每個 scored day 建立 rolling one-step window。對每個 scenario 而言，流程是：

1. 先用 full-scenario record 提供完整的 `score_mask` 與時間索引
2. 對每個被 `score_mask` 選中的 scored day，切出一個長度為 `lookback_days + 1` 的 rolling window
3. 對每個 rolling window 做前向傳遞，只取最後一天的 `portfolio_return`
4. 將所有 scored day 的 one-step returns 串成 scored path，再計算 validation loss

因此 validation 的資料載體是 full-scenario record，但真正的評分方式是 rolling one-step backtest 式驗證，而不是單次 full-scenario forward。

### 9.3 Holdout evaluation

訓練結束後，專案會使用 best checkpoint 對 test scenarios 執行 holdout evaluation。這一步會：

* 載入 holdout dataset 的 full-scenario records
* 對每個 scored day 以 `lookback_days + 1` 的 rolling one-step window 做前向傳遞
* 逐個 scenario 收集 scored path returns、stock weights、cash weights
* 計算每個 scenario 的 `final_return`
* 匯總整體的平均、標準差、中位數、最佳與最差 scenario 指標

### 9.4 Monitoring holdout backtests

訓練過程中，專案可依固定 epoch 或固定 interval 執行 monitoring holdout backtest。這些 backtests 會保存：

* 監控用 checkpoint
* holdout backtest manifest
* 各 scenario 的 JSON / CSV / PNG artifacts
* 多 loss 的 monitoring overview 圖

對應輸出通常會落在：

* `outputs/predictions/{state}/{epoch}_holdout_backtest/`

---

## 10. 輸出目錄與主要產物

目前 `PathsConfig` 會在 `outputs/` 下管理主要輸出目錄：

* `outputs/checkpoints/`
* `outputs/metrics/`
* `outputs/logs/`
* `outputs/predictions/`
* `outputs/status/`

其中：

* `checkpoints/`：訓練最佳模型、最後模型、monitoring checkpoints
* `metrics/`：train metrics 與 evaluation metrics
* `logs/`：訓練與評估過程的文字記錄
* `predictions/{state}/`：holdout scenario artifacts 與 monitoring backtest 結果
* `status/`：多 loss 訓練的即時狀態 JSON

對單一 holdout scenario 而言，常見輸出欄位包括：

* `final_return`
* `backtest_portfolio_sr`
* `mean_step_return`
* `std_step_return`
* `final_cash_weight`
* `mean_cash_weight`
* `top_k_stock_weights`
* `grouped_allocations`
* `weight_trajectory_chart`

此外，evaluation 也會搭配 source parquet 中的輔助欄位，例如：

* `mu`
* `alpha`
* `epsilon_variance`

來豐富輸出的持股分析資訊。

---

## 11. 直觀解讀

可以把目前模型口語化理解成以下流程：

1. 每支股票先把當前特徵投影到 hidden space。
2. 模型同時保留「當前表示」與「截至目前的因果累積摘要」。
3. 市場 FF3 走固定的線性投影與 causal running summary 分支，形成市場當前表示與市場摘要表示。
4. 每支股票再拿到自己的 stock ID representation；這個表示可能來自 learnable embedding，也可能來自 fixed Gaussian random code。
5. 若 `stock_embedding_type = "pre_temporal"`，模型會先把 stock embedding 與 projected `h^{\mathrm{stock,cur}}` 做逐元素相加，再送入股票時間分支；若是 `concat`，則 temporal 分支先不使用 stock identity。
6. 股票時間分支可選擇 `running_summary` 或 `causal_self_attention`，形成每支股票逐時點的時間表示與時間摘要。
7. `concat` 模式下，模型再把 stock ID representation 接到 cross-sectional scorer；`pre_temporal` 模式下則不再額外 concat。
8. stock 分支可選擇直接用 `mlp` 打分，或先做 `cross-sectional self-attention` 再打分。
9. attention 模式下，cash logit 由 attention 後的股票表示做平均池化而來；baseline 模式則沿用目前的 MLP cash head。
10. 每個時間位置都會對股票與現金一起做 softmax，得到當期配置。
11. 只有 `score_mask` 標記為有效的時間步，才會進入 loss 與 holdout backtest。

---

## 12. 常見誤解

### 常見誤解 1：`parsed_t` 就是模型每次看到的時間長度嗎？

不一定。

* 對 train rolling window，模型看到的是 $T_{\mathrm{ctx}}^{\mathrm{train}} = \texttt{lookback\_days} + \texttt{rolling\_horizon\_days}$
* 對 validation / test，模型看到的是整段可對齊長度 $T_{\mathrm{raw}} - 1$

因此 `parsed_t` 是原始 scenario 的總長度，不等於所有 split 中的實際 context 長度。

### 常見誤解 2：`x_stock` 與 `r_stock` 是同一個時間點嗎？

不是。

它們在同一條排序過的時間 grid 上相差一個位置：

* `x_stock` 對應 `feature_time_indices`
* `r_stock` 對應 `target_time_indices`

也就是說，模型在某個 feature 時間位置看到特徵，監督訊號來自下一個 target 時間位置的報酬。

### 常見誤解 3：模型吃的是 raw feature 嗎？

不是。

模型吃到的是：

* context 內價格先做 `relative_to_anchor` 轉換（若啟用）
* 再用 train-only global statistics 做標準化

之後的張量。

### 常見誤解 4：`score_mask` 只是附帶資訊嗎？

不是。

`score_mask` 直接決定：

* 哪些時間步參與訓練 loss
* 哪些時間步進入 validation / holdout backtest 指標

沒有被 mask 選中的 warmup 區段只提供 context，不直接計分。

### 常見誤解 5：`stock ID embedding` 還在嗎？

有。

更精確地說，現在保留的是 **stock ID representation** 這個介面。

* 當 `stock_id_representation_type = "learning"` 時，模型仍然使用 `stock_id_embedding`
* 當 `stock_id_representation_type = "gaussian"` 時，內部實作是 fixed Gaussian random code，而不是 dense one-hot

不論使用哪一種 representation，最後都會得到同維度的股票身份向量。

* 若 `stock_embedding_type = "concat"`，這個向量會在後段股票打分 head 使用
* 若 `stock_embedding_type = "pre_temporal"`，這個向量會先和 projected `h^{\mathrm{stock,cur}}` 做逐元素相加，再進 temporal encoder

### 常見誤解 6：`portfolio_return` 是單一 holding-period scalar 嗎？

不是。

目前 `portfolio_return` 是一條逐時間步的路徑：

$$
\texttt{portfolio\_return} \in \mathbb{R}^{S \times T}
$$

最終的 `final_return` 則是對 scored path 做複利累積後得到的 terminal return。
