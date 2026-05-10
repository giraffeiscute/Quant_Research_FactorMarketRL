# FactorMarketRL

<a name="繁體中文"></a>
## 專案簡介 (Introduction)

FactorMarketRL 是一個用於量化投資組合研究的模擬實驗專案。專案先建立一套可控、可檢查的 synthetic factor market，再使用 portfolio attention model 學習股票與現金部位的配置權重。

本專案的核心目的不是直接回測真實市場資料，也不是建立可直接上線的交易系統，而是透過白箱資料生成流程觀察：

* **模型是否能學出合理配置**：在已知 data generating process 下，檢查模型是否能根據 factor、characteristic 與 price path 做出合理權重。
* **不同 loss 的配置差異**：比較 return、Sharpe、Sortino、MDD、CVaR 等目標函數對投資組合行為的影響。
* **不同市場狀態的穩健性**：分別在 bear、neutral、bull 三種 market regime 下觀察模型表現。
* **模型結構與訓練設計的效果**：透過 ablation 比較 turnover penalty、prev-weight feedback、transaction cost 與 sample size 對績效的影響。

目前 repo 由三個層次組成：

| Module | 角色 | 說明 |
| :--- | :--- | :--- |
| **mt toy model** | 早期 baseline | 使用人工生成的股票 embedding、報酬與波動率，測試不同 loss function 下的靜態投組配置。 |
| **toy_ff_generator** | 資料生成器 | 生成 FF-style 三因子、latent characteristics、個股 beta、alpha、epsilon、return 與 price。 |
| **portfolio_attention** | 投組模型 | 讀取 scenario parquet，建立 scenario-aware dataset，訓練 long-only 且允許現金部位的 attention-based portfolio model。 |

---

## 專案結構 (Project Structure)

本倉庫的檔案結構組織如下：

```text
.
├── README.md                         # 專案總覽與實驗結果整理
├── requirements.txt                  # 根目錄最小依賴
├── mt toy model/                     # 早期 toy portfolio optimization baseline
│   ├── README.md
│   ├── Toy_Model_Loss_Simulation.py  # baseline 主入口
│   ├── market_environment.py         # 人工市場環境
│   ├── portfolio_model.py            # 簡單 portfolio model
│   └── performance_metrics.py        # 績效指標計算
├── toy_ff_generator/                 # synthetic factor market generator
│   ├── pyproject.toml
│   ├── generator_README.md           # generator 詳細說明
│   ├── src_architecture.md           # generator 架構文件
│   ├── src/toy_ff_generator/         # generator 原始碼
│   └── tests/                        # generator 測試
└── portfolio_attention/              # portfolio attention training pipeline
    ├── pyproject.toml
    ├── profolio_attention_README.md  # portfolio model 詳細說明
    ├── src_architecture.md           # portfolio_attention 架構文件
    ├── ablations/                    # ablation YAML 設定
    ├── src/portfolio_attention/      # portfolio model、dataset、training、evaluation 原始碼
    ├── outputs/                      # 實驗輸出與 ablation 結果
    └── tests/                        # portfolio_attention 測試
```

---

## 方法論 (Methodology)

### 1. 整體框架總覽

本專案分成兩個主要階段：第一階段用 `toy_ff_generator` 建立 synthetic market scenario；第二階段用 `portfolio_attention` 讀取 scenario parquet，訓練投資組合模型。

```text
market regime S_t
  -> FF factors X_t = [MKT_t, SMB_t, HML_t]
  -> stock latent state Z_{i,t}
  -> observable characteristics + factor exposure beta_{i,t}
  -> stock return r_{i,t} and price P_{i,t}
  -> scenario parquet
  -> scenario-aware dataset [scenario, time, stock]
  -> Portfolio Attention Model
  -> stock weights + cash weight
  -> portfolio return, loss, holdout backtest
```

換句話說，`toy_ff_generator` 負責「市場與股票資料如何被生成」，`portfolio_attention` 負責「模型如何讀取這些資料並產生投資組合配置」。

### 2. Synthetic Factor Market DGP

`toy_ff_generator` 先定義每個時間點的市場狀態：

$$
S_1, \ldots, S_T,\qquad S_t \in \{-1,0,1\}
$$

其中 $S_t=-1$ 代表 bear market，$S_t=0$ 代表 neutral market，$S_t=1$ 代表 bull market。市場狀態可以手動指定，也可以由 Markov transition matrix 產生。

接著將三個 FF-style factor 寫成向量：

$$
\mathbf{X}_t = [MKT_t, SMB_t, HML_t]^T
$$

並用 regime-dependent vector AR(1) 生成：

$$
\mathbf{X}_t=\Phi \mathbf{X}_{t-1}+\mu_X(S_t)+\mathbf{u}_t
$$

$$
\mathbf{u}_t \sim N(0,\Sigma_X(S_t))
$$

這裡 $\Phi$ 控制 factor persistence，$\mu_X(S_t)$ 讓不同市場狀態有不同 factor mean，$\Sigma_X(S_t)$ 則讓 bear、neutral、bull 擁有不同 covariance 結構。

### 3. 個股 Latent State、Beta、Return 與 Price

對每支股票 $i$，generator 會建立三維 latent characteristic state：

$$
\mathbf{Z}_{i,t}=
\begin{bmatrix}
Z_{i,t}^{(1)}\\
Z_{i,t}^{(2)}\\
Z_{i,t}^{(3)}
\end{bmatrix}
$$

目前主要使用 per-stock parameter 版本：

$$
\mathbf{Z}_{i,t}=\boldsymbol{\Omega}_i \odot \mathbf{Z}_{i,t-1}+\boldsymbol{\mu}_i+\boldsymbol{\lambda}_i S_t+\boldsymbol{\xi}_{i,t}
$$

$$
\boldsymbol{\xi}_{i,t}\sim N(0,\Sigma_{Z,i})
$$

observable characteristics 目前是白箱對應，不做額外非線性轉換：

$$
\text{characteristic}_1 = Z_{i,t}^{(1)},\qquad
\text{characteristic}_2 = Z_{i,t}^{(2)},\qquad
\text{characteristic}_3 = Z_{i,t}^{(3)}
$$

factor exposure 由 latent state 線性映射而來：

$$
\boldsymbol{\beta}_{i,t}=A \mathbf{Z}_{i,t}+b
$$

其中 $\boldsymbol{\beta}_{i,t}$ 對應到程式中的 `beta_mkt`、`beta_smb`、`beta_hml`。

個股固定效果與 idiosyncratic noise 分別為：

$$
\alpha_i = \text{alpha\_level}(\text{alpha\_group}_i)
$$

$$
\varepsilon_{i,t}\sim N(0,\sigma_{\varepsilon,i}^2)
$$

最後生成股票報酬：

$$
r_{i,t}
=
\alpha_i
+
\beta_{i,t,1}MKT_t
+
\beta_{i,t,2}SMB_t
+
\beta_{i,t,3}HML_t
+
\varepsilon_{i,t}
$$

實作上會先對報酬做漲跌幅裁切：

$$
r_{i,t}^{obs}
:=
\operatorname{clip}(r_{i,t},\text{limit\_down},\text{limit\_up})
$$

再遞推價格：

$$
P_{i,t}:=P_{i,t-1}(1+r_{i,t}^{obs})
$$

這個階段輸出的 scenario parquet 會成為後續模型訓練資料。

### 4. Scenario-Aware Dataset Construction

`portfolio_attention` 的資料單位不是單一 row，而是一條完整 scenario path。模型會保留三個維度：

| 維度 | 記號 | 說明 |
| :--- | :--- | :--- |
| Scenario / batch | $S$ | train 時是一批 rolling windows，validation / test 時是一批 scenarios。 |
| Time | $T$ | context window 或 full scenario 的可對齊時間長度。 |
| Stock | $N$ | 固定股票 universe。 |

train rolling window 的 context 長度為：

$$
T_{\mathrm{ctx}}^{train}
=
\texttt{lookback\_days}
+
\texttt{rolling\_horizon\_days}
$$

validation / test 不切短 window，而是保留整段 scenario 可對齊時間：

$$
T_{\mathrm{ctx}}^{val/test}=T_{\mathrm{raw}}-1
$$

模型 `forward` 的主要輸入張量如下：

| Tensor | Shape | 說明 |
| :--- | :--- | :--- |
| `x_stock` | `[S, T, N, F_stock]` | 個股特徵，包含三個 characteristic 與 price。 |
| `x_market` | `[S, T, F_market]` | 市場特徵，包含 `MKT`、`SMB`、`HML`。 |
| `stock_indices` | `[S, N]` | 股票身份索引，用於 embedding 或 Gaussian code。 |
| `target_returns` | `[S, T, N]` | 對齊到下一期的股票報酬。 |

其中：

$$
F_{\mathrm{stock}}=4,\qquad F_{\mathrm{market}}=3
$$

資料處理上，price 可以先做 `relative_to_anchor`，也就是把每個 context 內的價格轉成相對於起點的變化率；標準化統計量只用 train scenarios 估計，避免 validation / test leakage。

### 5. Portfolio Attention Model Framework

模型的任務是在每個 scenario $s$、時間點 $t$，對 $N$ 支股票與現金部位產生配置權重。

第一步，模型分別處理個股特徵與市場特徵：

| 分支 | 輸入 | 作用 |
| :--- | :--- | :--- |
| Stock branch | `characteristic_1`, `characteristic_2`, `characteristic_3`, `price` | 學習每支股票自身的時間訊號。 |
| Market branch | `MKT`, `SMB`, `HML` | 提供當前市場 regime 與 factor context。 |
| Identity branch | `stock_indices` | 提供股票身份訊號，避免模型只依賴當期特徵。 |

股票 temporal encoder 可以使用 running summary 或 causal self-attention。若使用 causal self-attention，對每支股票沿時間維度做注意力：

$$
H^{stock,tmp}_{s,i}
:=
\mathrm{softmax}\!\left(
\frac{Q_{s,i}K_{s,i}^{\top}}{\sqrt{d_t}}+M
\right)V_{s,i}
$$

其中 $M$ 是 causal mask，確保時間 $t$ 只能看見當前與過去資訊，不會使用未來資料。

模型之後會保留四塊內容訊號：

$$
c^{raw}_{s,t,i}
:=
\big[
h^{stock,cur}_{s,t,i};
h^{stock,sum}_{s,t,i};
h^{market,ctx}_{s,t};
h^{market,sum}_{s,t}
\big]
$$

這裡的含義是：一支股票在時間 $t$ 的打分，不只看自己的當前特徵，也看自己的歷史摘要、市場當前狀態與市場歷史摘要。

股票 identity 向量記為：

$$
e^{id}_i \in \mathbb{R}^{d_{id}}
$$

它可以是 learnable embedding，也可以是 fixed Gaussian code。若使用 concat 模式，模型會把內容訊號與身份訊號接在一起：

$$
\big[c^{raw}_{s,t,i};e^{id}_i\big]
$$

cross-sectional encoder 則決定如何在同一時間點比較不同股票：

| Encoder | 說明 |
| :--- | :--- |
| `mlp` | 每支股票各自打分，現金部位使用 pooled context 打分。 |
| `self_attention` | 先建立股票內容表示，再對同一時間點的股票集合做 cross-sectional self-attention。 |

最後，模型輸出所有股票 logits 與 cash logit：

$$
\tilde{\ell}_{s,t}
:=
\big[
\ell^{stock}_{s,t,1},\dots,\ell^{stock}_{s,t,N},
\ell^{cash}_{s,t}
\big]
$$

再透過 softmax 轉成投組權重：

$$
\big[
w_{s,t,1},\dots,w_{s,t,N},w^{cash}_{s,t}
\big]
:=
\mathrm{softmax}(\tilde{\ell}_{s,t})
$$

因此配置滿足：

$$
\sum_{i=1}^{N}w_{s,t,i}+w^{cash}_{s,t}=1
$$

$$
w_{s,t,i}\ge 0,\qquad w^{cash}_{s,t}\ge 0
$$

也就是 long-only 且允許持有現金。

### 6. Portfolio Return、Score Mask 與 Loss

若提供 `target_returns`，模型會在每個時間點計算投組報酬：

$$
R_{p,s,t}
:=
\sum_{i=1}^{N}w_{s,t,i}r_{s,t,i}
+
w^{cash}_{s,t}\cdot 0
$$

現金部位報酬目前等價於 0。

訓練與回測不一定使用完整 context。模型會先產生完整的 `portfolio_return: [S, T]`，再用 `score_mask` 擷取真正計分的區段：

$$
R^{score}_{p,s,1:T_{score}}
:=
\mathrm{Mask}\!\left(
R_{p,s,1:T},
\texttt{score\_mask}_{s,1:T}
\right)
$$

這代表 warmup 區段只提供歷史資訊，不直接參與 loss；真正進入 objective 的是 scored path。

以 `return_loss` 為例，先計算每條 scored path 的複利終值：

$$
R^{term}_s
:=
\prod_{t=1}^{T_{score}}
\big(1+R^{score}_{p,s,t}\big)
-1
$$

再取負號作為最小化目標：

$$
\mathcal{L}_{return}
:=
-\frac{1}{S}
\sum_{s=1}^{S}
R^{term}_s
$$

以 `sharpe_loss` 為例，先對每條 scored path 計算平均報酬與標準差：

$$
\mu_s
:=
\frac{1}{T_{score}}
\sum_{t=1}^{T_{score}}
R^{score}_{p,s,t}
$$

$$
\sigma_s
:=
\mathrm{Std}\!\left(
R^{score}_{p,s,1:T_{score}}
\right)
$$

再最小化負 Sharpe：

$$
\mathcal{L}_{sharpe}
:=
\frac{1}{S}
\sum_{s=1}^{S}
\left(
-\frac{\mu_s}{\sigma_s+\varepsilon}
\right)
$$

其他 objective 包含 `dsr`、`sortino`、`mdd`、`cvar`，分別對應動態 Sharpe、下行風險、最大回撤與尾部風險。

### 7. Ablation Study

本專案透過 ablation 檢查模型設計是否真的影響配置與回測表現：

| Ablation | 比較重點 |
| :--- | :--- |
| End-to-end vs stop-gradient feedback | 檢查 prev-weight feedback 是否需要讓梯度完整回傳。 |
| No prev-weight feedback | 檢查移除前一期配置資訊後，模型是否仍能穩定配置。 |
| No turnover penalty | 檢查不懲罰換手率時，報酬與 turnover 如何變化。 |
| With transaction cost | 檢查交易成本是否改變最終配置與 regime 表現。 |
| Larger universe / small sample | 檢查股票池大小與抽樣數量對訓練穩定性的影響。 |

---

## 實驗結果 (Results)

實驗結果整理自：

```text
portfolio_attention/outputs/ablation_best_epoch_summary.csv
```

下表比較不同 ablation 設定在 bear、neutral、bull 三種市場狀態下的最佳 epoch 表現。`SR` 為 Sharpe-style ratio，`Return` 為 holdout final return，`TO` 為 average turnover。

| Variant / Method | 表現 (Best Epoch Summary) | 備註 |
| :--- | :--- | :--- |
| **Main** | Bear SR **0.2042**, Neutral SR **0.3068**, Bull SR **2.3052**；Bull Return **3.9215** | End-to-end feedback + turnover penalty 的主要 baseline。 |
| **Stop-gradient feedback** | Bear SR **0.2090**, Neutral SR **0.4201**, Bull SR **2.2875** | Neutral SR 明顯高於 Main，但 bull return 下降至 **0.8834**。 |
| **No turnover penalty** | Neutral SR **0.4231**, Bull SR **2.3504**；Neutral TO **0.4454**, Bull TO **0.2650** | 移除 turnover penalty 後報酬表現改善，但 turnover 明顯升高。 |
| **With transaction cost** | Bear SR **0.2119**, Bear Return **0.1276**, Bull SR **2.3322** | Bear market 防守表現最佳，但 neutral SR 降至 **0.1489**。 |
| **No prev-weight feedback** | Neutral SR **0.4287**, Bull SR **2.3697**, Bull Return **3.4661** | 不使用 prev-weight feedback 時，中性與牛市 SR 均具競爭力。 |
| **No prev-weight feedback + penalty** | Bull SR **2.4017**；Bear SR **-0.0498**, Bear Return **-0.1715** | Bull SR 最高，但 bear market 表現最差，顯示 regime sensitivity 高。 |
| **Larger universe** | Bull Return **4.0010**, Bull SR **2.3412** | Bull return 最高，略高於 Main，但 bear / neutral 優勢不明顯。 |
| **Small sample (200) + larger batch** | Bull SR **2.2710**, Bull Return **2.1898** | 加入 transaction cost 且 sample 較小，整體表現偏保守。 |
| **Small sample (400)** | Bear SR **0.2116**, Neutral SR **0.4351**, Bull Return **3.7216** | Neutral SR 最高，且 bear SR 接近最佳，是目前較均衡的設定。 |

### 各市場狀態最佳表現

| 評估面向 | 最佳設定 | 表現 | 解讀 |
| :--- | :--- | :--- | :--- |
| Bear SR | **With transaction cost** | **0.2119** | 加入交易成本後，模型在 bear market 的風險調整後表現最佳。 |
| Bear Return | **With transaction cost** | **0.1276** | 防守型表現最佳，能在 bear scenario 下取得最高正報酬。 |
| Neutral SR | **Small sample (400)** | **0.4351** | 中性市場下最穩定，風險調整後表現最佳。 |
| Neutral Return | **No turnover penalty** | **0.2553** | 移除 turnover penalty 後 neutral return 最高，但 turnover 成本風險較大。 |
| Bull SR | **No prev-weight feedback + penalty** | **2.4017** | 牛市 SR 最高，但不代表最高累積報酬。 |
| Bull Return | **Larger universe** | **4.0010** | 擴大 universe 後 bull market 累積報酬最高。 |

### 結果分析

1. **沒有單一設定能同時支配三種市場**：不同 ablation 在 bear、neutral、bull 的最佳設定不同，因此不能只看單一總分。
2. **Small sample (400) 是較均衡的設定**：它取得最高 neutral SR，同時 bear SR 也接近最高，bull return 仍維持在高水準。
3. **Transaction cost 改變模型配置行為**：加入 transaction cost 後 bear market 表現改善，但 neutral SR 明顯下降，表示交易成本會改變模型在不同 regime 下的風險報酬平衡。
4. **Turnover penalty 是重要控制項**：移除 turnover penalty 可提升部分市場報酬，但 turnover 會升高，可能造成實務可交易性下降。
5. **Bull market 更偏好較大的投資集合**：`Larger universe` 在 bull return 上表現最佳，顯示牛市中擴大股票池可能提供更多可利用的 alpha / beta exposure。

---

## 安裝與使用 (Getting Started)

### 環境需求

建議使用 Python 3.10 以上。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e toy_ff_generator
python -m pip install -e portfolio_attention
```

若要執行完整 parquet 讀寫、圖表、dashboard 或 Lightning workflow，可能還需要：

```powershell
python -m pip install torch pyarrow rich PyYAML matplotlib
python -m pip install pytorch-lightning torchmetrics
```

### 1. 產生 Synthetic Scenario Data

```powershell
python -m toy_ff_generator.main
```

這會使用 `toy_ff_generator/src/toy_ff_generator/config.py` 的預設設定執行 batch mode。

若只想跑小規模單次模擬：

```powershell
python -c "from toy_ff_generator import run_simulation; run_simulation(N=100, T=50, S=1)"
```

`S` 對應的 market regime：

| S | 市場狀態 | 說明 |
| :--- | :--- | :--- |
| `-1` | `bear` | 熊市 scenario |
| `0` | `neutral` | 中性市場 scenario |
| `1` | `bull` | 牛市 scenario |

### 2. 訓練 Portfolio Attention Model

請先確認對應 state 的 scenario parquet 已存在：

```text
toy_ff_generator/outputs/data v3/<state>/
```

使用 `bull` scenario、`return` loss、較小股票抽樣數與較少 epoch 的 smoke run：

```powershell
python -m portfolio_attention.cli.train --state bull --loss return --sample-num-stocks 200 --num-epochs 5 --device cpu
```

多 loss 訓練範例：

```powershell
python -m portfolio_attention.cli.train --state bull --losses return,sharpe,sortino,mdd --parallel 1 --sample-num-stocks 200
```

### 3. 評估 Checkpoint

訓練完成後，checkpoint 會寫入：

```text
portfolio_attention/outputs/checkpoints/
```

可用 checkpoint 路徑執行評估：

```powershell
python -m portfolio_attention.cli.evaluate --checkpoint portfolio_attention/outputs/checkpoints/<checkpoint_name>.pt --loss return
```

### 4. 執行測試

```powershell
python -m pytest toy_ff_generator/tests
python -m pytest portfolio_attention/tests
```

若尚未做 editable install，可先設定 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = "toy_ff_generator/src;portfolio_attention/src"
python -m pytest toy_ff_generator/tests portfolio_attention/tests
```

---

## 相關文件 (References)

| 文件 | 說明 |
| :--- | :--- |
| `toy_ff_generator/generator_README.md` | synthetic factor market generator 的數學與實作細節。 |
| `toy_ff_generator/src_architecture.md` | generator 原始碼架構與資料流。 |
| `portfolio_attention/profolio_attention_README.md` | portfolio attention model 的問題設定、資料切分、模型與 loss 說明。 |
| `portfolio_attention/src_architecture.md` | portfolio_attention 原始碼架構、artifact 與 pipeline 說明。 |
| `portfolio_attention/outputs/ablation_best_epoch_summary.csv` | ablation best epoch summary 原始結果表。 |

*Last updated: 2026/05*
