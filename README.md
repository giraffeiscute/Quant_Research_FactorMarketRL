# FactorMarketRL

FactorMarketRL 是一個用來研究**量化投資組合配置**的模擬實驗專案。專案先用白箱方式生成可控的 synthetic factor market，再訓練 portfolio attention model，學習股票與現金部位的配置權重。

本專案不是直接用真實市場資料做回測，也不是可直接上線的交易系統。它更像是一個可檢查的實驗場，用來觀察：

- 在已知 data generating process 下，模型能否學出合理配置。
- 不同 objective，例如 return、Sharpe、Sortino、MDD、CVaR，如何影響配置行為。
- 同一個模型在 bear、neutral、bull 三種 market regime 下是否穩健。
- turnover penalty、prev-weight feedback、transaction cost、sample size 等設計，是否真的改變模型表現。

## Project Highlights

- **GRPO 強化學習式投組決策。** 運用 GRPO 訓練端到端投資組合決策模型，並結合 Dirichlet 機率式資產配置頭促進策略探索；相較於傳統非隨機策略輸出模型，淨夏普比率提升 **10%**。
- **可控白箱多因子合成市場。** 建立 synthetic factor market，系統性生成不同 market regime 與 factor distribution，用於分析並緩解 Domain Shift，降低模型盲點診斷與架構迭代成本。
- **更接近實務交易的風險報酬設計。** 實作交易成本與換手率正則化機制，並結合 Sharpe、Sortino、最大回撤與 CVaR 等路徑型風險報酬指標，提升策略在真實交易場景下的可執行性。

---

## 1. Project Overview

整個 repo 可以簡化成三個部分：

| Module | Role | Description |
| :--- | :--- | :--- |
| `mt toy model` | Early baseline | 早期靜態投組配置 baseline，用來測試不同 loss function。 |
| `toy_ff_generator` | Data generator | 生成 FF-style factors、latent characteristics、beta、return 與 price。 |
| `portfolio_attention` | Portfolio model | 讀取 scenario parquet，訓練 long-only 且允許現金部位的 attention-based portfolio model。 |

核心流程：

```text
synthetic market scenario
  -> stock-level returns and prices
  -> scenario parquet
  -> portfolio attention model
  -> stock weights + cash weight
  -> portfolio return and ablation results
```

---

## 2. Method Summary

`toy_ff_generator` 會先生成 bear、neutral、bull 三種 market regime 下的 FF-style factors：

```math
S_t \in \{-1, 0, 1\},
\qquad
\mathbf{X}_t = [MKT_t, SMB_t, HML_t]^\top
```

接著，每支股票會有自己的 latent state，並由 latent state 產生 factor exposure。股票報酬由 alpha、factor exposure、market factors 與 idiosyncratic noise 組成：

```math
r_{i,t}
= \alpha_i
+ \beta_{i,t,1}MKT_t
+ \beta_{i,t,2}SMB_t
+ \beta_{i,t,3}HML_t
+ \varepsilon_{i,t}
```

生成出的 scenario parquet 會交給 `portfolio_attention`。模型輸入包含個股特徵、市場特徵、股票身份索引與下一期報酬：

| Tensor | Shape | Description |
| :--- | :--- | :--- |
| `x_stock` | `[S, T, N, F_stock]` | 個股特徵，包含 characteristics 與 price。 |
| `x_market` | `[S, T, F_market]` | 市場特徵，包含 `MKT`、`SMB`、`HML`。 |
| `stock_indices` | `[S, N]` | 股票身份索引，用於 embedding 或 Gaussian code。 |
| `target_returns` | `[S, T, N]` | 對齊到下一期的股票報酬。 |

模型最後輸出股票與現金部位的 portfolio weights：

```math
\mathbf{w}_{s,t}
= \mathrm{softmax}\!\left(\widetilde{\ell}_{s,t}\right)
```

where:

```math
\mathbf{w}_{s,t}
= [w_{s,t,1}, \ldots, w_{s,t,N}, w^{\mathrm{cash}}_{s,t}]
```

因此配置滿足 long-only 且允許現金：

```math
\sum_{i=1}^{N} w_{s,t,i} + w^{cash}_{s,t} = 1,
\qquad
w_{s,t,i} \ge 0,
\qquad
w^{cash}_{s,t} \ge 0
```

Portfolio return 定義為：

```math
R_{p,s,t}
= \sum_{i=1}^{N}w_{s,t,i}r_{s,t,i}
+ w^{cash}_{s,t}\cdot 0
```

目前支援的 objective 包含 `return`、`sharpe`、`dsr`、`sortino`、`mdd` 與 `cvar`。在強化學習版本中，模型進一步採用 GRPO 訓練，並以 Dirichlet 機率式配置頭輸出資產權重分佈，使策略在訓練過程中能進行更充分的 allocation exploration。

---

## 3. Ablation Study

本專案透過 ablation 檢查模型設計是否真的影響配置與回測表現。

| Ablation | What it tests |
| :--- | :--- |
| End-to-end vs stop-gradient feedback | 檢查 prev-weight feedback 是否需要讓梯度完整回傳。 |
| No prev-weight feedback | 檢查移除前一期配置資訊後，模型是否仍能穩定配置。 |
| No turnover penalty | 檢查不懲罰換手率時，報酬與 turnover 如何變化。 |
| With transaction cost | 檢查交易成本是否改變最終配置與 regime 表現。 |
| Larger universe / small sample | 檢查股票池大小與抽樣數量對訓練穩定性的影響。 |

---

## 4. Results Summary


下表比較不同 ablation 設定在 bear、neutral、bull 三種市場狀態下的最佳 epoch 表現。

| Abbreviation | Meaning |
| :--- | :--- |
| `SR` | Sharpe ratio |
| `Return` | Holdout final return |
| `TO` | Average turnover |

### 4.1 Best Epoch Summary

| Variant / Method | Performance | Notes |
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

### 4.2 Best by Market Condition

| Evaluation | Best setting | Result | Interpretation |
| :--- | :--- | :--- | :--- |
| Bear SR | **With transaction cost** | **0.2119** | 加入交易成本後，模型在 bear market 的風險調整後表現最佳。 |
| Bear Return | **With transaction cost** | **0.1276** | 防守型表現最佳，能在 bear scenario 下取得最高正報酬。 |
| Neutral SR | **Small sample (400)** | **0.4351** | 中性市場下最穩定，風險調整後表現最佳。 |
| Neutral Return | **No turnover penalty** | **0.2553** | 移除 turnover penalty 後 neutral return 最高，但 turnover 成本風險較大。 |
| Bull SR | **No prev-weight feedback + penalty** | **2.4017** | 牛市 SR 最高，但不代表最高累積報酬。 |
| Bull Return | **Larger universe** | **4.0010** | 擴大 universe 後 bull market 累積報酬最高。 |

### 4.3 Additional Experimental Highlights

| Experiment / Design | Result | Interpretation |
| :--- | :--- | :--- |
| **GRPO + Dirichlet allocation head** | 淨夏普比率相較於傳統非隨機策略輸出模型提升 **10%**。 | Dirichlet 機率式資產配置頭讓模型在端到端 GRPO 訓練中保留策略探索能力，避免過早收斂到單一 deterministic allocation。 |
| **White-box synthetic factor market** | 可系統性控制 bear、neutral、bull regimes 與 factor distributions。 | 讓實驗能針對 Domain Shift、regime sensitivity 與模型盲點進行診斷，而不只依賴單一真實市場樣本。 |
| **Transaction cost + turnover regularization** | 與 Sharpe、Sortino、MDD、CVaR 等路徑型指標一起納入策略優化與評估流程。 | 不只追求高報酬，也同時檢查換手率、下行風險、最大回撤與尾部風險，讓模型評估更接近實務交易需求。 |

---

## 5. Conclusion and Interpretation

### 5.1 Main Takeaways

1. **沒有單一設定能同時支配三種市場。**  
   不同 ablation 在 bear、neutral、bull 的最佳設定不同，因此不能只看單一總分。

2. **Small sample (400) 是較均衡的設定。**  
   它取得最高 neutral SR，同時 bear SR 也接近最高，bull return 仍維持在高水準。若目標是找一個較穩定的 default setting，這是目前最值得優先考慮的版本。

3. **Transaction cost 會讓模型更保守。**  
   加入 transaction cost 後，bear market 的 SR 與 return 都是最佳，但 neutral SR 明顯下降。這表示交易成本會改變模型在不同 regime 下的風險報酬平衡，尤其會抑制較積極的換倉行為。

4. **Turnover penalty 是重要控制項。**  
   移除 turnover penalty 可提升部分市場報酬，但 turnover 也明顯升高。若未來要把模型往更接近實務交易的設定推進，turnover penalty 或 transaction cost 不應被忽略。

5. **Bull market 更受益於較大的股票池。**  
   `Larger universe` 在 bull return 上表現最佳，顯示牛市中擴大股票池可能提供更多可利用的 alpha / beta exposure。不過它在 bear 與 neutral market 下沒有明顯全面優勢。

### 5.2 Practical Reading of the Results

| Goal | Suggested setting | Reason |
| :--- | :--- | :--- |
| 找整體較均衡的 baseline | **Small sample (400)** | Neutral SR 最高，bear SR 接近最佳，bull return 仍高。 |
| 強化 bear market 防守 | **With transaction cost** | Bear SR 與 Bear Return 都最佳。 |
| 追求 bull market 累積報酬 | **Larger universe** | Bull Return 最高。 |
| 研究 turnover 的影響 | **No turnover penalty** vs **Main** | 可清楚觀察 turnover penalty 對報酬與換手率的影響。 |
| 研究 feedback 是否必要 | **Main** vs **No prev-weight feedback** | 可比較 prev-weight feedback 對不同 regime 的影響。 |

整體而言，實驗結果顯示 portfolio attention model 確實會受到 objective、feedback mechanism、turnover design、transaction cost 與 sample design 影響。進一步加入 GRPO 與 Dirichlet 機率式配置頭後，模型也能透過 stochastic allocation exploration 改善淨夏普表現。這也支持本專案的核心定位：它不是為了追求單一最佳績效，而是用可控的 synthetic market 來檢查模型設計如何改變投組行為，並用交易成本、換手率與多種路徑型風險指標提高實驗對真實交易場景的參考價值。

---

## 6. Getting Started

### Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e toy_ff_generator
python -m pip install -e portfolio_attention
```

### Generate Synthetic Scenario Data

```powershell
python -m toy_ff_generator.main
```

### Train Portfolio Attention Model

```powershell
python -m portfolio_attention.cli.train --state bull --loss return --sample-num-stocks 200 --num-epochs 5 --device cpu
```

Multi-loss training example：

```powershell
python -m portfolio_attention.cli.train --state bull --losses return,sharpe,sortino,mdd --parallel 1 --sample-num-stocks 200
```

### Evaluate Checkpoint

```powershell
python -m portfolio_attention.cli.evaluate --checkpoint portfolio_attention/outputs/checkpoints/<checkpoint_name>.pt --loss return
```

### Run Tests

```powershell
python -m pytest toy_ff_generator/tests portfolio_attention/tests
```
---

_Last updated: 2026/05_
