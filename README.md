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

### 1. Synthetic Factor Market Data Generation

`toy_ff_generator` 負責建立可重現的 synthetic market scenario。它以市場狀態序列為起點，依序生成 FF-style factor、個股 latent characteristic、factor exposure、alpha、epsilon、return 與 price。

主要流程如下：

* **市場狀態**：支援 `bear`、`neutral`、`bull` 三種 regime，可由固定 state sequence 或 Markov transition matrix 產生。
* **三因子生成**：使用 vector AR(1) 生成 `MKT`、`SMB`、`HML`。
* **個股 latent state**：每支股票具有三維 latent characteristic state，並可使用 per-stock 參數。
* **Factor exposure**：使用線性映射 `beta_t = A @ Z_t + b` 生成 `beta_mkt`、`beta_smb`、`beta_hml`。
* **報酬與價格**：以 `alpha + beta * factor + epsilon` 生成報酬，再由報酬遞推價格。

預設輸出位置：

```text
toy_ff_generator/outputs/data v3/<state>/
```

常見輸出檔案：

| Artifact | 說明 | 範例 |
| :--- | :--- | :--- |
| `panel_long` parquet | 每個 scenario 的 long panel 資料 | `bull_4860_200_PL_17.parquet` |
| `market_index_csv` | 由 scenario 聚合出的 market index 表格 | `bull_4860_200_market_index_17.csv` |
| `market_index_png` | market index 視覺化圖 | `summary/bull_4860_200_market_index_17.png` |

### 2. Scenario-Aware Portfolio Dataset

`portfolio_attention` 不會把 scenario 與時間直接攤平成單一路徑，而是保留 `[scenario, time, stock]` 的資料結構。

資料集設計重點如下：

* **Scenario-level split**：先在 scenario 層級切分 train、validation、holdout test。
* **Train rolling windows**：訓練階段使用 rolling windows，並以 lookback 區段提供 warmup context。
* **Validation / test full scenario**：驗證與測試階段保留整段 scenario，只有 warmup 後的 scored path 參與 loss 與 backtest。
* **Train-only standardization**：標準化統計量只使用 train scenarios 估計，避免 validation / test leakage。

模型輸入特徵：

| Feature Group | 欄位 | 說明 |
| :--- | :--- | :--- |
| Stock features | `characteristic_1`, `characteristic_2`, `characteristic_3`, `price` | 個股 characteristic 與價格資訊 |
| Market features | `MKT`, `SMB`, `HML` | FF-style 三因子 |
| Stock identity | learnable embedding 或 Gaussian code | 股票身份訊號 |

### 3. Portfolio Attention Model

模型在每個時間點輸出股票權重與現金權重，配置約束為 long-only 且允許現金部位。

主要設計如下：

* **Temporal encoder**：支援 `running_summary` 與 `causal_self_attention`。
* **Cross-sectional encoder**：支援 `mlp` 與 `self_attention`。
* **Stock identity representation**：支援可訓練 embedding 或固定 Gaussian code。
* **Allocation output**：以 softmax 產生股票權重與 cash weight，使總權重為 1。
* **Risk objective**：支援 `return`、`sharpe`、`dsr`、`sortino`、`mdd`、`cvar`。

### 4. Ablation Study

本專案透過 ablation 比較下列設計：

* **End-to-end feedback vs stop-gradient feedback**
* **是否使用 prev-weight feedback**
* **是否加入 turnover penalty**
* **是否加入 transaction cost**
* **不同股票 universe / sample size 設定**

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
