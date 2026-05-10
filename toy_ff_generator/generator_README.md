# Toy FF Generator

## 目前實作說明

- 目前的 generator 使用 **3 個 latent characteristic 軸**：  
  `latent_characteristic_1_state`、`latent_characteristic_2_state`、`latent_characteristic_3_state`
- observable characteristics 是與這 3 個軸對齊的診斷欄位：  
  `characteristic_1`、`characteristic_2`、`characteristic_3`
- 目前 observable characteristics 直接複製 latent states；程式碼**不會**對 latent states 套用 `exp(...)`
- 曝險映射（exposure mapping）目前採用矩陣形式：  
  `beta_t = A @ Z_t + b`
- 預設設定使用：  
  `A = [[0.05, 0, 0], [0, 0.4, 0], [0, 0, 0.4]]`  
  以及 `b = [1.0, 0.4, 0.0]`
- `mu_i` 是一個 white-box、固定的 stock-level center，由 low / mid / high 三種 class center `{-0.5, 0.0, 0.5}` 生成
- 目前會持久化保存的 artifacts 為：  
  `{state_token}_{N}_{T}_PL.parquet`、  
  `{state_token}_{N}_{T}_market_index.csv`、  
  `summary/{state_token}_{N}_{T}_market_index.png`
- 在 batch mode 下，檔名還會額外附加 `_{dataset_number}`
- 目前的報酬生成方式採用當期對齊（contemporaneously aligned）：  
  `r_{i,t} = alpha_i + beta_{i,t,1} MKT_t + beta_{i,t,2} SMB_t + beta_{i,t,3} HML_t + epsilon_{i,t}`
- `save_outputs(...)` 目前只會寫出上述三種 artifact。雖然回傳的 `output_paths` dict 仍保留 `prices`、`returns`、`metadata`、`excel_workbook` 這些 keys，但它們目前的值都是 `None`；程式也不會輸出 wide `price.csv`、wide `return.csv`、`metadata.json` 或 Excel workbook
- `panel_long_df["epsilon_variance"]` 目前儲存的是來自 `epsilon_levels` 的 epsilon sigma 設定值，而不是平方後的 variance；這個欄位名稱是為了向後相容而保留


這個專案是一個透明、可檢查、可解釋的 toy data generating process（DGP），用來模擬股票報酬與價格。它的用途是做檢查、除錯與實驗，不是實盤交易策略。

# 模型生成流程

這個 toy generator 的核心流程如下。

## Step 1：設定市場狀態序列

先定義整段模擬期間的市場狀態：

$$
S_1, \ldots, S_T,\qquad S_t \in \{-1,0,1\}
$$

其中：

- $S_t=-1$：bear market
- $S_t=0$：neutral market
- $S_t=1$：bull market

目前支援兩種方式：

- 直接手動指定整段 `state_sequence`
- 由 `initial_state` 與 Markov `transition_matrix` 生成 state sequence

---

## Step 2：生成 FF 三因子

三個因子被視為同一個 3 維向量：

$$
\mathbf{X}_t = [MKT_t, SMB_t, HML_t]^T
$$

並使用向量 AR(1)：

$$
\mathbf{X}_t=\Phi \mathbf{X}_{t-1}+\mu_X(S_t)+\mathbf{u}_t
$$

其中：

$$
\mathbf{u}_t \sim N(0,\Sigma_X(S_t))
$$

說明如下：

- $\Phi$ 是 $3\times 3$ 矩陣
- $\mu_X(S_t)$ 是由 regime 決定的長度 3 mean vector
- $S_t$ 會影響因子系統的均值位移
- regime 也會影響 factor innovation covariance
- bear / neutral / bull 可分別使用不同的 covariance matrix

也就是：

$$
\Sigma_X^{bear},\qquad \Sigma_X^{neutral},\qquad \Sigma_X^{bull}
$$

對應關係為：

$$
X_{1,t}=MKT_t,\qquad
X_{2,t}=SMB_t,\qquad
X_{3,t}=HML_t
$$

---

## Step 3：生成個股 latent characteristic state 與 observable characteristic

目前實作中的核心狀態不是 2 維 characteristic，而是 **3 維 latent characteristic state**：

$$
\mathbf{Z}_{i,t}=
\begin{bmatrix}
Z_{i,t}^{(1)}\\
Z_{i,t}^{(2)}\\
Z_{i,t}^{(3)}
\end{bmatrix}
$$

shared 模式下，遞迴形式為：

$$
\mathbf{Z}_{i,t}=\boldsymbol{\mu}+\boldsymbol{\Omega} \odot (\mathbf{Z}_{i,t-1}-\boldsymbol{\mu})+\boldsymbol{\lambda} S_t+\boldsymbol{\xi}_{i,t}
$$

per-stock 模式下，遞迴形式為：

$$
\mathbf{Z}_{i,t}=\boldsymbol{\Omega}_i \odot \mathbf{Z}_{i,t-1}+\boldsymbol{\mu}_i+\boldsymbol{\lambda}_i S_t+\boldsymbol{\xi}_{i,t}
$$

其中：

$$
\boldsymbol{\xi}_{i,t}\sim N(0,\Sigma_{Z,i})
$$

目前實作採逐維獨立的 diagonal covariance，也就是：

$$
\Sigma_{Z,i}=
\operatorname{diag}\left(
(\sigma_{Z,i}^{(1)})^2,
(\sigma_{Z,i}^{(2)})^2,
(\sigma_{Z,i}^{(3)})^2
\right)
$$

這代表：

- latent state 不是 i.i.d. across time
- 每個 stock-time pair 都有一個 3 維 latent state vector
- $\mathbf{Z}_{i,t}$ 依賴前一期的 $\mathbf{Z}_{i,t-1}$
- $\boldsymbol{\Omega}$ / $\boldsymbol{\Omega}_i$ 控制各維 persistence
- 初值 $\mathbf{Z}_0$ / $\mathbf{Z}_{i,0}$ 可手動設定

observable characteristic 目前就是對 latent state 的一對一白箱輸出：

$$
\begin{aligned}
\text{characteristic}_1 &= Z_{i,t}^{(1)} \\
\text{characteristic}_2 &= Z_{i,t}^{(2)} \\
\text{characteristic}_3 &= Z_{i,t}^{(3)}
\end{aligned}
$$

也就是程式中的 `state_to_firm_characteristics(...)` 目前只是把 latent state 複製成 observable characteristic，不做非線性轉換。

目前支援兩種參數模式：

- shared params：
  所有股票共用同一組長度 3 的 `Omega`, `mu_Z`, `lambda_Z`, `sigma_Z`, `Z0`
- per-stock params：
  每支股票各自擁有 shape 為 `(N, 3)` 的 `Omega_i`, `mu_i`, `lambda_i`, `sigma_Z_i`, `Z0_i`

---

## Step 4：由 latent state 生成因子曝險

對每支股票、每個時間點、每個因子曝險，定義：

$$
\boldsymbol{\beta}_{i,t}=A \mathbf{Z}_{i,t}+b
$$

其中：

- $\mathbf{Z}_{i,t}$ 是長度 3 的 latent state vector
- `A` 是 `3 × 3` exposure mapping matrix
- `b` 是長度 3 的 intercept vector

對應程式中的欄位名稱：

- `beta_mkt`
- `beta_smb`
- `beta_hml`

這一版先不引入更複雜的非線性 $g_k(\cdot)$。

---

## Step 5：生成個股固定效果與噪音

目前 `alpha_i` 不是常態抽樣，而是由 group lookup 決定的固定值：

$$
\alpha_i = \text{alpha\_level}(\text{alpha\_group}_i)
$$

預設 group label 為：

- `low`
- `mid`
- `high`

另外生成 idiosyncratic noise：

$$
\varepsilon_{i,t}\sim N(0,\sigma_{\varepsilon,i}^2)
$$

其中 epsilon 目前支援：

- shared `epsilon_group` 對應單一共同 sigma
- per-stock `epsilon_group_i` 對應每支股票固定 sigma group

也就是說，epsilon 是常態抽樣，但其波動參數來自 group mapping，而不是直接手動傳入一個 $\sigma_{\varepsilon,i}$ 向量。

---

## Step 6：生成個股報酬

本版維持當期對齊（contemporaneous alignment）：

$$
r_{i,t}=\alpha_i+\beta_{i,t,1}MKT_t+\beta_{i,t,2}SMB_t+\beta_{i,t,3}HML_t+\varepsilon_{i,t}
$$

也就是：

- factor 的 $t$ 使用 $\mathbf{X}_t$
- characteristic 的 $t$ 使用當期由 $\mathbf{Z}_{i,t}$ 對應出的 observable characteristic
- beta 的 $t$ 使用 $\boldsymbol{\beta}_{i,t}$
- epsilon 的 $t$ 使用 $\varepsilon_{i,t}$

目前不引入 return lag 結構。

在報酬生成後，程式還會進一步：

1. 對 `raw_return` 做 clipping
2. 用 clipping 後的 `return` 遞推價格
3. 把最終結果整理成 long panel
4. 由 `panel_long_df` 再建立 market index csv / png

---

# 需要手動調整的輸入參數

目前主要預設參數集中在：

```text
toy_ff_generator/src/toy_ff_generator/config.py
```

也就是 `build_default_config()`。

`main.py` 則負責：

- 讀入 config
- 套用 `run_simulation(...)` / `run_batch_simulations(...)` 的 override
- 執行生成流程
- 輸出檔案

## (A) 基本維度與市場狀態

- `simulation_setup["N"]`：股票數量
- `simulation_setup["T"]`：時間點數量
- `simulation_setup["random_seed"]`
- `simulation_setup["dataset_count"]`：batch 模式預設資料份數
- `market_state_setup["state_sequence"]`
- `market_state_setup["initial_state"]`
- `market_state_setup["transition_matrix"]`

另外 `run_simulation(...)` / `run_batch_simulations(...)` 也提供幾個直接 override：

- `output_dir`
- `seed`
- `N`
- `T`
- `S`
- `dataset_count`（batch only）

其中 `S` 代表把整段 state sequence 強制設成同一個 regime。

## (B) latent characteristic state 的參數

目前 latent characteristic 參數是 **3 維動態遞迴版本**。

shared 模式下，主要參數為：

- `Omega`：長度 3
- `mu_Z`：長度 3
- `lambda_Z`：長度 3
- `sigma_Z`：長度 3
- `Z0`：長度 3

per-stock 模式下，主要參數為：

- `Omega_i`：shape `(N, 3)`
- `mu_i`：shape `(N, 3)`
- `lambda_i`：shape `(N, 3)`
- `sigma_Z_i`：shape `(N, 3)`
- `Z0_i`：shape `(N, 3)`

目前預設 `mu_i` 由 `low / mid / high` 三種 class center `{-0.5, 0.0, 0.5}` 組成固定 triplet。

## (C) beta 映射參數

目前使用向量線性形式：

$$
\boldsymbol{\beta}_{i,t}=A \mathbf{Z}_{i,t}+b
$$

對應需要設定：

- `exposure_setup["A"]`：`3 × 3` exposure mapping matrix
- `exposure_setup["b"]`：長度 3 的 beta intercept vector

分別對應：

- `beta_mkt`
- `beta_smb`
- `beta_hml`

## (D) FF 三因子向量 AR(1) 的參數

目前使用的主要參數為：

- `X0`
- `Phi`
- `mu_bear`
- `mu_neutral`
- `mu_bull`
- `Sigma_X_bear`
- `Sigma_X_neutral`
- `Sigma_X_bull`

其中：

- `X0` 是長度 3 的初始向量
- `Phi` 是 `3 × 3` matrix
- `mu_bear`, `mu_neutral`, `mu_bull` 是三組 regime-specific 長度 3 mean vector
- 三個 covariance matrix 分別對應 bear / neutral / bull regime

另外程式仍保留舊參數 `Delta` 的相容入口，但已屬 deprecated fallback；目前預設 config 使用的是 `mu_bear / mu_neutral / mu_bull`。

## (E) 個股固定效果與噪音參數

- `alpha_group`
- `epsilon_group`
- `alpha_levels`
- `epsilon_levels`
- `per_stock_alpha_groups`
- `per_stock_epsilon_groups`

## (F) clipping / price / output 參數

- `limit_up`
- `limit_down`
- `shared_init_price`
- `initial_price`
- `per_stock_initial_price`
- `output_dir`

---

# 輸出 / 中間產物

這個專案會產生最終輸出檔，也會在 `run_simulation(...)` 的回傳結果中保留可檢查的中間 DataFrame。

## 回傳的中間產物

`run_simulation(...)` 目前會回傳：

- `dataset_number`
- `run_seed`
- `config`
- `state_sequence`
- `factor_df`
- `latent_state_df`
- `firm_characteristics_df`
- `beta_df`
- `alpha_df`
- `epsilon_df`
- `panel_long_df`
- `output_paths`

其中：

- `dataset_number` 在 single-run 模式下為 `None`
- `run_seed` 是本次實際使用的 random seed

`run_batch_simulations(...)` 目前會回傳一個 list，list 中每個元素至少包含：

- `dataset_number`
- `run_seed`
- `output_paths`
- `status_path`
- `batch_run_id`
- `elapsed_seconds`

注意：

- batch 成功結束後，暫存狀態檔會被清掉，因此回傳結果中的 `status_path` 會被設成 `None`

其中各表欄位為：

- `factor_df`: `[t, state, MKT, SMB, HML]`
- `latent_state_df`: `[stock_id, t, latent_characteristic_1_state, latent_characteristic_2_state, latent_characteristic_3_state]`
- `firm_characteristics_df`: `[stock_id, t, characteristic_1, characteristic_2, characteristic_3]`
- `beta_df`: `[stock_id, t, beta_mkt, beta_smb, beta_hml]`
- `alpha_df`: `[stock_id, alpha]`
- `epsilon_df`: `[stock_id, t, epsilon]`

注意：

- 這些 in-memory DataFrame 裡的 `t` 目前是字串標籤：`t_0, t_1, ..., t_{T-1}`
- 寫出 parquet / csv 前，`t` 會被轉成整數 `0, 1, ..., T-1`

## 最終 `panel_long_df`

最終 `panel_long_df` 目前會整理成：

`[stock_id, t, state, characteristic_1, characteristic_2, characteristic_3, mu, alpha, epsilon_variance, beta_mkt, beta_smb, beta_hml, MKT, SMB, HML, epsilon, raw_return, return, price]`

其中：

- `mu` 是該股票對應的固定三維 `mu` 向量，會以字串形式輸出，例如 `"(0.0,0.5,-0.5)"`
- `epsilon_variance` 欄位名稱沿用現有實作，但內容實際上是 epsilon 的標準差設定值

## 實際寫出的檔案

目前 `save_outputs(...)` 實際會寫出 3 類 artifact：

- `panel_long` parquet
- `market_index_csv`
- `market_index_png`

對應 `output_paths` 目前的 key 為：

- `prices`
- `returns`
- `panel_long`
- `market_index_csv`
- `market_index_png`
- `metadata`
- `excel_workbook`

不會另外寫出：

- wide `return.csv`
- wide `price.csv`
- `metadata.json`
- Excel workbook

---

# 核心白箱數學總結

為了方便快速回顧，核心模型可整理如下。

$$
S_t\in\{-1,0,1\}
$$

$$
\mathbf{X}_t =
\begin{bmatrix}
\mathrm{MKT}_t \\
\mathrm{SMB}_t \\
\mathrm{HML}_t
\end{bmatrix}
$$

$$
\mathbf{X}_t
=
\Phi \mathbf{X}_{t-1}
+
\mu_X(S_t)
+
\mathbf{u}_t,
\qquad
\mathbf{u}_t \sim \mathcal{N}\!\left(\mathbf{0},\,\Sigma_X(S_t)\right)
$$

shared latent-state mode：

$$
\mathbf{Z}_{i,t}
=
\boldsymbol{\mu}
+
\boldsymbol{\Omega}\odot(\mathbf{Z}_{i,t-1}-\boldsymbol{\mu})
+
\boldsymbol{\lambda} S_t
+
\boldsymbol{\xi}_{i,t}
$$

per-stock latent-state mode：

$$
\mathbf{Z}_{i,t}
=
\boldsymbol{\Omega}_i\odot \mathbf{Z}_{i,t-1}
+
\boldsymbol{\mu}_i
+
\boldsymbol{\lambda}_i S_t
+
\boldsymbol{\xi}_{i,t}
$$

$$
\boldsymbol{\xi}_{i,t}
\sim
\mathcal{N}
\left(
\mathbf{0},
\Sigma_{Z,i}
\right)
$$

$$
\Sigma_{Z,i}
=
\operatorname{diag}
\left(
(\sigma_{Z,i}^{(1)})^2,
(\sigma_{Z,i}^{(2)})^2,
(\sigma_{Z,i}^{(3)})^2
\right)
$$

$$
\mathbf{C}_{i,t}^{\mathrm{obs}} = \mathbf{Z}_{i,t}
$$

$$
\boldsymbol{\beta}_{i,t}
=
A\mathbf{Z}_{i,t}+b
$$

$$
\alpha_i = \text{alpha\_level}(\text{alpha\_group}_i)
$$

$$
\varepsilon_{i,t}\sim \mathcal{N}(0,\sigma_{\varepsilon,i}^2)
$$

$$
r_{i,t}
=
\alpha_i
+
\beta_{i,t,1}\mathrm{MKT}_t
+
\beta_{i,t,2}\mathrm{SMB}_t
+
\beta_{i,t,3}\mathrm{HML}_t
+
\varepsilon_{i,t}
$$

$$
r_{i,t}^{\mathrm{obs}}
=
\operatorname{clip}(r_{i,t},\text{limit\_down},\text{limit\_up})
$$

$$
P_{i,t}
=
P_{i,t-1}(1+r_{i,t}^{\mathrm{obs}})
$$

註記：目前 README 不再把 characteristic 解讀為 `size / book-to-price` 的 `exp(Z)` 版本；實作層面它們只是三個可檢查的 characteristic 軸。

## 安裝

在 repo 根目錄執行：

```bash
python -m pip install -r requirements.txt
```

另外，`toy_ff_generator` 子目錄目前已有獨立的 `pyproject.toml`，因此可使用 editable install：

```bash
python -m pip install -e toy_ff_generator
```

若不想安裝，仍可選擇把 `toy_ff_generator/src` 放進 `PYTHONPATH`。

## 執行

目前專案的主要入口在：

```text
toy_ff_generator/src/toy_ff_generator/main.py
```

若未做 editable install，從 repo 根目錄執行 batch 模式可使用：

```bash
PYTHONPATH=toy_ff_generator/src python -m toy_ff_generator.main
```

若已做 editable install，也可直接使用：

```bash
python -m toy_ff_generator.main
```

注意：

- `main()` 目前預設是 `batch=True`
- 因此上面這個指令會呼叫 `run_batch_simulations()`
- 預設 batch 份數來自 `build_default_config()["simulation_setup"]["dataset_count"]`

如果要在 Python 內呼叫單次模擬，若未做 editable install，可使用：

```bash
PYTHONPATH=toy_ff_generator/src python -c "from toy_ff_generator import run_simulation; run_simulation()"
```

若已做 editable install，也可直接使用：

```bash
python -c "from toy_ff_generator import run_simulation; run_simulation()"
```

如果想直接手動調整預設參數，請修改：

```text
toy_ff_generator/src/toy_ff_generator/config.py
```

你可以在 `build_default_config()` 中調整：

- `N`
- `T`
- `random_seed`
- `dataset_count`
- `state_sequence` / `initial_state` / `transition_matrix`
- factor vector AR 參數
- latent characteristic state 參數
- exposure matrix `A` 與 intercept `b`
- alpha / epsilon / clipping / price / output 參數

## 輸出格式

預設輸出目錄為：

```text
toy_ff_generator/outputs/data v3/<state>
```

其中 `<state>` 是 `initial_state` 對應的 `bear` / `neutral` / `bull`。

實際檔名中的 `state_token` 規則如下：

- 若整段 sequence 只有單一 state，使用 `bear` / `neutral` / `bull`
- 若存在多個 state 且是手動 `state_sequence`，使用 `sequence`
- 若存在多個 state 且由 Markov 生成，使用 `markov`

目前 `save_outputs(...)` 會寫出：

1. `{state_token}_{N}_{T}_PL.parquet`
   - single-run 模式
   - 內容為 final `panel_long_df`
   - 寫出前 `t` 會轉成整數欄位

2. `{state_token}_{N}_{T}_PL_{dataset_number}.parquet`
   - batch 模式
   - 每個 dataset 一個 parquet

3. `{state_token}_{N}_{T}_market_index.csv`
   或 `{state_token}_{N}_{T}_market_index_{dataset_number}.csv`
   - Columns: `t, market_index, price_std, price_min, price_max, MKT, SMB, HML`
   - `market_index` 是各股票價格在時間 `t` 的平均值
   - `price_std` 使用 cross-sectional population standard deviation (`ddof=0`)
   - `t` 為整數索引

4. `summary/{state_token}_{N}_{T}_market_index.png`
   或 `summary/{state_token}_{N}_{T}_market_index_{dataset_number}.png`
   - Top panel: `market_index`、`market_index +/- price_std` band、`price_min ~ price_max` band
   - Bottom panel: `MKT / SMB / HML`
   - Title includes state / N / T，batch 模式下也會包含 dataset number

另外：

- batch 執行過程中會暫時建立 `output_dir/_status/toy_ff_generator/`
- 若 batch 正常完成，這個 `_status` 目錄會在結束時清掉

## 測試

在 repo 根目錄執行：

```bash
python -m pytest toy_ff_generator/tests
```