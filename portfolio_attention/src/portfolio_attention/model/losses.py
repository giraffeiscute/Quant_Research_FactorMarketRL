"""Path-based loss functions for portfolio optimization."""

from __future__ import annotations

import warnings
from typing import Literal

import torch


def _coerce_portfolio_returns(portfolio_returns: torch.Tensor) -> torch.Tensor:
    if portfolio_returns.numel() == 0:
        raise ValueError("portfolio_returns must not be empty.")
    if portfolio_returns.ndim == 1:
        return portfolio_returns.unsqueeze(0)
    if portfolio_returns.ndim != 2:
        raise ValueError(
            "portfolio_returns must have shape [num_scenarios_in_batch, time_steps]. "
            f"Received {tuple(portfolio_returns.shape)}."
        )
    return portfolio_returns


def _terminal_return_per_scenario(
    portfolio_returns: torch.Tensor,
    mode: Literal["multiplicative", "additive"] = "multiplicative",
) -> torch.Tensor:
    portfolio_returns = _coerce_portfolio_returns(portfolio_returns)
    if mode == "multiplicative":
        return torch.prod(1 + portfolio_returns, dim=1) - 1
    return torch.sum(portfolio_returns, dim=1)


def _coerce_turnover(turnover: torch.Tensor | None, *, reference: torch.Tensor) -> torch.Tensor | None:
    if turnover is None:
        return None
    if turnover.numel() == 0:
        raise ValueError("turnover must not be empty when provided.")
    if turnover.ndim == 1:
        turnover = turnover.unsqueeze(0)
    if turnover.ndim != 2:
        raise ValueError(
            "turnover must have shape [num_scenarios_in_batch, time_steps]. "
            f"Received {tuple(turnover.shape)}."
        )
    if tuple(turnover.shape) != tuple(reference.shape):
        raise ValueError(
            "turnover must match portfolio_returns shape. "
            f"Received turnover={tuple(turnover.shape)} portfolio_returns={tuple(reference.shape)}."
        )
    return turnover


def _lagged_standardized_positive_cumulative_return_weight(
    portfolio_returns: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    cumulative_returns = torch.cumprod(1 + portfolio_returns, dim=1) - 1
    lagged_cumulative_returns = torch.zeros_like(cumulative_returns)
    if cumulative_returns.shape[1] > 1:
        lagged_cumulative_returns[:, 1:] = cumulative_returns[:, :-1]
    if portfolio_returns.shape[1] < 2:
        return torch.zeros_like(lagged_cumulative_returns)
    return_std = portfolio_returns.std(dim=1, unbiased=True).unsqueeze(1).clamp_min(eps)
    return torch.relu(lagged_cumulative_returns / return_std).detach()


def compute_turnover_penalty(
    turnover: torch.Tensor,
    *,
    norm: str = "l1",
    portfolio_returns: torch.Tensor | None = None,
    allocation_change_l2: torch.Tensor | None = None,
) -> torch.Tensor:
    scored_turnover = _coerce_turnover(turnover, reference=_coerce_portfolio_returns(turnover))
    resolved_norm = str(norm).strip().lower()
    if resolved_norm == "l1":
        regularizer = scored_turnover.abs()
    elif resolved_norm == "l2":
        if allocation_change_l2 is None:
            raise ValueError("allocation_change_l2 is required when turnover_penalty_norm='l2'.")
        regularizer = _coerce_turnover(allocation_change_l2, reference=scored_turnover)
    else:
        raise ValueError(
            "turnover_penalty_norm must be one of {'l1', 'l2'}, "
            f"received {norm!r}."
        )

    if portfolio_returns is None:
        return regularizer.mean()

    scored_returns = _coerce_portfolio_returns(portfolio_returns)
    if tuple(scored_returns.shape) != tuple(scored_turnover.shape):
        raise ValueError(
            "portfolio_returns must match turnover shape when computing weighted turnover penalty. "
            f"Received portfolio_returns={tuple(scored_returns.shape)} turnover={tuple(scored_turnover.shape)}."
        )
    weights = _lagged_standardized_positive_cumulative_return_weight(scored_returns)
    return (weights * regularizer).mean()


# 計算投資組合整段路徑的最終報酬損失，回傳負值供最小化。
def return_loss(
    portfolio_returns: torch.Tensor,
    mode: Literal["multiplicative", "additive"] = "multiplicative",
) -> torch.Tensor:
    """計算整段投資組合報酬的負終值損失。"""
    terminal_returns = _terminal_return_per_scenario(portfolio_returns, mode=mode)
    return -terminal_returns.mean()  # 取負號，讓最佳化器用最小化方式最大化報酬


# 計算 Sharpe Ratio 損失，資料不足或波動太小時退回報酬損失。
def sharpe_loss(
    portfolio_returns: torch.Tensor,
    eps: float = 1e-6,
    min_time_steps: int = 2,
    risk_free_rate: float = 0.0,
    fallback_mode: Literal["multiplicative", "additive"] = "multiplicative",
) -> torch.Tensor:
    """計算整段投資組合報酬的負 Sharpe Ratio 損失。"""
    portfolio_returns = _coerce_portfolio_returns(portfolio_returns)

    _, time_steps = portfolio_returns.shape  # 取得 batch 與時間長度

    if time_steps < min_time_steps:
        warnings.warn(
            f"Sharpe loss received too few time steps ({time_steps}); falling back to return loss.",
            RuntimeWarning,
            stacklevel=2,
        )
        return return_loss(portfolio_returns, mode=fallback_mode)  # 時間步太少時退回終值報酬

    excess_returns = portfolio_returns - risk_free_rate  # 扣掉無風險利率得到超額報酬
    mean_ret = excess_returns.mean(dim=1)  # 每條路徑的平均超額報酬
    std_ret = excess_returns.std(dim=1, unbiased=True)  # 每條路徑的樣本標準差

    sharpe = mean_ret / (std_ret + eps)  # Sharpe = 平均報酬 / 波動
    fallback_losses = -_terminal_return_per_scenario(portfolio_returns, mode=fallback_mode)
    scenario_losses = torch.where(std_ret > eps, -sharpe, fallback_losses)
    return scenario_losses.mean()


# 依序更新 A/B 統計量來計算 Differential Sharpe Ratio 損失。
def differential_sharpe_loss(
    portfolio_returns: torch.Tensor,
    eta: float = 0.2,
    A0: float = 0.0,
    B0: float = 1e-4,
    eps: float = 1e-8,
    reduction: Literal["mean", "sum", "last"] = "mean",
) -> torch.Tensor:
    """計算整段投資組合報酬的負 Differential Sharpe Ratio 損失。"""
    portfolio_returns = _coerce_portfolio_returns(portfolio_returns)

    batch_size, T = portfolio_returns.shape
    device = portfolio_returns.device  # 保持中間張量與輸入在同一裝置

    A = torch.full((batch_size,), A0, device=device)  # 一階動差的初始估計
    B = torch.full((batch_size,), B0, device=device)  # 二階動差的初始估計
    scores = []

    for t in range(T):
        Rt = portfolio_returns[:, t]  # 取出第 t 個時間步的報酬
        delta_A = Rt - A  # 當前報酬對均值估計的偏差
        delta_B = Rt**2 - B  # 當前平方報酬對二階動差估計的偏差

        numerator = B * delta_A - 0.5 * A * delta_B  # DSR 分子
        denominator = (B - A**2 + eps) ** 1.5  # DSR 分母
        Dt = numerator / (denominator + eps)  # 單步 Differential Sharpe 分數
        scores.append(Dt)

        A = A + eta * delta_A  # 更新一階動差估計
        B = B + eta * delta_B  # 更新二階動差估計

    all_scores = torch.stack(scores, dim=1)  # [B, T]，收集所有時間步分數

    if reduction == "last":
        score = all_scores[:, -1]  # 只取最後一步
    elif reduction == "sum":
        score = all_scores.sum(dim=1)  # 對時間維度加總
    else:
        score = all_scores.mean(dim=1)  # 對時間維度取平均

    return -score.mean()  # 取負號轉成可最小化的損失


# 計算 Sortino Ratio 損失，只懲罰低於目標報酬的下行波動。
def sortino_loss(
    portfolio_returns: torch.Tensor,
    target_return: float = 0.0,
    eps: float = 1e-6,
    min_time_steps: int = 2,
    fallback_mode: Literal["multiplicative", "additive"] = "multiplicative",
) -> torch.Tensor:
    """計算整段投資組合報酬的負 Sortino Ratio 損失。"""
    portfolio_returns = _coerce_portfolio_returns(portfolio_returns)

    _, time_steps = portfolio_returns.shape
    if time_steps < min_time_steps:
        return return_loss(portfolio_returns, mode=fallback_mode)  # 時間步不足時退回報酬損失

    excess = portfolio_returns - target_return  # 相對目標報酬的超額報酬
    mean_ret = excess.mean(dim=1)  # 平均超額報酬

    downside = torch.min(excess, torch.zeros_like(excess))  # 只保留負向偏離
    downside_deviation = torch.sqrt((downside**2).mean(dim=1) + eps)  # 下行標準差

    sortino = mean_ret / (downside_deviation + eps)  # Sortino = 平均報酬 / 下行波動
    return -sortino.mean()  # 取負號轉成可最小化的損失


# 計算最大回撤風險，回傳正值表示風險大小而不是 reward 型損失。
def max_drawdown_loss(
    portfolio_returns: torch.Tensor,
    mode: Literal["multiplicative", "additive"] = "multiplicative",
    eps: float = 1e-8,
) -> torch.Tensor:
    """計算投資組合路徑的平均最大回撤風險。"""
    portfolio_returns = _coerce_portfolio_returns(portfolio_returns)

    if mode == "multiplicative":
        equity = torch.cumprod(1 + portfolio_returns, dim=1)  # 用複利方式累積資產曲線
    else:
        equity = 1 + torch.cumsum(portfolio_returns, dim=1)  # 用加法方式累積資產曲線

    running_peak = torch.cummax(equity, dim=1)[0]  # 每個時間點之前的最高淨值
    drawdown = (running_peak - equity) / (running_peak + eps)  # 當前回撤比例
    max_dd = drawdown.max(dim=1)[0]  # 每條路徑的最大回撤

    return max_dd.mean()  # 回傳 batch 平均最大回撤


# 計算 CVaR（Expected Shortfall）風險，聚焦最差尾部損失區間。
def cvar_loss(
    portfolio_returns: torch.Tensor,
    alpha: float = 0.05,
) -> torch.Tensor:
    """計算投資組合報酬分布的平均 CVaR 風險。"""
    portfolio_returns = _coerce_portfolio_returns(portfolio_returns)

    losses = -portfolio_returns  # 報酬轉損失，方便計算尾部風險
    batch_size, T = losses.shape

    cvar_list = []
    for i in range(batch_size):
        path_losses = losses[i]  # 單一路徑的全部損失
        var = torch.quantile(path_losses, 1 - alpha)  # 先求 VaR 門檻
        tail_losses = path_losses[path_losses >= var]  # 取超過 VaR 的尾部損失
        if tail_losses.numel() == 0:
            cvar_list.append(path_losses.max())  # 若尾部為空，退回最大損失
        else:
            cvar_list.append(tail_losses.mean())  # CVaR = 尾部損失平均

    return torch.stack(cvar_list).mean()  # 回傳 batch 平均 CVaR


# 依名稱分派對應的 loss function，方便外部統一呼叫。
def build_loss(
    name: str,
    portfolio_returns: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """依據名稱建立並呼叫對應的路徑損失函式。"""
    normalized = name.lower().replace("_", "")  # 統一名稱格式，方便比對別名

    if normalized in {"return", "totalreturn", "terminalreturn"}:
        return return_loss(portfolio_returns, **kwargs)  # 終值報酬類別名

    if normalized in {"sharpe", "sr"}:
        return sharpe_loss(portfolio_returns, **kwargs)  # Sharpe 類別名

    if normalized in {"dsr", "differentialsharpe"}:
        return differential_sharpe_loss(portfolio_returns, **kwargs)  # DSR 類別名

    if normalized == "sortino":
        return sortino_loss(portfolio_returns, **kwargs)  # Sortino

    if normalized in {"mdd", "maxdrawdown"}:
        return max_drawdown_loss(portfolio_returns, **kwargs)  # 最大回撤類別名

    if normalized == "cvar":
        return cvar_loss(portfolio_returns, **kwargs)  # CVaR

    raise ValueError(f"Unsupported loss: {name}")  # 不支援的 loss 名稱直接報錯


def build_portfolio_objective_loss(
    name: str,
    portfolio_returns: torch.Tensor,
    turnover: torch.Tensor | None = None,
    allocation_change_l2: torch.Tensor | None = None,
    turnover_penalty: float = 0.0,
    transaction_cost_rate: float = 0.0,
    turnover_penalty_norm: str = "l1",
    **kwargs,
) -> torch.Tensor:
    scored_returns = _coerce_portfolio_returns(portfolio_returns)
    scored_turnover = _coerce_turnover(turnover, reference=scored_returns)
    resolved_turnover_penalty = float(turnover_penalty)
    resolved_transaction_cost_rate = float(transaction_cost_rate)

    objective_returns = scored_returns
    if resolved_transaction_cost_rate > 0.0:
        if scored_turnover is None:
            raise ValueError("turnover is required when transaction_cost_rate > 0.")
        objective_returns = scored_returns - resolved_transaction_cost_rate * scored_turnover

    loss = build_loss(name, objective_returns, **kwargs)
    if resolved_turnover_penalty > 0.0:
        if scored_turnover is None:
            raise ValueError("turnover is required when turnover_penalty > 0.")
        turnover_regularizer = compute_turnover_penalty(
            scored_turnover,
            norm=turnover_penalty_norm,
            portfolio_returns=objective_returns,
            allocation_change_l2=allocation_change_l2,
        )
        loss = loss + resolved_turnover_penalty * turnover_regularizer
    return loss
