============================================================
Portfolio Weight Evaluation README
============================================================

This document explains the statistical quantities and figures used in DataEval for analyzing neural-network-generated portfolio weights on synthetic factor markets.

------------------------------------------------------------
I. INPUT DATA FORMAT
------------------------------------------------------------
data_weights:
    numpy array with shape
        (num_time,num_stock+1)
    The last column [-1] corresponds to cash allocation.
        weights[:,:-1] -> stock weights
        weights[:,-1]  -> cash weights
data_beta:
    numpy array with shape
        (num_stock,num_factor)
    beta[s,k] is the exposure of stock-s to factor-k.
------------------------------------------------------------
II. WRITE OUTPUT STATISTICS
------------------------------------------------------------
============================================================
Factor-k Weight Corr
============================================================

Definition:
    Corr(weight_avg,beta_k)
where:
    weight_avg[s] = time-averaged portfolio weight of stock-s
    beta_k[s] = exposure of stock-s to factor-k

Statistical meaning:
    Measures whether the neural network portfolio is aligned with factor-k.

Interpretation:
    Corr ~ +1:        Network strongly longs positive-beta stocks.
    Corr ~ - 1:        Network strongly shorts positive-beta stocks.
    Corr ~   0:        Portfolio is approximately factor-neutral.

============================================================
Factor-k Turnover Corr
============================================================

Definition:
    Corr(turnover_avg,beta_k)
where:
    turnover_avg[s]
        = time-averaged absolute weight change of stock-s

Statistical meaning：
    Measures whether portfolio trading activity depends on factor exposure.

Interpretation:
    Large magnitude: Certain factor sectors are traded more aggressively.
    Near zero: Trading activity approximately independent of factor.

============================================================
Factor Regression R^2
============================================================

Definition:
    weight_avg ≈ Σ_k a_k beta_k

R^2 measures how much of the cross-sectional portfolio
structure can be explained by linear factor exposures.

Statistical meaning：
    Measures whether the portfolio is essentially a linear factor portfolio.

Interpretation:
    R^2 ~ 1:        Portfolio almost fully explained by factor model.
    Small R^2:    Network learns nonlinear or idiosyncratic structures.

============================================================
Market Bias
============================================================

Definition:
    mean(weight_avg)

Statistical meaning：
    Measures global long-short imbalance.

Interpretation:
    Positive:		Net long bias.
    Negative:		Net short bias.
    Near zero:		Approximately market-neutral.

============================================================
Cash Average
============================================================

Definition:
    mean(cash_weight)

Statistical meaning：
    Average cash allocation level.

Interpretation:
    Large:		Conservative / low-risk allocation.
    Small:		 Aggressive full-investment behavior.

============================================================
Concentration HHI (Herfindahl-Hirschman Index)
============================================================

Definition:
    HHI = Σ_s weight_avg[s]^2

Statistical meaning：
    Measures concentration of portfolio holdings.

Interpretation:
    Large HHI:		Portfolio concentrated in a few stocks.
    Small HHI:		Diversified allocation.

============================================================
Dominant Mode Strength
============================================================

Definition:
    largest_eigenvalue / trace(covariance)

where covariance is the stock-stock covariance matrix of portfolio weights.

Statistical meaning：
    Measures whether portfolio structure is dominated by a single collective mode.

Interpretation:
    Near 1:	Portfolio essentially rank-1.
    Smaller:	Multiple independent structures exist.

============================================================
Effective Rank
============================================================

Definition:
    p_i = λ_i / Σ_i λ_i
    S = -Σ_i p_i log(p_i)
    effective_rank = exp(S)

Statistical meaning：
    Measures effective dimensionality of the portfolio covariance structure.

Interpretation:
    Small effective rank:		Portfolio dominated by few modes.
    Large effective rank:		Rich multi-factor structure.

------------------------------------------------------------
III. FIGURE OUTPUTS
------------------------------------------------------------

============================================================
Figure 1:
Exposure vs Weights
============================================================

Scatter plot:
    x-axis:        factor exposure beta
    y-axis:        time-averaged portfolio weight

Purpose:
    Visualize whether portfolio weights align with factor structure.

Typical behaviors:
1. Linear relation:		Pure factor portfolio.
2. Nonlinear relation:	Saturation / threshold effects.
3. No structure:		Factor ignored by network.

============================================================
Figure 2:
Exposure vs Turnover
============================================================

Scatter plot:
    x-axis:        factor exposure beta
    y-axis:        average turnover

Purpose:
    Analyze whether trading activity depends on factor exposure.

Possible interpretations:

1. Large-beta stocks trade more:	Factor timing behavior.
2. Uniform turnover:				Exposure-independent trading.

============================================================
Figure 3:
Eigen Spectrum
============================================================

Plot:
    eigenvalues of stock covariance matrix

Purpose:
    Analyze collective portfolio modes.

Interpretation:

1. One dominant eigenvalue:	Portfolio nearly rank-1.
2. Broad spectrum:			Multi-factor structure.
Effective rank shown in title summarizes dimensionality.


============================================================
Figure 4:
Portfolio Turnover Distribution
============================================================

Histogram:
    total portfolio turnover per time step

Purpose:
    Analyze temporal stability of portfolio trading.

Interpretation:

1. Narrow distribution:		Stable trading behavior.
2. Fat tail:				Occasional violent rebalancing.
3. Large variance:			Unstable dynamic strategy.


------------------------------------------------------------
IV. PRACTICAL INTERPRETATION 
------------------------------------------------------------

(These interpretations are generated by ChatGPT)
For synthetic factor markets:

A good factor-learning portfolio network might generally show:

1. Large factor-weight correlation
2. High factor regression R^2
3. Moderate effective rank
4. Controlled concentration
5. Stable turnover distribution
6. Reasonable market neutrality

Abnormal situations:

1. Very high dominant mode strength:
        Network collapsed to one market mode.

2. Very large HHI:
        Portfolio excessively concentrated.

3. Large turnover variance:
        Unstable trading dynamics.

4. High factor corr but low R^2:
        Network only partially learns factor structure.

5. Very low effective rank:
        Portfolio lacks structural diversity.


------------------------------------------------------------
END OF FILE
------------------------------------------------------------