============================================================
Synthetic Market Generator README 
============================================================

Files
------------------------------------------------------------
model.py
    Core market generator: Factor / Exposure / Observable / Market
para.py
    Recommended parameter ranges and example observable settings
main.py
    Example script for generating and saving datasets

============================================================
1. Basic Workflow
============================================================

Step 1:
    Create a Market object

    market=Market(num_stock,time_scale,
                  is_exposure_dyna=False,
                  master_seed=42)

Step 2:
    Add factors using factor_design()

Step 3:
    Build latent space

    market.LatentBuild()

Step 4:
    Add observables using obs_build()

Step 5:
    Export data using parquet / npy


============================================================
2. Factor Design
============================================================

Example:
------------------------------------------------------------
market.factor_design(
    'Style',
    0.90,
    0.00,
    0.10,
    0.05,
    0.95
)
------------------------------------------------------------

Arguments:
------------------------------------------------------------
fac_type
    Factor type:
        'Global'
        'Style'
        'Sector'

fac_time_corr
    Temporal AR(1) correlation

fac_trend
    Mean trend

fac_noisy
    Noise amplitude

beta_avg
    Mean exposure

beta_fluct
    Exposure fluctuation


============================================================
3. Optional Factor Switches
============================================================

Jump process:
------------------------------------------------------------
fac_is_jump=True
fac_jump_prob=0.1
fac_jump_scale=0.5

Macro trend:
------------------------------------------------------------
fac_is_macro_trend=True
fac_macro_trend=0.3

Stochastic volatility:
------------------------------------------------------------
fac_is_stochastic_noisy=True
fac_noisy_time_corr=0.95
fac_noisy_noisy=0.2


============================================================
4. Exposure Dynamics
============================================================

Static exposure:
------------------------------------------------------------
is_exposure_dyna=False

Dynamic exposure:
------------------------------------------------------------
is_exposure_dyna=True

Additional parameters:
------------------------------------------------------------
beta_time_corr
beta_noisy

Optional low-rank motion:
------------------------------------------------------------
is_LowRank=True
LowRank_dim=3
LowRank_fluc=0.1


============================================================
5. Observable Types
============================================================

Linea:
------------------------------------------------------------
beta(t) * F(t)

Convo:
------------------------------------------------------------
Exponential convolution in time

Satur:
------------------------------------------------------------
tanh(beta(t) * F(t))

Squar:
------------------------------------------------------------
(beta(t) * F(t))^2

Noisy:
------------------------------------------------------------
Pure random noise


============================================================
6. Build Observables
============================================================

Example:
------------------------------------------------------------
market.obs_build('Linea',u=u_return)

market.obs_build('Convo',
                 u=u_moment,
                 tau_length=50,
                 tau_bench=10)

market.obs_build('Squar'）
------------------------------------------------------------

u:
------------------------------------------------------------
Projection direction in latent space

Different u vectors generate different observables.


============================================================
7. Latent Space
============================================================

LatentBuild() creates hidden factor directions.

Example:
------------------------------------------------------------
market.LatentBuild(
    dim_latent=3,
    noisy=0.1
)

Purpose:
------------------------------------------------------------
Different factors mix into observables through latent vectors.


============================================================
8. Output Data
============================================================

data_obs
------------------------------------------------------------
Shape:
    (num_obs,time_scale,num_stock)

Observable features

data_factor
------------------------------------------------------------
Shape:
    (num_factor,time_scale)

True hidden factors

data_beta
------------------------------------------------------------
Shape:
    (num_factor,num_stock)

Factor exposures

============================================================
9. Notes
============================================================

1. All randomness is controlled by master_seed.
2. Factor and exposure generators support independent RNGs.
3. Convo observables introduce temporal memory.
4. Dynamic exposure significantly increases difficulty.
5. Large fac_time_corr means stronger temporal persistence.
6. Large beta_fluct means stronger cross-sectional structure.

============================================================
FILE ENDED