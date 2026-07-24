# salk_turnout_models

Python package for turnout modeling with MRP, ecological inference, and selection-correction variants.

This is the implementation of "A Unified Bayesian Model for Voter Turnout Estimation: Combining Surveys, Aggregate Data, and  Selection Correction" by M.Niitsoo, R. Rebane, T. Jüristo and also contains the data and code for the simulations provided therein.

> **Note — this is the `paper-replication` branch.** It preserves the original FS
> overreporting placement described in the paper's erratum (the overreporting
> shift is applied to the *conditional* outcome probability, after the Heckman
> scaling), i.e. the exact code used to produce the paper's simulation and
> empirical results. The `main` branch contains the corrected placement. As the
> erratum explains, the two are equivalent up to the reparameterization
> `beta_or -> beta_or * sqrt(1 - rho^2)`, so the reported numerical results are
> unaffected; this branch exists only for exact replication.

## Usage

### Basic example

```python
import salk_turnout_models as tm

config = {
    'name': 'estonia_pm',
    'model_type': 'PM',
    'outcome_col': 'voting_intent',
    'input_cols': ['age_group', 'gender', 'education', 'municipality', 'nationality'],
    'interactions': True,
    'population': './data/census.csv',
    'margin': './data/estonia_turnout/rk2023_unit_margins.csv',
    'survey': './data/estonia_turnout/survey.csv',
}
result = tm.run_model(config, sample_kwargs={'chains': 2, 'tune': 500, 'draws': 500})
draws = result['draws']
```

### Inputs and outputs

`run_model(config, sample_kwargs=None, save_path=None)` expects a config dict with:

- `name`: run name used for logging/output paths.
- `model_type`: one of `BP`, `EI`, `GG`, `PM`, `FS`.
- `input_cols`: list of categorical covariates used in the model.
- `outcome_col`: column name for the binary/ordinal outcome.
- `population`: DataFrame or file path for census/poststrat data.
- `survey`: DataFrame, file path, or list of either (required for `BP`, `GG`, `PM`, `FS`).
- `margin`: DataFrame, file path, or list of either (required for `EI`, `GG`, `PM`, `FS`).
- `interactions`: optional; `True` for all 2-way interactions, or explicit list of tuples/lists, e.g. `[('age_group', 'gender'), ('education', 'nationality')]`.

#### Advanced configuration options

- `centered`: Should we use centered parametrization; defaults to `False` for `BP`, `GG` and `True` for `EI`, `PM`, `FS`.
- `priors`: optional; dict of prior functions overriding defaults (see full list below).
- `margin_dist`: optional; one of `binomial`, `poisson`, `betabinomial`, `potential`. Defaults to `binomial` for `EI`/`PM`/`FS` and `potential` for `GG`.
- `robust_selection`: optional; `FS` only. If `True`, selection is modeled as Student-t with `selection_nu` prior.
- `survey_outcome_col`: optional; outcome column in survey data (defaults to `outcome_col`).
- `selection_survey`: optional; DataFrame, file path, or list of either for `FS` selection process.
  Defaults to `survey` when omitted.
- `selection_cols`: optional; `FS` only. Subset of `input_cols` used in the selection model. Defaults to `input_cols`.

For `FS`, `interactions` apply to both predictors; `selection_cols` controls which variables are available to the selection predictor.

#### Priors (all keys in `DEFAULT_PRIORS`)

Each prior is a callable: `lambda name, **kwargs: <PyMC RV or constant>`.
You can override any subset via `config['priors']`.

- `outcome_intercept`: default `Normal(mu=0, sigma=1)`; \(\beta_0^o\) in the paper.
- `outcome_scale`: default `HalfNormal(sigma=1.0)`; scale for outcome main effects \(\beta^o_{d,i}\).
- `outcome_interaction_scale`: default `HalfNormal(sigma=1.0)`; scale for outcome interaction effects (same \(\sigma^o\) convention).
- `outcome_cutpoints`: default `Normal(mu=0, sigma=2)`; ordered/zero-sum cutpoints for ordinal survey outcomes.
- `overreporting`: default `AsymmetricLaplace(mu=0, kappa=1/3, b=15)`; \(\beta^o_{or}\), additive survey overreporting bias (`PM`, `FS`).
- `selection_intercept`: default `StudentT(nu=3, mu=-1.5, sigma=0.75)`; \(\beta_0^s\), selection LP intercept (`FS`).
- `selection_scale`: default `HalfNormal(sigma=1.0)`; scale for selection main effects \(\beta^s_{d,i}\) (`FS`).
- `selection_interaction_scale`: default `HalfNormal(sigma=1.0)`; scale for selection interaction effects (`FS`, same \(\sigma^s\) convention).
- `rho`: default transformed `2 * Beta(6, 3) - 1`; \(\rho\), outcome/selection latent correlation (`FS`).
- `selection_nu`: default `Gamma(alpha=1.0, beta=0.1)`; degrees-of-freedom-like robustness parameter (`FS` when `robust_selection=True`).
- `gg_delta`: default `Normal(sigma=10)`; near-flat correction terms in `GG`.

Notes:

- To disable multilevel shrinkage, set scale priors to constants, e.g. `outcome_scale`, `selection_scale`, `outcome_interaction_scale`, `selection_interaction_scale`.
- To disable overreporting correction, set `overreporting` prior to constant zero.

Complex `FS` example:

```python
import pymc as pm
import salk_turnout_models as tm

result = tm.run_model({
      'name': 'estonia_fs_complex',
      'model_type': 'FS',
      'outcome_col': 'voting_intent',
      'input_cols': ['age_group', 'gender', 'education', 'municipality', 'nationality'],
      'selection_cols': ['age_group', 'gender', 'education', 'nationality'], # Drop municipality level
      'interactions': [ # Limit interactions
          ('age_group', 'gender'),
          ('education', 'nationality'),
          ('municipality', 'nationality'),
      ],
      'population': './data/census.csv',
      'survey': './data/estonia_turnout/survey.csv',
      'selection_survey': './data/estonia_turnout/survey_qfree.csv', # Separate survey for selection
      'margin': [ # Multiple margins
          './data/estonia_turnout/rk2023_unit_margins.csv',
          './data/estonia_turnout/rk2023_age_group_margins.csv',
          './data/estonia_turnout/rk2023_gender_margins.csv',
      ],
      'margin_dist': 'poisson', # Different margin model
      'robust_selection': True, # Robust selection model
      'priors': {
          # Lower scales -> stronger regularization on selection
          'selection_scale': lambda name, **kwargs: 0.5,
          'selection_interaction_scale': lambda name, **kwargs: 0.25,
          # Disable overreporting correction
          'overreporting': lambda name, **kwargs: 0.0,
          # Optional alternative rho prior
          'rho': lambda name, **kwargs: pm.Uniform(name, lower=-1, upper=1, **kwargs),
      },
    }, sample_kwargs={'chains': 2, 'tune': 800, 'draws': 500, 'target_accept': 0.95},
    save_path='./output/estonia_fs_complex',
)
```

`sample_kwargs` can be used to pass arguments to `pm.sample`

Return value is a dict with:

- `model`: fitted PyMC model object.
- `post_model`: poststratification model.
- `idata`: ArviZ InferenceData from sampling.
- `post_idata`: ArviZ InferenceData from posterior predictive poststrat.
- `draws`: poststratified draws as a DataFrame.

## Simulation study

The `simulation_study/` directory contains the notebooks that reproduce the paper's
simulation study:

- `generate-simulation-datasets.ipynb`: creates the synthetic datasets and margins used in the study.
- `run-simulations.ipynb`: fits the model grid over the synthetic datasets and records outputs (`run-simulations.py` is its script export).
- `simulation-figures.ipynb`: aggregates results into the paper's figures.
- `simulation-numbers.ipynb`: recomputes the quantitative claims (tables and in-text numbers).
- `coverage-rerun.ipynb`: refits the baseline scenario retaining full posteriors, for the interval-calibration and convergence diagnostics.

The full model grid takes roughly 320 hours of MCMC. To avoid re-running it, the
aggregated results are committed at `tmp/models/model_results.csv`, so
`simulation-figures.ipynb` and `simulation-numbers.ipynb` regenerate every figure
and number directly from that file.

Extra dependencies required for these notebooks can be installed via

```bash
pip install -e ".[dev]"
```

## Estonian turnout

The `estonia_turnout/` directory contains the empirical application to the 2023 Estonian
parliamentary election:

- `estonia_turnout.ipynb`: the main analysis (data preparation, model fits, and figures).
- `betabinomial-margins.ipynb`: the beta-binomial margin-likelihood robustness check.
