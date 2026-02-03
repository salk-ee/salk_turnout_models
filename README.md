# salk_turnout_models

Python package for turnout modeling with MRP, ecological inference, and selection-correction variants.

This is the implementation of "A Unified Bayesian Model for Voter Turnout Estimation: Combining Surveys, Aggregate Data, and  Selection Correction" by M.Niitsoo, R. Rebane, T. Jüristo and also contains the data and code for the simulations provided therein. 

## Usage

### Example

```python
import salk_turnout_models as tm

config = {
    'model_type': 'FS',
    'outcome_col': 'voting_intent',
    'input_cols': ['age_group', 'gender', 'education', 'unit', 'nationality'],
    'interactions': True, # or [['age_group','gender'],['education','nationality']]
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
- `interactions`: optional; `True` for all 2-way interactions, or explicit list of tuples.

#### Advanced configuration options
- `multilevel`: Should multilevel priors be used; defaults to `True`.
- `centered`: Should we use centered parametrization; defaults to `False` for `BP`, `GG` and `True` for `EI`, `PM`, `FS`.
- `priors`: optional; dict of prior scales and constants.
- `margin_dist`: optional; one of `binomial` (default), `poisson`, `betabinomial`, `potential`.
- `imr`: optional; `FS` only, use inverse Mills ratio approximation.
- `survey_outcome_col`: optional; outcome column in survey data (defaults to `outcome_col`).
- `selection_survey`: optional; DataFrame, file path, or list of either for `FS` selection process.
  Defaults to `survey` when omitted.

`sample_kwargs` can be used to pass arguments to `pm.sample`

Return value is a dict with:

- `model`: fitted PyMC model object.
- `post_model`: poststratification model.
- `idata`: ArviZ InferenceData from sampling.
- `post_idata`: ArviZ InferenceData from posterior predictive poststrat.
- `draws`: poststratified draws as a DataFrame.

## Simulation study

The `simulation_study/` directory contains three notebooks:

- `generate-simulation-datasets.ipynb`: creates synthetic datasets and margins used in the study.
- `run-simulations.ipynb`: fits the model grid over synthetic datasets and records outputs.
- `simulation-figures.ipynb`: aggregates results and produces figures/tables for analysis.

## Estonian turnout

The `estonia_turnout/estonia_turnout.ipynb` notebook is the analysis of 2023 Parliamentary election turnout.
