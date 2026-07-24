import itertools as it
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import pymc as pm
import pathlib
import pytensor.tensor as pt
import re
import xarray as xr
from collections.abc import Sequence

from pymc.model.fgraph import clone_model

DEFAULT_PRIORS = {

    # Outcome model
    # beta_0^o ~ Normal(0, 1)  (Implementation section)
    'outcome_intercept': lambda name, **kwargs: pm.Normal(name, mu=0, sigma=1, **kwargs),
    # tau_d^o ~ Half-Normal(0, sigma^o), with sigma^o=1 by default (Eq. 76-77, Implementation)
    'outcome_scale': lambda name, **kwargs: pm.HalfNormal(name, sigma=1.0, **kwargs),
    # Same scale prior for interaction terms under the same sigma^o convention
    'outcome_interaction_scale': lambda name, **kwargs: pm.HalfNormal(name, sigma=1.0, **kwargs),

    # PM/FS
    # beta_or^o ~ AsymmetricLaplace(0, 1/3, 15) (Implementation section)
    'overreporting': lambda name, **kwargs: pm.AsymmetricLaplace(name, mu=0, kappa=1/3, b=5*3, **kwargs), # Informative

    # FS model
    # beta_0^s ~ StudentT(3, -1.5, 0.75) (Implementation section)
    'selection_intercept': lambda name, **kwargs: pm.StudentT(name, nu=3, mu=-1.5, sigma=0.75, **kwargs), # Informative
    # tau_d^s ~ Half-Normal(0, sigma^s), with sigma^s=1 by default
    'selection_scale': lambda name, **kwargs: pm.HalfNormal(name, sigma=1.0, **kwargs),
    # Same scale prior for interaction terms under the same sigma^s convention
    'selection_interaction_scale': lambda name, **kwargs: pm.HalfNormal(name, sigma=1.0, **kwargs),
    # rho ~ 2*Beta(6,3)-1 (Implementation section)
    'rho': lambda name, **kwargs: pm.Deterministic(name, 2*pm.Beta(f"{name}_v", alpha=6, beta=3, **kwargs) - 1), # Informative
    'selection_nu': lambda name, **kwargs: pm.Gamma(name, alpha=1.0, beta=0.1, **kwargs), # nu for robust selection (if used)
    
    # Ordered Probit cutpoints
    # kappa_k ~ Normal(0, sigma_c) with ordering (Extensions: Ordinal Outcomes)
    'outcome_cutpoints': lambda name, **kwargs: pm.Normal(name, mu=0.0, sigma=2, **kwargs),

    # GG
    'gg_delta': lambda name, **kwargs: pm.Normal(name, sigma=10.0, **kwargs), # This is meant to be flat, but pure pm.Flat causes issues
}

# Standard normal distribution for stable probit log-likelihoods
STD_NORMAL = pm.Normal.dist(mu=0, sigma=1.0)
def logPhi(x):
    # log Phi(x), where Phi(x) is the standard normal CDF used in probit links.
    return pm.logcdf(STD_NORMAL, x)


def encode_ordered_outcome(series: pd.Series) -> tuple[np.ndarray, int]:
    """
    Encode an ordered outcome for use with Ordered Probit models.

    This helper function handles:
    1. Binary "No"/"Yes" outcomes (mapped to 0/1).
    2. Explicitly ordered pandas Categoricals (mapped to integer codes 0..K-1).
    3. Numeric outcomes (shifted to start at 0).

    Returns:
        tuple[np.ndarray, int]: A tuple containing:
            - Integer-coded outcome array (values in 0..K-1).
            - The number of categories K.

    This prepares the data for the ordinal probit extension described in Section 7
    ("Limits and extensions"), specifically the "Ordinal Outcomes" subsection.
    """
    # Special-case common binary "No/Yes" outcomes (even if not an ordered categorical)
    # so OrderedProbit can be used with a stable, explicit ordering.
    uniq = pd.unique(series.dropna())
    if set(uniq) == {'No', 'Yes'}:
        codes = series.map({'No': 0, 'Yes': 1}).to_numpy()
        return codes.astype(int), 2

    # Ordered categorical -> integer codes in category order
    if isinstance(series.dtype, pd.CategoricalDtype) and series.cat.ordered:
        print('Ordered categorical outcome:', list(series.cat.categories))
        codes = series.cat.codes.to_numpy()
        if (codes < 0).any():
            raise ValueError(
                'Ordered categorical outcome contains missing values (code -1).'
            )
        K = int(len(series.cat.categories))
        if K < 2:
            raise ValueError('OrderedProbit requires at least 2 outcome categories.')
        return codes.astype(int), K

    # Fallback: numeric-coded categories (e.g. 1..K or 0..K-1)
    vals = pd.to_numeric(series, errors='raise')
    if vals.isna().any():
        raise ValueError('Outcome contains missing values.')
    vals = vals.astype(int).to_numpy()
    shifted = vals - int(vals.min())
    K = int(shifted.max()) + 1
    if K < 2:
        raise ValueError('OrderedProbit requires at least 2 outcome categories.')
    return shifted, K


def validate_margin_data(
    margin_df: pd.DataFrame, margin_vars: list[str], coords: dict
) -> None:
    """
    Validate that margin data contains no duplicates and all categories match coords.
    """
    if np.any(margin_df.duplicated(subset=margin_vars)):
        raise ValueError('Margin data has duplicates')

    for var in margin_vars:
        for cat in coords[var]:
            if cat not in margin_df[var].unique():
                raise ValueError(
                    f'Category {cat} for variable {var} not in margin data'
                )


def get_coords(df: pd.DataFrame, input_vars: list[str]) -> dict:
    """
    Extract sorted unique values for each input variable to define coordinates.
    Respects categorical order if the column has a CategoricalDtype.
    """
    coords = {}
    for var in input_vars:
        if isinstance(df[var].dtype, pd.CategoricalDtype):
            coords[var] = list(df[var].cat.categories)
        else:
            coords[var] = sorted(list(df[var].unique()))
    return coords


def category_indices(
    df: pd.DataFrame, vars: list[str], coords: dict
) -> dict[str, np.ndarray]:
    """
    Convert dataframe columns to integer codes based on provided coordinates.
    """
    return {var: pd.Categorical(df[var], categories=coords[var]).codes for var in vars}


def apply_categories(
    df: pd.DataFrame, input_vars: list[str], coords: dict
) -> pd.DataFrame:
    """
    Enforce categorical dtypes on dataframe columns using provided coordinates.
    """
    df_copy = df.copy()

    for var in input_vars:
        cats = coords[var]
        for cat in df_copy[var].unique():
            if cat not in cats:
                raise ValueError(f'Category {cat} for variable {var} not in {cats}')
        df_copy[var] = pd.Categorical(df_copy[var], categories=cats)

    return df_copy


def get_cell_counts(
    df: pd.DataFrame, cell_vars: list[str], coords: dict
) -> pd.DataFrame:
    """
    Compute cell counts for the full cartesian product of cell_vars, filling missing cells with 0.
    """
    ordered_df = pd.DataFrame(
        it.product(*[coords[var] for var in cell_vars]), columns=cell_vars
    )
    ordered_df = ordered_df.merge(df, how='left', on=cell_vars)
    return (
        ordered_df.groupby(cell_vars, observed=False, sort=False)['N']
        .sum()
        .fillna(0)
        .astype('int64')
        .reset_index()
    )


def filter_survey_to_census_support(
    survey_df: pd.DataFrame,
    census_counts: pd.DataFrame,
    input_vars: list[str],
    *,
    report_prefix: str = 'model_FS',
) -> tuple[pd.DataFrame, int]:
    """
    Drop survey rows whose poststrat cell has zero census population (or cannot be matched),
    returning (filtered_df, n_removed).
    """
    census_n = census_counts[input_vars + ['N']]
    merged = survey_df[input_vars].merge(
        census_n,
        on=input_vars,
        how='left',
        validate='many_to_one',
    )
    ok = merged['N'].fillna(0).to_numpy() > 0
    n_removed = int((~ok).sum())
    if n_removed:
        print(
            f'{report_prefix}: removed {n_removed} survey observations in census-zero (or missing) cells'
        )
    return survey_df.loc[ok].reset_index(drop=True), n_removed


def normalize_interactions(
    input_vars: list[str],
    interactions: bool | Sequence[Sequence[str]] | None,
) -> list[tuple[str, ...]]:
    """
    Normalize interaction specification.

    Supported:
    - False/None: no interactions
    - True: all 2-way interactions between `input_vars`
    - Sequence of groups: e.g. [('age','sex'), ('gender','nationality','education')]
      (JSON-friendly lists also OK)
    """
    if not interactions:
        return []

    if interactions is True:
        return [(a, b) for a, b in it.combinations(input_vars, 2)]

    groups: list[tuple[str, ...]] = []
    seen = set()
    for ituple in interactions:
        if not isinstance(ituple, (list, tuple)) or len(ituple) < 2:
            raise ValueError(
                f'Invalid interaction spec {ituple!r}; expected a list/tuple of variable names, length >= 2.'
            )
        vars_ = tuple(ituple)
        if any(v not in input_vars for v in vars_):
            raise ValueError(
                f'Interaction {vars_!r} contains vars not in input_vars={input_vars}.'
            )
        if len(set(vars_)) != len(vars_):
            raise ValueError(
                f'Invalid interaction {vars_!r}; variables must be unique within an interaction.'
            )
        if vars_ in seen:
            continue
        seen.add(vars_)
        groups.append(vars_)

    return groups


@dataclass
class LinearPredictor:
    input_vars: list[str]
    interactions: bool | Sequence[Sequence[str]] | None
    prefix: str
    coefs: dict = field(default_factory=dict)


def linear_predictor_model(
    input_vars: list[str],
    interactions: bool | Sequence[Sequence[str]] | None,
    prefix: str = '',
    priors: dict | None = None,
    centered: bool = False,  # Non-centered is usually better for sampling
    reuse_coefs: dict | None = None,
) -> LinearPredictor:
    """
    Construct coefficients for the linear predictor including intercept, main effects, and interactions.
    Returns LinearPredictor object.
    """
    if priors is None:
        priors = {}

    coefs = {} if reuse_coefs is None else reuse_coefs.copy()

    # Intercept
    if f'{prefix}_intercept' not in coefs:
        coefs[f'{prefix}_intercept'] = priors[f'{prefix}_intercept'](f'{prefix}_intercept')

    # Main effects
    for var in input_vars:
        tau_name = f'{prefix}_{var}_tau'
        raw_effect_name = f'{prefix}_{var}_effect_raw'
        effect_name = f'{prefix}_{var}_effect'

        if tau_name not in coefs:
            coefs[tau_name] = priors[f'{prefix}_scale'](tau_name)

        if centered:
            if effect_name not in coefs:
                coefs[effect_name] = pm.ZeroSumNormal(
                    effect_name, sigma=coefs[tau_name], dims=[var]
                )
        else:
            if raw_effect_name not in coefs:
                coefs[raw_effect_name] = pm.ZeroSumNormal(
                    raw_effect_name, sigma=1.0, dims=[var]
                )
            if effect_name not in coefs:
                coefs[effect_name] = pm.Deterministic(
                    effect_name,
                    coefs[raw_effect_name] * coefs[tau_name],
                    dims=[var],
                )

    # Interaction effects
    interaction_vars = normalize_interactions(input_vars, interactions)

    for vars_ in interaction_vars:
        inter_key = '_'.join(vars_)
        tau_name = f'{prefix}_{inter_key}_tau'
        raw_effect_name = f'{prefix}_{inter_key}_effect_raw'
        effect_name = f'{prefix}_{inter_key}_effect'

        if tau_name not in coefs:
            coefs[tau_name] = priors[f'{prefix}_interaction_scale'](tau_name)

        if centered:
            if effect_name not in coefs:
                coefs[effect_name] = pm.ZeroSumNormal(
                    effect_name, sigma=coefs[tau_name], dims=list(vars_)
                )
        else:
            if raw_effect_name not in coefs:
                coefs[raw_effect_name] = pm.ZeroSumNormal(
                    raw_effect_name, sigma=1.0, dims=list(vars_)
                )
            if effect_name not in coefs:
                coefs[effect_name] = pm.Deterministic(
                    effect_name, coefs[raw_effect_name] * coefs[tau_name]
                )

    return LinearPredictor(
        input_vars=input_vars,
        interactions=interactions,
        prefix=prefix,
        coefs=coefs,
    )


def apply_linear_predictor(
    indices: dict[str, np.ndarray],
    lp: LinearPredictor,
) -> pt.TensorVariable:
    r"""
    Apply coefficients to construct the linear predictor (mu).
    Throws an error if any required coefficient is missing.
    Corresponds to the linear predictor $\eta$ defined in Equation (1) (Outcome)
    and Equation (5) (Selection) of the manuscript.
    """
    prefix = lp.prefix
    coefs = lp.coefs
    input_vars = lp.input_vars
    interactions = lp.interactions

    # Intercept
    if f'{prefix}_intercept' not in coefs:
        raise ValueError(f'Missing coefficient: {prefix}_intercept')
    mu = pt.zeros(len(indices[input_vars[0]])) + coefs[f'{prefix}_intercept']

    # Main effects
    for var in input_vars:
        effect_name = f'{prefix}_{var}_effect'
        if effect_name not in coefs:
            raise ValueError(f'Missing coefficient: {effect_name}')
        mu = mu + coefs[effect_name][indices[var]]

    # Interaction effects
    interaction_vars = normalize_interactions(input_vars, interactions)

    for vars_ in interaction_vars:
        inter_key = '_'.join(vars_)
        effect_name = f'{prefix}_{inter_key}_effect'
        if effect_name not in coefs:
            raise ValueError(f'Missing coefficient: {effect_name}')
        idx = tuple(indices[v] for v in vars_)
        mu = mu + coefs[effect_name][idx]

    return mu


def gg_correction(
    mu: pt.TensorVariable,
    indices: dict[str, np.ndarray],
    gg_vars: list[str],
    lp: LinearPredictor,
    priors: dict,
) -> pt.TensorVariable:
    """
    Apply Ghitza-Gelman correction (random effects on variables) to the linear predictor.
    Adds GG coefficients to the LinearPredictor in-place.
    """
    coefs = lp.coefs
    prefix = lp.prefix

    if len(gg_vars) == 0:
        var_name = f'{prefix}_gg_delta'
        # Use a flat-like prior
        if var_name not in coefs:
            coefs[var_name] = priors['gg_delta'](var_name)
        mu = mu + coefs[var_name]
        return mu

    # Main effects
    for var in gg_vars:
        var_name = f'{prefix}_{var}_gg_delta'
        # Use a flat-like prior
        if var_name not in coefs:
            coefs[var_name] = priors['gg_delta'](var_name, dims=var)
        mu = mu + coefs[var_name][indices[var]]

    # Ghitza-Gelman correction is only applied to the main effects, not the interaction effects

    return mu


def poststratify(
    model: pm.Model,
    census_df: pd.DataFrame,
    coords: dict,
    lp: LinearPredictor,
    priors: dict,
    gg_vars: list[str] | None = None,
) -> pm.Model:
    r"""
    Create a poststratification model to predict outcomes on census cells using fitted coefficients.

    Implements the aggregation described in Equation (3) to obtain region-level estimates ($p_r$)
    from cell-level probabilities ($\theta_c$).

    ``lp.coefs`` may contain numeric scales (e.g. fixed ``priors_scale_sigma``) that are not
    separate named variables on the fitted model; those are embedded in likelihoods on
    ``model`` but must be re-instantiated on the cloned ``ps_model`` with
    ``pt.as_tensor_variable``.
    """
    census_df = apply_categories(census_df, lp.input_vars, coords)
    census_counts = get_cell_counts(census_df, lp.input_vars, coords)
    census_indices = category_indices(census_counts, lp.input_vars, coords)

    with clone_model(model) as ps_model:
        # Map LP coefficients to the cloned model's variables, so posterior draws align.
        # Fixed hyperparameters (e.g. ``priors_scale_sigma``) are plain scalars in
        # ``lp.coefs`` and are not named on ``ps_model``; lift them into this graph.
        ps_coefs = {}
        for name, coef in lp.coefs.items():
            if name in ps_model.named_vars:
                ps_coefs[name] = ps_model[name]
            else:
                ps_coefs[name] = pt.as_tensor_variable(coef)
        ps_lp = LinearPredictor(
            input_vars=lp.input_vars,
            interactions=lp.interactions,
            prefix=lp.prefix,
            coefs=ps_coefs,
        )
        ps_mu = apply_linear_predictor(census_indices, ps_lp)
        if gg_vars is not None:
            ps_mu = gg_correction(ps_mu, census_indices, gg_vars, ps_lp, priors)
        ps_shape = [len(coords[var]) for var in lp.input_vars]
        ps_n = census_counts['N'].values.reshape(ps_shape)
        ps_p = pm.math.invprobit(ps_mu).reshape(ps_shape)
        ps_obs = np.zeros_like(ps_n)
        pm.Binomial(
            'poststratified_outcome',
            n=ps_n,
            p=ps_p,
            dims=lp.input_vars,
            observed=ps_obs,
        )

    return ps_model


def prepare_census_data(
    census_df: pd.DataFrame, input_vars: list[str]
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    """
    Prepare census data: extract coords, apply categories, get counts and indices.
    """
    coords = get_coords(census_df, input_vars)
    census_df = apply_categories(census_df, input_vars, coords)
    census_counts = get_cell_counts(census_df, input_vars, coords)
    census_indices = category_indices(census_counts, input_vars, coords)
    return coords, census_df, census_counts, census_indices


def prepare_margin_data(
    margin_df: pd.DataFrame,
    input_vars: list[str],
    outcome_var: str,
    coords: dict,
    census_counts: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Prepare margin data: validate categories and aggregate census counts to match margin structure.
    """
    margin_vars = margin_df.columns.drop([outcome_var, 'N']).tolist()
    margin_df = apply_categories(margin_df, margin_vars, coords)
    margin_df = margin_df[margin_df[outcome_var] == 'Yes'].drop(columns=[outcome_var])

    if len(margin_vars) == 0:
        margin_total_counts = census_counts['N'].sum().item()
        margin_observed = margin_df['N'].sum().item()
    else:
        validate_margin_data(margin_df, margin_vars, coords)

        margin_shape = [len(coords[var]) for var in margin_vars]
        margin_total_counts = get_cell_counts(census_counts, margin_vars, coords)[
            'N'
        ].values.reshape(margin_shape)
        margin_observed = get_cell_counts(margin_df, margin_vars, coords)[
            'N'
        ].values.reshape(margin_shape)

    return margin_vars, margin_observed, margin_total_counts


def prepare_survey_data(
    survey_df: pd.DataFrame,
    input_vars: list[str],
    outcome_var: str,
    coords: dict,
    survey_outcome_var: str | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """
    Prepare survey data: apply categories, get indices, and encode outcome.
    """
    if survey_outcome_var is None:
        survey_outcome_var = outcome_var

    survey_df = apply_categories(survey_df, input_vars, coords)
    survey_indices = category_indices(survey_df, input_vars, coords)

    survey_outcome_cats, K = encode_ordered_outcome(survey_df[survey_outcome_var])
    return survey_indices, survey_outcome_cats, K


def add_survey_likelihood(
    mu: pt.TensorVariable,
    observed: np.ndarray,
    dims: str,
    priors: dict,
    K: int = 2,
    name_prefix: str | None = None,
) -> None:
    """
    Add the survey outcome likelihood via ``pm.OrderedProbit``. The binary case
    (K=2) reduces to a probit link with a single zero cutpoint.
    """
    prefix = f'{name_prefix}_' if name_prefix else ''

    if K==2:
        survey_outcome_cutpoints = pt.zeros(1)
    else:
        survey_outcome_cutpoints = priors['outcome_cutpoints'](
            f'{prefix}survey_cutpoints',
            transform=pm.distributions.transforms.ChainedTransform(
                [
                    pm.distributions.transforms.ordered,
                    pm.distributions.transforms.ZeroSumTransform([0]),
                ]
            ),
            size=K - 1,
            initval=np.linspace(-1, 1, K - 1),
        )
    
    pm.OrderedProbit(
        f'{prefix}survey_obs',
        eta=mu,
        cutpoints=survey_outcome_cutpoints,
        observed=observed,
        dims=dims,
    )


def poisson_log_obs(name, log_lambda, obs):
    """
    Log-likelihood for Poisson(obs | lambda), ignoring constant terms involving obs.
    Non-log form: Poisson(k | lambda) = lambda^k * exp(-lambda) / k!
    log_p = k * log(lambda) - lambda - gammaln(k+1)
          ~ k * log_lambda - exp(log_lambda)
    """
    k = pt.as_tensor_variable(obs)
    log_p = k * log_lambda - pt.exp(log_lambda)
    pm.Potential(name, log_p.sum())


def add_margin_likelihood(
    mu: pt.TensorVariable,
    census_counts: pd.DataFrame,
    cell_shape: list[int],
    input_vars: list[str],
    margin_vars: list[str],
    margin_observed: np.ndarray,
    margin_total: np.ndarray,
    dist: str,
    prefix: str = 'cell',
    name_prefix: str | None = None,
) -> None:
    """
    Compute log-probability and log-expected-counts for cells using compute_log_cell_counts,
    then add likelihood for aggregated margin data.
    """
    _, cell_contrib = compute_log_cell_counts(
        mu, census_counts, cell_shape, input_vars, prefix=name_prefix
    )

    # Aggregation
    if len(margin_vars) == 0:
        sum_axes = None  # Sum all
    else:
        # Find axes that are NOT in margin_vars
        sum_axes = [idx for idx, var in enumerate(input_vars) if var not in margin_vars]

    # logsumexp over cell log-counts corresponds to summing expected counts in normal space:
    # margin_counts = sum_{c in margin} cell_counts
    log_margin_counts = pm.math.logsumexp(cell_contrib, axis=sum_axes)
    margin_counts = pm.math.exp(log_margin_counts)

    name_pref = f'{name_prefix}_' if name_prefix else ''

    # Non-log form: margin_prob = margin_counts / margin_total
    log_margin_total = pt.log(pt.as_tensor_variable(margin_total))
    log_margin_prob = log_margin_counts - log_margin_total
    margin_prob = pm.Deterministic(
        f'{name_pref}margin_prob', pt.exp(log_margin_prob), dims=margin_vars
    )

    match dist:
        case 'binomial':
            # Naive implementation:
            # pm.Binomial('margin_obs', n=margin_total, p=margin_prob, observed=margin_observed, dims=margin_vars)

            # Numerically-stable log-likelihood for Binomial(n, k, p) with log_p=log(p).
            # Omits the combinatorial factor log(nCk) as it is constant with respect to parameters.
            n_t = pt.as_tensor_variable(margin_total)
            k_t = pt.as_tensor_variable(margin_observed)

            # log(1-p) = log(1 - exp(log_p))
            bin_logp = k_t * log_margin_prob + (n_t - k_t) * pt.log1mexp(
                log_margin_prob
            )

            pm.Potential(f'{name_pref}margin_obs_logp', bin_logp.sum())
        case 'poisson':
            # Naive implementation:
            # pm.Poisson('margin_obs', mu=margin_counts, observed=margin_observed, dims=margin_vars)

            # We omit the constant term -gammaln(k+1) as it doesn't affect optimization/sampling of parameters
            poisson_log_obs(f'{name_pref}margin_obs_logp', log_margin_counts, margin_observed)

            # Still track expected count for inspection
            pm.Deterministic(f'{name_pref}margin_mu', margin_counts, dims=margin_vars)

        case 'betabinomial':
            # Clip to avoid numerical issues
            margin_prob_clipped = pm.math.clip(margin_prob, 1e-9, 1 - 1e-9)

            scale = pm.Exponential(f'{name_pref}obs_scale', lam=1.0 / 1000)
            pm.BetaBinomial(
                f'{name_pref}margin_obs',
                n=margin_total,
                alpha=margin_prob_clipped * scale,
                beta=(1 - margin_prob_clipped) * scale,
                observed=margin_observed,
                dims=margin_vars,
            )
        case 'potential':
            # Linear L1 loss - for GG
            margin_prob_clipped = pm.math.clip(margin_prob, 1e-9, 1 - 1e-9)
            pm.Potential(
                f'{name_pref}margin_abs_diff',
                -pt.abs(margin_prob_clipped * margin_total - margin_observed).sum(),
            )
        case _:
            raise ValueError(f'Invalid margin distribution: {dist}')


def compute_log_cell_counts(
    mu: pt.TensorVariable,
    census_counts: pd.DataFrame,
    cell_shape: list[int],
    input_vars: list[str],
    prefix: str = 'cell',
    nu: float | None = None,
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    """
    Compute log-probability and log-expected-counts for cells in log-space for stability.
    Non-log equivalent:
      prob_c = F(mu_c), where F is Phi (or Student-t CDF if nu is set)
      expected_count_c = prob_c * N_c
    Returns (log_prob, log_counts).
    """

    if nu is not None:  # Use student t if nu provided
        student_t_dist = pm.StudentT.dist(nu=nu, mu=0.0, sigma=1.0)
        log_prob = pm.logcdf(student_t_dist, mu).reshape(cell_shape)
    else:
        log_prob = logPhi(mu).reshape(cell_shape)  # Default is standard normal
    # Also deterministic prob for inspection/consistency, though we use log_prob for math
    pm.Deterministic(f'{prefix}_prob', pm.math.exp(log_prob), dims=input_vars)

    n_cells = pt.as_tensor_variable(census_counts['N'].values.reshape(cell_shape))
    # Non-log base is N_c; we clamp to avoid log(0) for empty cells.
    log_n_cells = pt.log(pt.maximum(n_cells, 1e-10)) # Avoid log(0)

    log_counts = log_prob + log_n_cells

    return log_prob, log_counts


def model_BP(
    census_df: pd.DataFrame,
    survey_dfs: list[pd.DataFrame],
    input_vars: list[str],
    outcome_var: str,
    interactions: bool | Sequence[Sequence[str]] | None = False,
    priors: dict | None = None,
    centered: bool = True,
) -> tuple[pm.Model, pm.Model]:
    """
    Basic Probit (BP) model: MRP with survey data only.
    """
    coords, census_df, census_counts, census_indices = prepare_census_data(
        census_df, input_vars
    )
    survey_items: list[tuple[dict[str, np.ndarray], np.ndarray, int, str]] = []
    for i, survey_df in enumerate(survey_dfs):
        survey_indices, survey_outcome, K = prepare_survey_data(
            survey_df, input_vars, outcome_var, coords
        )
        dims_name = f'survey_obs_idx_{i}'
        coords[dims_name] = list(range(len(survey_df)))
        survey_items.append((survey_indices, survey_outcome, K, dims_name))

    with pm.Model(coords=coords) as model:
        lp = linear_predictor_model(
            input_vars,
            interactions,
            prefix='outcome',
            priors=priors,
            centered=centered,
        )
        for i, (survey_indices, survey_outcome, K, dims_name) in enumerate(
            survey_items
        ):
            survey_mu = apply_linear_predictor(
                survey_indices,
                lp,
            )
            add_survey_likelihood(
                survey_mu,
                survey_outcome,
                dims_name,
                priors or {},
                K=K,
                name_prefix=f'survey_{i}',
            )

    ps_model = poststratify(model, census_df, coords, lp, priors)

    return model, ps_model


def model_EI(
    census_df: pd.DataFrame,
    margin_dfs: list[pd.DataFrame],
    input_vars: list[str],
    outcome_var: str,
    interactions: bool | Sequence[Sequence[str]] | None = False,
    priors: dict | None = None,
    centered: bool = True,
    margin_dist: str = 'binomial',
) -> tuple[pm.Model, pm.Model]:
    """
    Ecological Inference (EI) model: MRP with margin data only (no survey microdata).
    """
    coords, census_df, census_counts, census_indices = prepare_census_data(
        census_df, input_vars
    )
    with pm.Model(coords=coords) as model:
        cell_shape = [len(coords[var]) for var in input_vars]
        lp = linear_predictor_model(
            input_vars,
            interactions,
            prefix='outcome',
            priors=priors,
            centered=centered,
        )
        cell_mu = apply_linear_predictor(
            census_indices,
            lp,
        )

        # We use log-scale cell counts for numerical stability
        for i, margin_df in enumerate(margin_dfs):
            margin_vars, margin_observed, margin_total_counts = prepare_margin_data(
                margin_df, input_vars, outcome_var, coords, census_counts
            )
            add_margin_likelihood(
                cell_mu,
                census_counts,
                cell_shape,
                input_vars,
                margin_vars,
                margin_observed,
                margin_total_counts,
                margin_dist,
                name_prefix=f'margin_{i}',
            )

    ps_model = poststratify(model, census_df, coords, lp, priors)

    return model, ps_model


def model_GG(
    census_df: pd.DataFrame,
    survey_dfs: list[pd.DataFrame],
    margin_dfs: list[pd.DataFrame],
    input_vars: list[str],
    outcome_var: str,
    survey_outcome_var: str | None = None,
    interactions: bool | Sequence[Sequence[str]] | None = False,
    priors: dict | None = None,
    centered: bool = True,
    margin_dist: str = 'potential',
) -> tuple[pm.Model, pm.Model]:
    """
    Ghitza-Gelman (GG) model: MRP with survey + margin data and flexible correction terms.
    """
    if priors is None:
        priors = {}
    coords, census_df, census_counts, census_indices = prepare_census_data(
        census_df, input_vars
    )
    margin_vars_list: list[list[str]] = []
    for margin_df in margin_dfs:
        margin_vars, _, _ = prepare_margin_data(
            margin_df, input_vars, outcome_var, coords, census_counts
        )
        margin_vars_list.append(margin_vars)
    gg_vars = sorted({v for vars_ in margin_vars_list for v in vars_})
    survey_items: list[tuple[dict[str, np.ndarray], np.ndarray, int, str]] = []
    for i, survey_df in enumerate(survey_dfs):
        survey_indices, survey_outcome, K = prepare_survey_data(
            survey_df,
            input_vars,
            outcome_var,
            coords,
            survey_outcome_var=survey_outcome_var,
        )
        dims_name = f'survey_obs_idx_{i}'
        coords[dims_name] = list(range(len(survey_df)))
        survey_items.append((survey_indices, survey_outcome, K, dims_name))

    with pm.Model(coords=coords) as model:
        gg_post = pm.Data('gg_post', 0.0)

        cell_shape = [len(coords[var]) for var in input_vars]
        lp = linear_predictor_model(
            input_vars,
            interactions,
            prefix='outcome',
            priors=priors,
            centered=centered,
        )
        cell_mu = apply_linear_predictor(
            census_indices,
            lp,
        )
        # GG correction adds delta terms to the LP coefficients.
        cell_mu = gg_correction(cell_mu, census_indices, gg_vars, lp, priors)

        # GG-specific scaling
        cell_mu_scaled = gg_post * cell_mu

        # Log-space computation
        for i, margin_df in enumerate(margin_dfs):
            margin_vars, margin_observed, margin_total_counts = prepare_margin_data(
                margin_df, input_vars, outcome_var, coords, census_counts
            )
            add_margin_likelihood(
                cell_mu_scaled,
                census_counts,
                cell_shape,
                input_vars,
                margin_vars,
                margin_observed,
                margin_total_counts,
                margin_dist,
                name_prefix=f'margin_{i}',
            )

        for i, (survey_indices, survey_outcome, K, dims_name) in enumerate(
            survey_items
        ):
            survey_mu = apply_linear_predictor(
                survey_indices,
                lp,
            )
            survey_mu_scaled = (1.0 - gg_post) * survey_mu

            add_survey_likelihood(
                survey_mu_scaled,
                survey_outcome,
                dims_name,
                priors,
                K,
                name_prefix=f'survey_{i}',
            )

    ps_model = poststratify(model, census_df, coords, lp, priors, gg_vars=gg_vars)

    return model, ps_model


def model_GG_postprocess(model, idata):
    """
    Post-process GG model to apply correction term constraints/centering.
    """
    with model:
        pm.set_data({'gg_post': 1.0})

        gg_rvs = {}
        for rv in model.basic_RVs:
            if match := re.match(r'^(selection|outcome)_?(.*)_gg_delta$', rv.name):
                var_name = match.group(2)
                if ':' in var_name:
                    gg_rvs[rv] = tuple(var_name.split(':'))
                else:
                    gg_rvs[rv] = (var_name,) if var_name else None

        ivals = {
            rv.name: idata.posterior[rv.name].mean(['chain', 'draw']).values
            for rv in (set(model.basic_RVs) - set(gg_rvs.keys()))
            if rv.name in idata.posterior
        }

        opt = pm.find_MAP(
            start=ivals,
            vars=list(gg_rvs.keys()),
            tol=0.0,
            options={'ftol': 0.0, 'gtol': 0.0},
            maxeval=5000,
        )

        for k in ivals:
            assert ((ivals[k] - opt[k]) ** 2).sum() <= 1e-6, (
                f'Warning: {k} changed by {((ivals[k] - opt[k]) ** 2).sum()}, {ivals[k]} -> {opt[k]}'
            )

        idp = idata.posterior
        for rv, dims in gg_rvs.items():
            cd = (idp.sizes['chain'], idp.sizes['draw'])
            data = np.ones(cd + tuple(rv.shape.eval())) * opt[rv.name]

            if dims is None:
                data_dims = ('chain', 'draw')
                data_coords = {
                    **{'chain': idp.coords['chain'], 'draw': idp.coords['draw']}
                }
            else:
                data_dims = ('chain', 'draw') + dims
                data_coords = {
                    **{'chain': idp.coords['chain'], 'draw': idp.coords['draw']},
                    **{d: idp.coords[d] for d in dims},
                }

            idp[rv.name] = xr.DataArray(data, dims=data_dims, coords=data_coords)

    return idata


def model_PM(
    census_df: pd.DataFrame,
    survey_dfs: list[pd.DataFrame],
    margin_dfs: list[pd.DataFrame],
    input_vars: list[str],
    outcome_var: str,
    survey_outcome_var: str | None = None,
    interactions: bool | Sequence[Sequence[str]] | None = False,
    priors: dict | None = None,
    centered: bool = True,
    margin_dist: str = 'binomial',
) -> tuple[pm.Model, pm.Model]:
    """
    Polls & Margins (PM) model: MRP with survey + margin data (combining BP and EI).
    """
    if priors is None:
        priors = {}
    coords, census_df, census_counts, census_indices = prepare_census_data(
        census_df, input_vars
    )
    survey_items: list[tuple[dict[str, np.ndarray], np.ndarray, int, str]] = []
    for i, survey_df in enumerate(survey_dfs):
        survey_indices, survey_outcome, K = prepare_survey_data(
            survey_df,
            input_vars,
            outcome_var,
            coords,
            survey_outcome_var=survey_outcome_var,
        )
        dims_name = f'survey_obs_idx_{i}'
        coords[dims_name] = list(range(len(survey_df)))
        survey_items.append((survey_indices, survey_outcome, K, dims_name))

    with pm.Model(coords=coords) as model:
        cell_shape = [len(coords[var]) for var in input_vars]
        lp = linear_predictor_model(
            input_vars,
            interactions,
            prefix='outcome',
            priors=priors,
            centered=centered,
        )
        cell_mu = apply_linear_predictor(
            census_indices,
            lp,
        )

        

        # Log-space computation
        for i, margin_df in enumerate(margin_dfs):
            margin_vars, margin_observed, margin_total_counts = prepare_margin_data(
                margin_df, input_vars, outcome_var, coords, census_counts
            )
            add_margin_likelihood(
                cell_mu,
                census_counts,
                cell_shape,
                input_vars,
                margin_vars,
                margin_observed,
                margin_total_counts,
                margin_dist,
                name_prefix=f'margin_{i}',
            )

        for i, (survey_indices, survey_outcome, K, dims_name) in enumerate(
            survey_items
        ):
            survey_mu = apply_linear_predictor(
                survey_indices,
                lp,
            )

            survey_mu += priors['overreporting'](f'survey_{i}_or_bias')

            add_survey_likelihood(
                survey_mu,
                survey_outcome,
                dims_name,
                priors,
                K,
                name_prefix=f'survey_{i}',
            )

    ps_model = poststratify(model, census_df, coords, lp, priors)

    return model, ps_model

def model_FS(
    census_df: pd.DataFrame,
    survey_dfs: list[pd.DataFrame],
    selection_survey_dfs: list[pd.DataFrame] | None,
    margin_dfs: list[pd.DataFrame],
    input_vars: list[str],
    outcome_var: str,
    survey_outcome_var: str | None = None,
    selection_vars: list[str] | None = None,
    interactions: bool | Sequence[Sequence[str]] | None = False,
    priors: dict | None = None,
    centered: bool = True,
    margin_dist: str = 'binomial',
    robust_selection: bool = False,
) -> tuple[pm.Model, pm.Model]:
    """
    Full Selection (FS) model: Heckman selection model on top of PM structure (survey selection bias correction).

    See Section 3.3 "Full Selection (FS)", Equations (4)-(8).
    """
    if priors is None:
        priors = {}
    if selection_survey_dfs is None:
        selection_survey_dfs = survey_dfs
    selection_vars = input_vars if selection_vars is None else selection_vars
    if len(selection_vars) == 0:
        raise ValueError('selection_vars must contain at least one variable.')

    coords, census_df, census_counts, census_indices = prepare_census_data(
        census_df, input_vars
    )
    selection_census_counts = get_cell_counts(census_counts, selection_vars, coords)
    selection_census_indices = category_indices(selection_census_counts, selection_vars, coords)
    selection_cell_shape = [len(coords[var]) for var in selection_vars]
    non_zero_selection_cells = np.where(selection_census_counts['N'] > 0)[0]
    coords['non_zero_selection_obs_idx'] = list(range(len(non_zero_selection_cells)))

    selection_frames: list[pd.DataFrame] = []
    for i, selection_df in enumerate(selection_survey_dfs):
        selection_df = apply_categories(selection_df, selection_vars, coords)
        selection_df, n_removed = filter_survey_to_census_support(
            selection_df,
            selection_census_counts,
            selection_vars,
            report_prefix=f'model_FS_selection_{i}',
        )
        selection_frames.append(selection_df[selection_vars].copy())
    selection_df = pd.concat(selection_frames, ignore_index=True)
    selection_cell_counts = get_cell_counts(
        selection_df.assign(N=1), selection_vars, coords
    )

    # Ensure alignment with selection_census_counts
    selection_cell_counts = selection_census_counts[selection_vars].merge(
        selection_cell_counts,
        on=selection_vars,
        how='left',
    )
    selection_cell_counts['N'] = selection_cell_counts['N'].fillna(0).astype(int)

    with pm.Model(coords=coords) as model:
        # Cell-level outcome process
        cell_shape = [len(coords[var]) for var in input_vars]
        lp_outcome = linear_predictor_model(
            input_vars,
            interactions,
            prefix='outcome',
            priors=priors,
            centered=centered,
        )
        cell_o_mu = apply_linear_predictor(
            census_indices,
            lp_outcome,
        )

        for i, margin_df in enumerate(margin_dfs):
            margin_vars, margin_observed, margin_total_counts = prepare_margin_data(
                margin_df, input_vars, outcome_var, coords, census_counts
            )
            add_margin_likelihood(
                cell_o_mu,
                census_counts,
                cell_shape,
                input_vars,
                margin_vars,
                margin_observed,
                margin_total_counts,
                margin_dist,
                prefix='cell_o',
                name_prefix=f'margin_{i}',
            )
    
        lp_selection = linear_predictor_model(
            selection_vars,
            interactions,
            prefix='selection',
            priors=priors,
            centered=centered,
        )
        cell_s_mu = apply_linear_predictor(
            selection_census_indices,
            lp_selection,
        )

        # Robust selection uses Student T distribution with gamma distributed variance
        nu = priors['selection_nu']('selection_nu') if robust_selection else None
        
        # Log-space for selection
        cell_s_logprob, log_cell_s_counts = compute_log_cell_counts(
            cell_s_mu,
            selection_census_counts,
            selection_cell_shape,
            selection_vars,
            prefix='cell_s',
            nu=nu,
        )

        # Selection Multinomial using log counts
        log_cell_s_counts_non_zero = log_cell_s_counts.flatten()[non_zero_selection_cells]

        # Non-log form (Eq. selection multinomial):
        # pi_i^s = (Phi(eta_i^s) * N_i) / sum_j (Phi(eta_j^s) * N_j)
        # Here log_cell_s_counts = log(Phi(eta_i^s) * N_i), and softmax performs
        # exp(log_count_i) / sum_j exp(log_count_j) in a stable way.
        # Corresponds to Equation (8) for identifying selection bias under random contact.
        selection_prob = pm.Deterministic(
            'selection_prob',
            pm.math.softmax(log_cell_s_counts_non_zero),
            dims='non_zero_selection_obs_idx',
        )
        sel_obs = selection_cell_counts['N'].values[non_zero_selection_cells]
        
        pm.Multinomial(
            'selection_obs',
            n=sel_obs.sum(),
            p=selection_prob,
            observed=sel_obs,
            dims='non_zero_selection_obs_idx',
        )

        rho = priors['rho']('rho')

        for i, survey_df in enumerate(survey_dfs):
            survey_df = apply_categories(survey_df, input_vars, coords)
            survey_df, n_removed = filter_survey_to_census_support(
                survey_df,
                census_counts,
                input_vars,
                report_prefix=f'model_FS_{i}',
            )
            survey_indices, survey_outcome, K = prepare_survey_data(
                survey_df,
                input_vars,
                outcome_var,
                coords,
                survey_outcome_var=survey_outcome_var,
            )
            selection_survey_indices = category_indices(survey_df, selection_vars, coords)
            dims_name = f'survey_obs_idx_{i}'
            model.add_coord(dims_name, list(range(len(survey_df))))

            # Heckman selection model
            survey_s_mu = apply_linear_predictor(
                selection_survey_indices,
                lp_selection,
            )
            survey_o_mu = apply_linear_predictor(
                survey_indices,
                lp_outcome,
            )

            # Poll Response Likelihood (bivariate probit, conditioning on selection)
            lb = -survey_s_mu
            if robust_selection:
                # Student T can be expressed as a mixture of normals with gamma distributed variance
                wv = pm.Gamma(f'survey_{i}_wv', nu/2, nu/2, dims=dims_name)
                lb *= pt.sqrt(wv)

            u = pm.TruncatedNormal(
                f'u_{i}',
                mu=0.0,
                sigma=1.0,
                lower=lb,
                upper=np.inf,
                dims=dims_name,
            )

            # 1.0001 rather than 1.0 guards the denominator against rho -> +-1
            # (rho is bounded to (-1, 1) by its 2*Beta-1 prior, so this only bites
            # in the extreme tail); the resulting bias in the correction is negligible.
            survey_o_mu_scaled = (survey_o_mu + rho * u) / pt.sqrt(1.0001 - rho**2)

            # ERRATUM (original code used to produce the paper's results): the
            # overreporting shift is applied to the *conditional* outcome
            # probability, i.e. AFTER the (eta + rho*u)/sqrt(1-rho^2) scaling,
            # rather than to eta before scaling as specified in the paper. The
            # released `main` branch applies it before scaling; see the Erratum in
            # Appendix A2. The two forms are equivalent up to the reparameterization
            # beta_or -> beta_or * sqrt(1 - rho^2), so the achievable fit -- and
            # hence the reported results -- are unchanged. This branch preserves the
            # original placement for exact replication.
            survey_o_mu_scaled = survey_o_mu_scaled + priors['overreporting'](
                f'survey_{i}_or_bias'
            )

            add_survey_likelihood(
                survey_o_mu_scaled,
                survey_outcome,
                dims_name,
                priors,
                K,
                name_prefix=f'survey_{i}',
            )

    ps_model = poststratify(model, census_df, coords, lp_outcome, priors)

    return model, ps_model


def mean_rhat(idata) -> float:
    """Mean R-hat across sampled parameters."""
    import arviz as az

    rhat = az.rhat(idata)
    ds = rhat.dataset if hasattr(rhat, 'dataset') else rhat
    values = np.concatenate(
        [
            np.asarray(var.values, dtype=float).ravel()
            for var in ds.data_vars.values()
        ]
    )
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float('nan')


def run_model(config, sample_kwargs: dict | None = None, save_path: str | None = None):
    """
    Build, sample, and poststratify one model from a configuration dictionary.

    This is the primary entry point of the package. It dispatches on
    ``config['model_type']`` to the corresponding model builder, samples with NUTS,
    poststratifies onto the census cells, and returns the fitted objects.

    Parameters
    ----------
    config : dict
        Model configuration. Data-frame fields (``population``, ``survey``,
        ``margin``, ``selection_survey``) accept either a ``pandas.DataFrame`` or a
        path to a ``.csv``/``.parquet`` file; ``survey``/``margin`` also accept a
        list to combine several polls or margin tables. Keys:

        - ``name`` (str): label for the run (printed at start; **required**).
        - ``model_type`` (str): one of ``'BP'``, ``'EI'``, ``'GG'``, ``'PM'``, ``'FS'``.
        - ``input_cols`` (list[str]): demographic dimensions defining the cells.
        - ``outcome_col`` (str): binary/ordinal outcome column (e.g. ``'voting_intent'``).
        - ``population`` (df|path): census cell counts (needs an ``'N'`` column).
        - ``survey`` (df|path|list): respondent-level data (BP, GG, PM, FS).
        - ``margin`` (df|path|list): official margin table(s) (EI, GG, PM, FS).
        - ``survey_outcome_col`` (str, optional): survey outcome column if it differs
          from ``outcome_col``.
        - ``interactions`` (bool | list[tuple], default ``False``): ``True`` adds all
          pairwise interactions of ``input_cols``; a list specifies groups explicitly.
        - ``margin_dist`` (str): margin likelihood; ``'binomial'`` (default for
          EI/PM/FS), ``'potential'`` (default for GG), ``'betabinomial'``, or ``'poisson'``.
        - ``centered`` (bool): centered vs non-centered parameterization
          (default: ``True`` for EI/PM/FS, ``False`` for BP/GG).
        - ``priors`` (dict, optional): overrides merged over ``DEFAULT_PRIORS``.
        - ``selection_cols`` (list[str], FS only): selection-equation covariates
          (defaults to ``input_cols``).
        - ``selection_survey`` (df|path|list, FS only): survey used for the selection
          multinomial (defaults to ``survey``).
        - ``robust_selection`` (bool, FS only, default ``False``): Student-t selection.
    sample_kwargs : dict, optional
        Overrides merged over the defaults (nutpie NUTS, 2 chains, 800 tune,
        500 draws, ``target_accept=0.9``).
    save_path : str, optional
        If given, writes ``idata.nc`` and ``draws.parquet`` under this directory.

    Returns
    -------
    dict
        ``{'model', 'post_model', 'idata', 'post_idata', 'draws'}`` — the fitted
        model, the poststratification model, the posterior, the poststratified
        posterior predictive, and a tidy per-cell draws data frame.
    """
    default_sample_kwargs = {
        'nuts_sampler': 'nutpie',
        'chains': 2,
        'tune': 800,
        'draws': 500,
        'target_accept': 0.9,
    }
    sample_kwargs = (
        default_sample_kwargs
        if sample_kwargs is None
        else default_sample_kwargs | sample_kwargs
    )

    input_vars = config['input_cols']
    outcome_var = config['outcome_col']
    interactions = config.get('interactions', False)
    
    priors = DEFAULT_PRIORS | config.get('priors', {})
    
    default_centered = config['model_type'] in {'EI', 'PM', 'FS'}
    centered = config.get('centered', default_centered)
    survey_outcome_var = config.get('survey_outcome_col', outcome_var)
    cat_dtype = {
        col: 'category' for col in input_vars + [outcome_var, survey_outcome_var]
    }

    def _get_df(spec, *, dtype=None) -> pd.DataFrame:
        """
        Helper to resolve data specification to a DataFrame (file path, dict, or direct DF).
        """
        if isinstance(spec, pd.DataFrame):
            return spec
        if isinstance(spec, str):
            if str(spec).endswith('.parquet'):
                return pd.read_parquet(spec)
            return pd.read_csv(spec, dtype=dtype)
        raise ValueError(
            f'Expected a DataFrame, or a string with a file path. Got {spec!r}'
        )

    def _get_df_list(specs, *, dtype=None) -> list[pd.DataFrame]:
        if isinstance(specs, (list, tuple)):
            return [_get_df(spec, dtype=dtype) for spec in specs]
        return [_get_df(specs, dtype=dtype)]

    census_df = _get_df(config['population'], dtype=cat_dtype)

    print(config['name'])
    match config['model_type']:
        case 'BP':
            survey_dfs = _get_df_list(config['survey'], dtype=cat_dtype)
            model, ps_model = model_BP(
                census_df,
                survey_dfs,
                input_vars,
                outcome_var,
                interactions,
                priors,
                centered,
            )
        case 'EI':
            margin_dfs = _get_df_list(config['margin'], dtype=cat_dtype)
            margin_dist = config.get('margin_dist', 'binomial')
            model, ps_model = model_EI(
                census_df,
                margin_dfs,
                input_vars,
                outcome_var,
                interactions,
                priors,
                centered,
                margin_dist,
            )
        case 'GG':
            survey_dfs = _get_df_list(config['survey'], dtype=cat_dtype)
            margin_dfs = _get_df_list(config['margin'], dtype=cat_dtype)
            margin_dist = config.get('margin_dist', 'potential')
            model, ps_model = model_GG(
                census_df,
                survey_dfs,
                margin_dfs,
                input_vars,
                outcome_var,
                survey_outcome_var=survey_outcome_var,
                interactions=interactions,
                priors=priors,
                centered=centered,
                margin_dist=margin_dist,
            )
        case 'PM':
            survey_dfs = _get_df_list(config['survey'], dtype=cat_dtype)
            margin_dfs = _get_df_list(config['margin'], dtype=cat_dtype)
            margin_dist = config.get('margin_dist', 'binomial')
            model, ps_model = model_PM(
                census_df,
                survey_dfs,
                margin_dfs,
                input_vars,
                outcome_var,
                survey_outcome_var=survey_outcome_var,
                interactions=interactions,
                priors=priors,
                centered=centered,
                margin_dist=margin_dist,
            )
        case 'FS':
            survey_dfs = _get_df_list(config['survey'], dtype=cat_dtype)
            selection_survey_dfs = config.get('selection_survey', None)
            selection_vars = config.get('selection_cols', input_vars)
            selection_survey_dfs = (
                None
                if selection_survey_dfs is None
                else _get_df_list(selection_survey_dfs, dtype=cat_dtype)
            )
            margin_dfs = _get_df_list(config['margin'], dtype=cat_dtype)
            margin_dist = config.get('margin_dist', 'binomial')
            robust_selection = config.get('robust_selection', False)
            model, ps_model = model_FS(
                census_df,
                survey_dfs,
                selection_survey_dfs,
                margin_dfs,
                input_vars,
                outcome_var,
                survey_outcome_var=survey_outcome_var,
                selection_vars=selection_vars,
                interactions=interactions,
                priors=priors,
                centered=centered,
                margin_dist=margin_dist,
                robust_selection=robust_selection,
            )
        case _:
            raise ValueError(f'Unknown model_type: {config["model_type"]}')

    with model:
        idata = pm.sample(**sample_kwargs)

    # Post-processing step for GG
    if config['model_type'] == 'GG':
        idata = model_GG_postprocess(model, idata)

    # Sample post-stratification
    with ps_model:
        post_idata = pm.sample_posterior_predictive(
            idata, var_names=['poststratified_outcome']
        )

    draws_df = (
        post_idata.posterior_predictive['poststratified_outcome']
        .rename('N')
        .to_dataframe()
        .reset_index()
    )
    draws_df = pd.merge(
        draws_df,
        census_df[input_vars + ['N']],
        on=input_vars,
        how='left',
        suffixes=('', '_census'),
    )
    draws_df['N_census'] = draws_df['N_census'].fillna(0).astype('int64')

    if save_path is not None:
        prefix_path = pathlib.Path(save_path)
        prefix_path.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(prefix_path / 'idata.nc')
        # post_idata.to_netcdf(prefix_path / 'post_idata.nc')
        draws_df.to_parquet(prefix_path / 'draws.parquet')

    return {
        'model': model,
        'post_model': ps_model,
        'idata': idata,
        'post_idata': post_idata,
        'draws': draws_df,
    }
