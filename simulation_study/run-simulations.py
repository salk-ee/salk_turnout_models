#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# In[ ]:


import os
import sys

# Adjust import path to import turnout models
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# In[ ]:


import copy
import datetime
import hashlib
import itertools as it
import json
import logging
import pathlib
import shutil
import traceback
import arviz as az
import numpy as np
import pandas as pd
import statsmodels.api as sm
import salk_turnout_models as tm
from time import perf_counter
from tqdm.notebook import tqdm


# In[ ]:


import warnings

warnings.filterwarnings('ignore', module='arviz', message='invalid value encountered in scalar divide')


# In[ ]:


LOGGER = logging.getLogger('synth-data-models')
LOGGER.setLevel(logging.DEBUG)


# ## Models

# In[ ]:


def df_margins(pdf, columns, outcome):
    if 'N' in pdf.columns:
        if columns:
            return (pdf.groupby(columns + [outcome], observed=False)['N'].sum() / pdf['N'].sum()).rename('proportion')
        else:
            return (pdf.groupby(outcome, observed=False)['N'].sum() / pdf['N'].sum()).rename('proportion')
    else:
        if columns:
            return (pdf.groupby(columns, observed=False)[outcome].value_counts() / len(pdf)).rename('proportion')
        else:
            return (pdf[outcome].value_counts() / len(pdf)).rename('proportion')

def df_turnout(pdf, columns, outcome):
    if 'N' in pdf.columns:
        if columns:
            groups = pdf.groupby(columns, observed=True)
            return (pdf.groupby(columns + [outcome], observed=True)['N'].sum() / groups['N'].sum()).rename('proportion')
        else:
            return (pdf.groupby(outcome, observed=True)['N'].sum() / pdf['N'].sum()).rename('proportion')
    else:
        if columns:
            groups = pdf.groupby(columns, observed=True)
            return (groups[outcome].value_counts() / groups[outcome].size()).rename('proportion')
        else:
            return (pdf[outcome].value_counts() / len(pdf)).rename('proportion')

def cell_margins(df, cell_cols, margin_cols, outcome):
    avg_df = df.groupby(cell_cols)[['N', 'N_census']].mean().reset_index()

    if margin_cols:
        grouped_df = avg_df.groupby(margin_cols)[['N', 'N_census']].sum().reset_index()
    else:
        grouped_df = pd.DataFrame({'N': avg_df['N'].sum(), 'N_census': avg_df['N_census'].sum()}, index=[0])

    yes_df = grouped_df.copy()
    yes_df['proportion'] = yes_df['N'] / yes_df['N_census'].sum()
    yes_df[outcome] = 'Yes'

    no_df = grouped_df.copy()
    no_df['proportion'] = (no_df['N_census'] - no_df['N']) / no_df['N_census'].sum()
    no_df[outcome] = 'No'

    return pd.concat([yes_df, no_df])[margin_cols + [outcome, 'proportion']].set_index(margin_cols + [outcome])

def cell_turnout(df, cell_cols, margin_cols, outcome):
    avg_df = df.groupby(cell_cols)[['N', 'N_census']].mean().reset_index()

    if margin_cols:
        grouped_df = avg_df.groupby(margin_cols)[['N', 'N_census']].sum().reset_index()
    else:
        grouped_df = pd.DataFrame({'N': avg_df['N'].sum(), 'N_census': avg_df['N_census'].sum()}, index=[0])

    yes_df = grouped_df.copy()
    yes_df['proportion'] = yes_df['N'] / yes_df['N_census']
    yes_df[outcome] = 'Yes'

    no_df = grouped_df.copy()
    no_df['proportion'] = (no_df['N_census'] - no_df['N']) / no_df['N_census']
    no_df[outcome] = 'No'

    return pd.concat([yes_df, no_df])[margin_cols + [outcome, 'proportion']].set_index(margin_cols + [outcome])

def kl_divergence(margins_df, epsilon=1e-10):
    # Clip values to avoid zero division and log(0)
    p = np.clip(margins_df['proportion_pop'].values.flatten(), epsilon, 1)
    q = np.clip(margins_df['proportion_mod'].values.flatten(), epsilon, 1)
    return np.sum(p * np.log(p / q)).item()

def em_distance(margins_df, epsilon=1e-10):
    # Clip values to avoid zero division and log(0)
    p = np.clip(margins_df['proportion_pop'].values.flatten(), epsilon, 1)
    q = np.clip(margins_df['proportion_mod'].values.flatten(), epsilon, 1)
    return np.abs(p - q).sum().item() / 2

def get_distances(pop_df, mod_df, columns, outcome):
    pop_margins = df_margins(pop_df, columns, 'voting_intent').reset_index()
    mod_margins = cell_margins(mod_df, columns, columns, 'voting_intent').reset_index()

    margins_df = pd.merge(pop_margins, mod_margins, on=columns + [outcome], how='outer', suffixes=('_pop', '_mod')).fillna(0)

    emd = em_distance(margins_df)
    kld = kl_divergence(margins_df)

    emd_1d = np.array([em_distance(margins_df.groupby([col, outcome])[['proportion_pop', 'proportion_mod']].sum()) for col in columns])
    kld_1d = np.array([kl_divergence(margins_df.groupby([col, outcome])[['proportion_pop', 'proportion_mod']].sum()) for col in columns])

    emd_2d = np.array([em_distance(margins_df.groupby([c1, c2, outcome])[['proportion_pop', 'proportion_mod']].sum()) for c1, c2 in it.combinations(columns, 2)])
    kld_2d = np.array([kl_divergence(margins_df.groupby([c1, c2, outcome])[['proportion_pop', 'proportion_mod']].sum()) for c1, c2 in it.combinations(columns, 2)])

    topline_margins = {
        'topline_yes': mod_margins[mod_margins.voting_intent == 'Yes']['proportion'].sum().item(),
        'pop_topline_yes': pop_margins[pop_margins.voting_intent == 'Yes']['proportion'].sum().item(),
    }

    return {
        'kld': kld,
        'kld_1d': kld_1d.mean().item(),
        'kld_2d': kld_2d.mean().item(),
        'emd': emd,
        'emd_1d': emd_1d.mean().item(),
        'emd_2d': emd_2d.mean().item(),
    } | topline_margins

def get_coefs(model_path, posterior=None):
    if posterior is None:
        posterior = az.from_netcdf(model_path / 'idata.nc').posterior
    heckman_coefs = json.load(open(model_path / 'heckman_coefs.json'))

    var_names = {
        'selection': 'selection',
        'outcome': 'outcome',
    }

    ignore_cols = ['selection_latent']

    dfs = []

    for p in ['selection', 'outcome']:
        vname = f'{var_names[p]}_%s_effect'
        hcoefs = heckman_coefs[p].get('beta', {})

        mvals = pd.Series()
        rvals = pd.Series()

        for c in hcoefs:
            cname = c

            if ':' in c:
                c1, c2 = c.split(':')

                if c1 in ignore_cols or c2 in ignore_cols:
                    continue

                index = []

                if vname % f'{c1},{c2}' in posterior:
                    c1, c2 = c1, c2

                    for cat in hcoefs[c]:
                        cat1, cat2 = cat.split(':')
                        index.append((cat1, cat2))
                elif vname % f'{c2},{c1}' in posterior:
                    c1, c2 = c2, c1

                    for cat in hcoefs[c]:
                        cat2, cat1 = cat.split(':')
                        index.append((cat1, cat2))
                else:
                    continue

                cname = f'{c1}:{c2}'
                rvals = pd.Series(hcoefs[c].values(), index=pd.MultiIndex.from_tuples(index, names=[c1, c2])).rename('real')
                mvals = posterior[vname % f'{c1},{c2}'].mean(dim=['chain','draw']).to_series().rename('model')
            else:
                rvals = pd.Series(hcoefs[c]).rename('real')
                if vname % c not in posterior: continue
                mvals = posterior[vname % c].mean(dim=['chain','draw']).to_series().rename('model')

            rvals -= rvals.mean()
            mvals -= mvals.mean()

            df = pd.concat([rvals, mvals], axis=1)
            df['index'] = list(df.index)
            df['process'] = p
            df['var'] = cname
            df.reset_index(drop=True, inplace=True)

            dfs.append(df)

    if len(dfs) == 0:
        return pd.DataFrame({'process': [], 'var': [], 'index': [], 'real': [], 'model': []})

    coefs = pd.concat(dfs)

    return coefs[['process', 'var', 'index', 'real', 'model']]


# In[ ]:


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
    """
    def __init__(self, stream, logger, level):
        self.stream = stream
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
        self.stream.write(buf)

    def flush(self):
        self.stream.flush()

class CaptureStdStreams:
    def __init__(self, logger):
        self.logger = logger

    def __enter__(self):
        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr
        sys.stdout = self.stdout = StreamToLogger(sys.stdout, self.logger, logging.INFO)
        sys.stderr = self.stderr = StreamToLogger(sys.stderr, self.logger, logging.ERROR)

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.sys_stdout
        sys.stderr = self.sys_stderr

class ModelLogger:
    def __init__(self, root_logger, mname, mpath):
        self.root_logger = root_logger
        self.logger = root_logger.getChild(mname)

        self.formatter = logging.Formatter(
            fmt='[{name}] {asctime} {levelname}: {message}',
            datefmt='%m/%d/%Y %H:%M:%S',
            style='{'
        )

        self.fh = logging.FileHandler(mpath / 'log.txt', mode='w')
        self.fh.setFormatter(self.formatter)
        self.logger.addHandler(self.fh)

    def __enter__(self):
        return self.logger

    def __exit__(self, type, value, traceback):
        pass

class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start

def dict_apply(obj, key, func):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                obj[k] = func(v)
            else:
                obj[k] = dict_apply(v, key, func)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = dict_apply(v, key, func)
    return obj

def get_file_hash(filename, root=None):
    if root:
        file_path = pathlib.Path(root) / filename
    else:
        file_path = pathlib.Path(filename)

    if file_path.suffix == '.json':
        meta = json.load(open(file_path, 'r'))
        meta = dict_apply(copy.deepcopy(meta), 'file', lambda fn: get_file_hash(fn, file_path.parent))
        return hashlib.md5(json.dumps(meta, sort_keys=True).encode('utf-8')).hexdigest()
    else:
        return hashlib.sha256(open(file_path, 'rb').read()).hexdigest()

def model_desc_json_safe(obj):
    """JSON-friendly snapshot for hashing and model_desc.json; callables become code fingerprints."""
    if isinstance(obj, dict):
        return {k: model_desc_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [model_desc_json_safe(v) for v in obj]
    if callable(obj):
        code = getattr(obj, '__code__', None)
        if code is None:
            return {'__callable__': False, 'type': type(obj).__name__, 'repr': repr(obj)}
        safe_consts = []
        for c in code.co_consts:
            if isinstance(c, (int, float, str, bool, type(None))):
                safe_consts.append(c)
            elif isinstance(c, tuple) and all(
                isinstance(x, (int, float, str, bool, type(None))) for x in c
            ):
                safe_consts.append(c)
        kd = getattr(obj, '__kwdefaults__', None)
        return {
            '__callable__': True,
            'defaults': model_desc_json_safe(list(obj.__defaults__))
            if obj.__defaults__ is not None
            else None,
            'kwdefaults': model_desc_json_safe(dict(kd)) if kd else None,
            'consts': safe_consts,
            'varnames': list(code.co_varnames),
            'argcount': int(code.co_argcount),
        }
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return model_desc_json_safe(obj.tolist())
    return str(obj)

def get_model_id(model_desc):
    desc = copy.deepcopy(model_desc)
    # Ignore model name in ID calculation
    desc.pop('name')
    # Replace file paths with the hashes of the file contents
    desc = dict_apply(desc, 'file', get_file_hash)
    return hashlib.md5(
        json.dumps(model_desc_json_safe(desc), sort_keys=True).encode('utf-8')
    ).hexdigest()

def run_model(model_desc, population_file, model_data, path_prefix='./tmp', root_logger=LOGGER, progress=None, progress_postfix=None):
    def aggregate_draws_to_chain_means(draws_df: pd.DataFrame) -> pd.DataFrame:
        # Collapse posterior draws to per-chain means (drops `draw`), keeping cell dims.
        if 'draw' not in draws_df.columns or 'chain' not in draws_df.columns or 'N' not in draws_df.columns:
            return draws_df

        group_cols = [c for c in draws_df.columns if c not in ['N', 'draw']]
        if 'N_census' in draws_df.columns:
            group_cols_wo_nc = [c for c in group_cols if c != 'N_census']
            agg_df = (
                draws_df
                .groupby(group_cols_wo_nc, observed=False, sort=False)
                .agg(N=('N', 'mean'), N_census=('N_census', 'first'))
                .reset_index()
            )
        else:
            agg_df = (
                draws_df
                .groupby(group_cols, observed=False, sort=False)['N']
                .mean()
                .reset_index()
            )

        return agg_df

    model_id = get_model_id(model_desc)
    model_name = f'{model_desc["name"]}-{model_id}'
    model_path = pathlib.Path(path_prefix) / model_id
    model_path.mkdir(parents=True, exist_ok=True)

    link_path = pathlib.Path(path_prefix) / model_name
    if not link_path.is_symlink():
        os.symlink(model_path.name, link_path)

    if progress:
        if progress_postfix:
            progress.set_postfix({**progress_postfix, 'model': model_name})
        else:
            progress.set_postfix({'model': model_name})
    else:
        print('Model:', model_name)

    model_path.mkdir(parents=True, exist_ok=True)

    summary_path = model_path / 'summary.json'

    print(model_path / 'idata.nc', (model_path / 'idata.nc').exists())

    if summary_path.exists(): # and not (model_path / 'idata.nc').exists():
        summary_data = json.load(open(summary_path, 'r'))
        # Cleanup legacy idata.nc if present

        return {
            'model_path': str(model_path),
            'model_id': model_id,
            'model_name': model_desc['name'],
        } | summary_data


    # Load or fit model
    if (model_path / 'draws.parquet').exists():
        mod_df = pd.read_parquet(model_path / 'draws.parquet')
        fit_time = float(open(model_path / 'time.txt').read())
    else:
        with ModelLogger(root_logger, model_name, model_path) as logger, CaptureStdStreams(logger):
            try:
                with open(model_path / 'model_desc.json', 'w') as f:
                    json.dump(model_desc_json_safe(model_desc), f, indent=2)

                with Timer() as timer:
                    run_result = tm.run_model(model_desc, sample_kwargs=model_desc.get('sample_kwargs'), save_path=str(model_path))
                    mod_df = run_result['draws']

                root_logger.info(f'Model {model_name} run in {timer.time:.3f} seconds')
                fit_time = timer.time

                with open(model_path / 'time.txt', 'w') as f:
                    f.write(f'{timer.time:.3f}')

                with open(model_path / 'model_id.txt', 'w') as f:
                    f.write(model_id)

                if not (model_path / 'heckman_coefs.json').exists():
                    shutil.copyfile(model_data['heckman_coefs.json'], model_path / 'heckman_coefs.json')
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                root_logger.error(f'Error running model {model_name}: {e}')
                traceback.print_exc(file=sys.stderr)
                return {
                    'model_path': str(model_path),
                    'model_id': model_id,
                    'model_name': model_desc['name'],
                }

    model_identifiers = {
        'model_path': str(model_path),
        'model_id': model_id,
        'model_name': model_desc['name'],
    }

    # Population
    pop_cols = ['age_group', 'education', 'gender', 'nationality', 'electoral_district', 'municipality', 'voting_intent']
    full_pop_df = pd.read_parquet(population_file)
    if 'N' in full_pop_df.columns:
        pop_cols.append('N')
    pop_df = full_pop_df[pop_cols]
    
    # Distances (includes BPV if draw-level is available)
    margin_cols = model_desc['input_cols']
    distances = get_distances(pop_df, mod_df, margin_cols, 'voting_intent')

    # Diagnostics from idata.nc

    
    idata = az.from_netcdf(model_path / 'idata.nc')
    from salk_turnout_models.models import mean_rhat as _mean_rhat

    mean_rhat = _mean_rhat(idata)
    divergences = idata.sample_stats.diverging.sum().item()

    model_fit = {
        'fit_time': fit_time,
        'mean_rhat': mean_rhat,
        'divergences': divergences,
    }

    posterior = idata.posterior

    coefs = get_coefs(model_path, posterior=posterior)
    ocoefs = coefs[coefs['process'] == 'outcome']

    rlm_stats = {}

    if len(ocoefs) > 0:
        # Fit robust regression using Huber's T norm
        X = ocoefs['real'].values
        y = ocoefs['model'].values
        rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
        rlm_fit = rlm_model.fit()
        y_pred = rlm_fit.predict(X)
        mae = np.mean(np.abs(y - y_pred))

        rlm_stats |= {
            'o_coef_rlm_slope': rlm_fit.params[0],
            'o_coef_rlm_mae': mae,
        }

    scoefs = coefs[coefs['process'] == 'selection']

    if len(scoefs) > 0:
        # Fit robust regression using Huber's T norm
        X = scoefs['real'].values
        y = scoefs['model'].values
        rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
        rlm_fit = rlm_model.fit()
        y_pred = rlm_fit.predict(X)
        mae = np.mean(np.abs(y - y_pred))

        rlm_stats |= {
            's_coef_rlm_slope': rlm_fit.params[0],
            's_coef_rlm_mae': mae,
        }

        rlm_stats['s_intercept_med'] = posterior['selection_intercept'].median(dim=['chain', 'draw']).item()

        if 'selection_nu' in posterior:
            rlm_stats['s_nu_med'] = posterior['selection_nu'].median(dim=['chain', 'draw']).item()
    
    if 'survey_0_or_bias' in posterior:
        rlm_stats['or_bias_med'] = posterior['survey_0_or_bias'].median(dim=['chain', 'draw']).item()

    model_coefs = {}
    if 'rho' in posterior:
        model_coefs['rho_med'] = posterior['rho'].median(dim=['chain', 'draw']).item()
        model_coefs['rho_sd'] = posterior['rho'].std(dim=['chain', 'draw']).item()


    # A few things for plotting

    plot_summary = {}

    # Total proportion selected
    if 'N' in full_pop_df.columns:
        plot_summary['selection_prop'] = (full_pop_df['selection'] * full_pop_df['N']).sum() / full_pop_df['N'].sum()
    else:
        plot_summary['selection_prop'] = full_pop_df['selection'].sum() / len(full_pop_df)

    # Overreport proportion in ground truth
    data_path = pathlib.Path(population_file).parent
    survey_df = pd.read_csv(data_path / 'estonia_selection.csv')
    if 'true_outcome' in survey_df.columns:
        overreport_diff = survey_df['outcome'].sum() - survey_df['true_outcome'].sum()
        plot_summary['overreport_prop'] = overreport_diff / len(survey_df)
    else:
        combined_df = pd.merge(survey_df, full_pop_df[['uid','outcome']], 
                                            on=['uid'], how='left', suffixes=('', '_pop'))
        overreport_diff = combined_df['outcome'].sum() - combined_df['outcome_pop'].sum()
        plot_summary['overreport_prop'] = overreport_diff / len(combined_df)

    # Persist aggregates for future runs
    summary_data = distances | rlm_stats | model_fit | model_coefs | plot_summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    # Shrink draws.parquet: per-chain means (drop draw)
    if 'draw' in mod_df.columns:
        agg_df = aggregate_draws_to_chain_means(mod_df)
        agg_df.to_parquet(model_path / 'draws.parquet', index=False)

    # Remove idata.nc after extracting summary stats
    try:
        os.unlink(model_path / 'idata.nc')
    except FileNotFoundError:
        pass

    print(str(model_path),rlm_stats)

    return model_identifiers | summary_data


# In[ ]:


MODEL_DESCRIPTION_TEMPLATE = {
    'outcome_col': 'voting_intent',
    'population': 'census_data.csv',
}

def replace_value(obj, orig_value, new_value):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = replace_value(v, orig_value, new_value)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = replace_value(v, orig_value, new_value)
    elif obj == orig_value:
        obj = new_value
    return obj

def get_model_description(name, config):
    model_desc = copy.deepcopy(MODEL_DESCRIPTION_TEMPLATE)
    model_desc['name'] = name
    model_desc['model_type'] = config['model_type']
    model_desc['input_cols'] = sorted(config.get('input_cols', ['age_group', 'gender', 'education', 'municipality', 'nationality']))

    if (interactions := config.get('interactions')) is not None:
        model_desc['interactions'] = interactions

    if model_desc['model_type'] in ['BP', 'GG', 'PM', 'FS', 'SE']:
        model_desc['survey'] = 'survey_data.csv'

    if model_desc['model_type'] in ['EI', 'GG', 'PM', 'FS']:
        margin_cols = sorted(config.get('margin_cols', ['municipality']))
        margin_file_names = [f'{margin_col}_margins_data.csv' for margin_col in margin_cols] if len(margin_cols) > 0 else ['margins_data.csv']
        model_desc['margin'] = margin_file_names

    if config.get('priors_scale_sigma') is not None or config.get('priors') is not None:
        model_desc['priors'] = dict(config.get('priors', {}))
        if config.get('priors_scale_sigma') is not None:
            sigma = config['priors_scale_sigma']
            model_desc['priors']['outcome_scale'] = lambda name, sigma=sigma, **kwargs: sigma
            model_desc['priors']['selection_scale'] = lambda name, sigma=sigma, **kwargs: sigma
            model_desc['priors']['outcome_interaction_scale'] = lambda name, sigma=sigma/2, **kwargs: sigma
            model_desc['priors']['selection_interaction_scale'] = lambda name, sigma=sigma/2, **kwargs: sigma

    if config.get('sample_kwargs') is not None:
        model_desc['sample_kwargs'] = config['sample_kwargs']

    if config.get('centered') is not None:
        model_desc['centered'] = config['centered']

    if config.get('margin_dist') is not None:
        model_desc['margin_dist'] = config['margin_dist']

    if config.get('robust_selection') is not None:
        model_desc['robust_selection'] = config['robust_selection']

    if config.get('selection_survey') is not None:
        model_desc['selection_survey'] = config['selection_survey']

    if config.get('selection_cols') is not None:
        model_desc['selection_cols'] = config['selection_cols']

    return model_desc

def get_model_data(data_name, margin_cols=[['municipality']], tmp_data_prefix='../tmp/data'):
    return {
        'census_data.csv': '../data/census.csv',
        'population.csv': f'{tmp_data_prefix}/{data_name}/population.parquet',
        'survey_data.csv': f'{tmp_data_prefix}/{data_name}/estonia_selection.csv',
        'margins_data.csv': f'{tmp_data_prefix}/{data_name}/estonia_margins.csv',
        'heckman_coefs.json': f'{tmp_data_prefix}/{data_name}/heckman_coefs.json',
    } | {
        f'{"_".join(sorted(cols))}_margins_data.csv': f'{tmp_data_prefix}/{data_name}/estonia_{"_".join(sorted(cols))}_margins.csv' for cols in margin_cols
    }
    
def create_symlink(target_path, link_path):
    if isinstance(link_path, str):
        link_path = pathlib.Path(link_path)

    if link_path.is_symlink():
        os.unlink(link_path)
    os.symlink(target_path, link_path)

def get_data_list(file_path, n_seeds_limit=1000):
    data_list = json.load(open(file_path, 'r'))
    seeds = {data_name: np.unique([data_config['seed'] for _, data_config in data_descs]) for data_name, data_descs in data_list.items()}
    return {data_name: [[data_id, data_config] for data_id, data_config in data_descs if data_config['seed'] in seeds[data_name][:n_seeds_limit]] for data_name, data_descs in data_list.items()}


# In[ ]:


tmp_data_prefix = '../tmp/data'
data_list = get_data_list(f'{tmp_data_prefix}/data_list.json', 11)

demography_cols = ['age_group', 'gender', 'education', 'municipality', 'nationality', 'electoral_district']
input_cols = [col for col in demography_cols if col != 'electoral_district']
all_margin_cols = [[col] for col in demography_cols] + [[c1, c2] for c1, c2 in it.combinations(demography_cols, 2)]


def make_common_models(icols):
    bp = {'model_type': 'BP', 'input_cols': icols, 'centered': False}
    ei = {'model_type': 'EI', 'input_cols': icols, 'centered': True}
    gg = {'model_type': 'GG', 'input_cols': icols, 'centered': False}
    pm = {'model_type': 'PM', 'input_cols': icols, 'centered': True}
    fs = {'model_type': 'FS', 'input_cols': icols, 'centered': True}
    return [
        get_model_description('1_bp', bp),
        get_model_description('2_ei', ei),
        get_model_description('3_gg', gg),
        get_model_description('4_pm', pm),
        get_model_description('5_fs', fs),
    ]


def make_fsr_models(icols):
    fsr = {'model_type': 'FS', 'input_cols': icols, 'centered': True, 'robust_selection': True}
    return [get_model_description('2_fsr', fsr)]


def make_int_models(icols):
    bp = {'model_type': 'BP', 'input_cols': icols, 'centered': False}
    ei = {'model_type': 'EI', 'input_cols': icols, 'centered': True}
    gg = {'model_type': 'GG', 'input_cols': icols, 'centered': False}
    pm = {'model_type': 'PM', 'input_cols': icols, 'centered': True}
    fs = {'model_type': 'FS', 'input_cols': icols, 'centered': True}
    return [
        get_model_description('1_int_bp', bp | {'interactions': True}),
        get_model_description('2_int_ei', ei | {'interactions': True}),
        get_model_description('3_int_gg', gg | {'interactions': True}),
        get_model_description('4_int_pm', pm | {'interactions': True}),
        get_model_description('5_int_fs', fs | {'interactions': True}),
    ]


bp_config = {'model_type': 'BP', 'input_cols': input_cols, 'centered': False}
ei_config = {'model_type': 'EI', 'input_cols': input_cols, 'centered': True}
gg_config = {'model_type': 'GG', 'input_cols': input_cols, 'centered': False}
pm_config = {'model_type': 'PM', 'input_cols': input_cols, 'centered': True}
fs_config = {'model_type': 'FS', 'input_cols': input_cols, 'centered': True}

common_models = make_common_models(input_cols)
fsr_models = make_fsr_models(input_cols)

def get_margin_model_name(margin_cols):
    return 'tpl' if len(margin_cols) == 0 else '_'.join(margin_cols)

# Margin informativeness figure (simulation-figures): tpl / municipality / electoral_district only.
margin_cols_list = [[], ['municipality'], ['electoral_district']]
ei_margin_models = [get_model_description(f'2_margin_{get_margin_model_name(margin_cols)}_ei', ei_config | {'input_cols': demography_cols, 'margin_cols': margin_cols}) for margin_cols in margin_cols_list]
gg_margin_models = [get_model_description(f'3_margin_{get_margin_model_name(margin_cols)}_gg', gg_config | {'input_cols': demography_cols, 'margin_cols': margin_cols}) for margin_cols in margin_cols_list]
pm_margin_models = [get_model_description(f'4_margin_{get_margin_model_name(margin_cols)}_pm', pm_config | {'input_cols': demography_cols, 'margin_cols': margin_cols}) for margin_cols in margin_cols_list]
fs_margin_models = [get_model_description(f'5_margin_{get_margin_model_name(margin_cols)}_fs', fs_config | {'input_cols': demography_cols, 'margin_cols': margin_cols}) for margin_cols in margin_cols_list]
margin_models = ei_margin_models + gg_margin_models + pm_margin_models + fs_margin_models

scale_models = [
    get_model_description('1_scale_bp', bp_config | {'priors_scale_sigma': 0.5}),
    get_model_description('2_scale_ei', ei_config | {'priors_scale_sigma': 0.5}),
    get_model_description('3_scale_gg', gg_config | {'priors_scale_sigma': 0.5}),
    get_model_description('4_scale_pm', pm_config | {'priors_scale_sigma': 0.5}),
    get_model_description('5_scale_fs', fs_config | {'priors_scale_sigma': 0.5}),
]

experiments = {
    'est-default': common_models + scale_models + make_int_models(input_cols),
    'est-non-response': common_models,
    'est-electoral-district': margin_models,
    'est-agg-bias': common_models,
    'est-no-selection': common_models,
    'est-hcoef-cor': common_models,
    'est-hcoef-sigma': common_models,
    'est-heck-cor': common_models,
    'est-sample-size': common_models,
    'est-overreport-const': common_models,
    'est-int': make_common_models(demography_cols) + make_int_models(demography_cols),
    'est-noise-out': common_models,
    'est-noise-sel': common_models, # + fsr_models,
    'est-non-normal-error': common_models# + fsr_models,
}

model_path_prefix = '../tmp/models'

def save_results(model_results):
    model_results_df = pd.concat([pd.DataFrame(data=result, index=[result['model_path']]) for result in model_results])

    ignored_cols = ['model_path']
    ordered_cols = ['model_id', 'data_name', 'data_id', 'desc', 'model_name']
    other_cols = [col for col in model_results_df.columns if col not in ordered_cols and col not in ignored_cols]

    model_results_df = model_results_df[ordered_cols + other_cols]
    model_results_path = pathlib.Path(f'../tmp/models/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_model_results.csv')
    model_results_df.to_csv(model_results_path)
    create_symlink(model_results_path.name, '../tmp/models/model_results.csv')
    return model_results_df


def _model_result_from_path(model_desc, model_id, data_name, data_desc, path_prefix=model_path_prefix):
    model_path = pathlib.Path(path_prefix) / model_id
    summary_path = model_path / 'summary.json'
    if not summary_path.exists():
        return None

    summary_data = json.load(open(summary_path, 'r'))
    result = {
        'model_path': str(model_path),
        'model_id': model_id,
        'model_name': model_desc['name'],
        'data_name': '-'.join(data_name.split('-')[:-1]),
        'data_id': data_name,
        'desc': ','.join([f'{k.split('/')[-1]}={v}' for k, v in data_desc.items()]),
    } | summary_data
    if 'target_accept' not in result:
        desc_path = model_path / 'model_desc.json'
        if desc_path.exists():
            saved = json.load(open(desc_path, 'r'))
            result['target_accept'] = saved.get('sample_kwargs', {}).get('target_accept', 0.9)
        else:
            result['target_accept'] = model_desc.get('sample_kwargs', {}).get('target_accept', 0.9)
    return result


def _model_desc_lookup_variants(model_desc):
    # Original runs were hashed without sample_kwargs; do not inject target_accept=0.9.
    variants = [copy.deepcopy(model_desc)]
    desc_99 = copy.deepcopy(model_desc)
    desc_99.setdefault('sample_kwargs', {})['target_accept'] = 0.99
    if get_model_id(desc_99) != get_model_id(variants[0]):
        variants.append(desc_99)
    return variants


def get_model_result_from_summary(model_desc, data_name, data_desc, path_prefix=model_path_prefix):
    # One row per grid cell: prefer the on-disk run with fewest divergences.
    candidates = []
    seen_ids = set()
    for desc in _model_desc_lookup_variants(model_desc):
        model_id = get_model_id(desc)
        if model_id in seen_ids:
            continue
        seen_ids.add(model_id)
        result = _model_result_from_path(desc, model_id, data_name, data_desc, path_prefix)
        if result is not None:
            candidates.append(result)
    if not candidates:
        return None
    return min(candidates, key=lambda r: r.get('divergences', 9999))


model_results = []
model_counts = {
    exp_name: len(experiments[exp_name]) * len(data_list.get(exp_name, []))
    for exp_name in experiments
}
print(model_counts)

total_models = sum(model_counts.values())
existing_model_results = []
existing_model_ids = set()
existing_grid_keys = set()

for experiment_name, model_descriptions in experiments.items():
    for data_name, data_desc in data_list.get(experiment_name, []):
        model_data = get_model_data(data_name, all_margin_cols)
        for desc in model_descriptions:
            model_desc = copy.deepcopy(desc)
            for key, value in model_data.items():
                model_desc = replace_value(model_desc, key, value)

            existing_model_result = get_model_result_from_summary(model_desc, data_name, data_desc)
            if existing_model_result is not None:
                existing_model_results.append(existing_model_result)
                existing_model_ids.add(existing_model_result['model_id'])
                existing_grid_keys.add((model_desc['name'], data_name))

print(f'{len(existing_model_results)} of {total_models} models already exist in {model_path_prefix}')
if existing_model_results:
    model_results = existing_model_results.copy()
    model_results_df = save_results(model_results)


# In[ ]:


progress = tqdm(total=total_models, initial=len(existing_model_results))

try:
    for experiment_name, model_descriptions in experiments.items():
        data_descs = data_list.get(experiment_name, [])
        if not data_descs:
            print(f'Skipping {experiment_name!r}: no datasets in data_list.json (generate data for this family first)')
            continue
        for data_name, data_desc in data_descs:
            model_data = get_model_data(data_name, all_margin_cols)

            for desc in model_descriptions:
                # Apply data configuration to the model description
                model_desc = copy.deepcopy(desc)

                for key, value in model_data.items():
                    model_desc = replace_value(model_desc, key, value)

                if (model_desc['name'], data_name) in existing_grid_keys:
                    continue

                print(model_desc)
                progress_postfix = {'experiment': experiment_name, 'dataset': data_name, 'description': data_desc}
                model_result = run_model(model_desc, model_data['population.csv'], model_data, path_prefix=model_path_prefix, progress=progress, progress_postfix=progress_postfix)
                model_result['data_name'] = '-'.join(data_name.split('-')[:-1])
                model_result['data_id'] = data_name
                model_result['desc'] = ','.join([f'{k.split('/')[-1]}={v}' for k, v in data_desc.items()])
                is_success = model_result.get('fit_time') is not None

                model_results.append(model_result)

                if is_success:
                    create_symlink(pathlib.Path(model_result['model_path']).name, pathlib.Path(model_path_prefix) / model_desc['name'])

                progress.update(1)

    progress.close()
finally:
    model_results_df = save_results(model_results)


# # Force diverging traces to re-calculate

# In[ ]:


import shutil
import pathlib

# Identify models with > 25 divergences
rerun_df = model_results_df[model_results_df['divergences'] > 25]
print(f"Found {len(rerun_df)} models with > 25 divergences to re-run.")


# In[ ]:


# Delete their output directories so they are forced to re-run
for _, row in rerun_df.iterrows():
    mpath = pathlib.Path(row.name)
    if mpath.exists() and (mpath / 'summary.json').exists():
        print(f"Removing {mpath} to force re-run...")
        shutil.rmtree(mpath)

# Re-populate existing_model_results and existing_grid_keys
existing_model_results = []
existing_model_ids = set()
existing_grid_keys = set()

for experiment_name, model_descriptions in experiments.items():
    for data_name, data_desc in data_list.get(experiment_name, []):
        model_data = get_model_data(data_name, all_margin_cols)
        for desc in model_descriptions:
            model_desc = copy.deepcopy(desc)
            for key, value in model_data.items():
                model_desc = replace_value(model_desc, key, value)

            existing_model_result = get_model_result_from_summary(model_desc, data_name, data_desc)
            if existing_model_result is not None:
                existing_model_results.append(existing_model_result)
                existing_model_ids.add(existing_model_result['model_id'])
                existing_grid_keys.add((model_desc['name'], data_name))

print(f'{len(existing_model_results)} of {total_models} models already exist in {model_path_prefix}')

model_results = existing_model_results.copy()
progress = tqdm(total=total_models, initial=len(existing_model_results))

try:
    for experiment_name, model_descriptions in experiments.items():
        data_descs = data_list.get(experiment_name, [])
        if not data_descs:
            continue
        for data_name, data_desc in data_descs:
            model_data = get_model_data(data_name, all_margin_cols)

            for desc in model_descriptions:
                model_desc = copy.deepcopy(desc)

                for key, value in model_data.items():
                    model_desc = replace_value(model_desc, key, value)

                if (model_desc['name'], data_name) in existing_grid_keys:
                    continue

                print(f"Re-running model: {model_desc['name']}")
                progress_postfix = {'experiment': experiment_name, 'dataset': data_name, 'description': data_desc}
                
                # Increase target_accept for the re-run to help with divergences
                if 'sample_kwargs' not in model_desc:
                    model_desc['sample_kwargs'] = {}
                model_desc['sample_kwargs']['target_accept'] = 0.99
                
                model_result = run_model(model_desc, model_data['population.csv'], model_data, path_prefix=model_path_prefix, progress=progress, progress_postfix=progress_postfix)
                model_result['data_name'] = '-'.join(data_name.split('-')[:-1])
                model_result['data_id'] = data_name
                model_result['desc'] = ','.join([f'{k.split("/")[-1]}={v}' for k, v in data_desc.items()])
                model_result['target_accept'] = model_desc.get('sample_kwargs', {}).get('target_accept', 0.9)
                is_success = model_result.get('fit_time') is not None

                model_results.append(model_result)

                if is_success:
                    create_symlink(pathlib.Path(model_result['model_path']).name, pathlib.Path(model_path_prefix) / model_desc['name'])

                progress.update(1)

    progress.close()
finally:
    model_results_df = save_results(model_results)


# # Debug chosen traces

# In[ ]:


data_name = 'est-non-response-dd287e6ce09771fd326a48a3a53bf962'
model_desc = get_model_description('one_off', {
    'model_type': 'FS',
    'input_cols': input_cols,
    'priors': {'overreporting': lambda name, **kwargs: 0.0},
})
model_data = get_model_data(data_name, all_margin_cols)
for key, value in model_data.items():
    model_desc = replace_value(model_desc, key, value)
model_result = run_model(model_desc, model_data['population.csv'], model_data, path_prefix=model_path_prefix)
print(model_result)


# In[ ]:


def plot_scatter(model_path, process='selection'):
    model_path = pathlib.Path(model_path)
    idata = az.from_netcdf(model_path / 'idata.nc')
    coefs = get_coefs(model_path, posterior=idata.posterior)

    df = coefs[coefs['process'] == process]
    import altair as alt

    # Scatter plot of Real vs Model with 1:1 abline
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X('real:Q', title='Real Value'),
        y=alt.Y('model:Q', title='Model Value'),
        color=alt.Color('var:N', title='Variable'),
        tooltip=[
            alt.Tooltip('var:N', title='Variable'),
            alt.Tooltip('index:N', title='Index'),
            alt.Tooltip('real:Q', title='Real Value'),
            alt.Tooltip('model:Q', title='Model Value')
        ]
    ).properties(
        width=400,
        height=350,
        title=f"Scatter plot of Real vs Model ({process})"
    )

    # 1:1 abline
    min_val = float(df[['real', 'model']].min().min())
    max_val = float(df[['real', 'model']].max().max())
    abline = alt.Chart(
        pd.DataFrame({'real': [min_val, max_val], 'model': [min_val, max_val]})
    ).mark_line(
        color='black', strokeDash=[4,4]
    ).encode(
        x='real:Q',
        y='model:Q'
    )

    chart = chart + abline

    return chart

#plot_scatter('../tmp/models/db4b8b39595b1a0a91f38322ee5e8322')#,process='outcome')

for r in model_results_df.itertuples():
    if r.model_name != '4_fsp': continue
    print(r.model_name, r.model_id)
    plot_scatter(r.Index).display()


# In[ ]:





# In[ ]:




