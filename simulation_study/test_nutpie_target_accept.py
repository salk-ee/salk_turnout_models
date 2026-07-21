import pymc as pm
import numpy as np

with pm.Model() as model:
    x = pm.Normal('x', mu=0, sigma=1)
    try:
        idata = pm.sample(draws=10, tune=10, chains=1, target_accept=0.99, nuts_sampler='nutpie')
        print("Success with nutpie")
    except Exception as e:
        print(f"Error with nutpie: {e}")
