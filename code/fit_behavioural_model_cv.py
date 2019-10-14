import sys
sys.path.insert(0, '../code')
import dill
import re
import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T
import json
import os
from DMpy import DMModel, Parameter
from DMpy.utils import beta_response_transform
from DMpy.observation import softmax
import pandas as pd
import copy
from learning_models import *
import pymc3 as pm
from sklearn.metrics import r2_score

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("model_instance")
    parser.add_argument("block")  # block to exclude
    args = parser.parse_args()

    with open(args.model_instance, 'rb') as f:
        model = dill.load(f)

    exclusions = pd.read_csv('data/exclusions.txt')['Subject'].tolist()
    combined_data = pd.read_csv('data/behavioural_data/uncertainty_attention_shock_combined.csv')
    combined_data = combined_data[~combined_data['Subject'].astype(int).isin(exclusions)]

    block = int(args.block)
    excluded_runs = [block * 2, block * 2 + 1]

    print("FITTING {0}, fold {1}".format(model.name, block+1))
    model.fit(copy.deepcopy(combined_data), fit_method='mcmc', model_inputs=['Outcome_2'], hierarchical=True,
    sample_kwargs={'draws': 3000, 'tune': 1000, 'njobs':1}, exclude_subjects=exclusions, fit_stats=True, plot=False,
    response_transform=beta_response_transform, suppress_table=True, exclude_runs=excluded_runs)

    print("FITTING {0}, fold {1}".format(model.name, block+1))
    temp_df = combined_data[combined_data.Run.isin(excluded_runs)]

    sim, _ = model.simulate(outcomes=temp_df, response_variable='value', model_inputs=['Outcome_2'])

    sim.results.to_csv(args.model_instance + '_cv_block_{0}_simulated.csv'.format(block), index=False)

    print("MODEL FITTING FINISHED")
