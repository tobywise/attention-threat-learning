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

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("model_instance")
    args = parser.parse_args()

    with open(args.model_instance, 'rb') as f:
        model = dill.load(f)

    exclusions = pd.read_csv('data/exclusions.txt')['Subject'].tolist()
    combined_data = 'data/behavioural_data/uncertainty_attention_shock_combined.csv'

    if 'fixation' in model.name:
        if 'nofixation' in model.name:
            inputs = ['Outcome_2']
            combined_data = 'notebooks/et_behav_data_full.csv'
        elif not 'duration' in model.name:
            inputs = ['Outcome_2', 'fixation_proportion_A', 'fixation_proportion_B']
            combined_data = 'notebooks/et_behav_data_full.csv'
        else:
            inputs = ['Outcome_2', 'fixation_duration_A', 'fixation_duration_B']
            combined_data = 'notebooks/et_behav_data_full.csv'
    else:
        inputs = ['Outcome_2']
        combined_data = 'data/behavioural_data/uncertainty_attention_shock_combined.csv'

    print("FITTING {0}".format(model.name))
    model.fit(combined_data, fit_method='mcmc', model_inputs=inputs, hierarchical=True,
      sample_kwargs={'draws': 3000, 'tune': 1000, 'njobs':1}, exclude_subjects=exclusions, fit_stats=True, plot=False,
      response_transform=beta_response_transform, suppress_table=True)

#     m.fit(combined_data, fit_method='variational', model_inputs=['Outcome_2'], hierarchical=True,
#           fit_kwargs={'n': 40000}, exclude_subjects=exclusions, fit_stats=True, plot=False, 
#           response_transform=beta_response_transform, suppress_table=True)

    with open(args.model_instance + '_fit.pklz', 'wb') as f:
        dill.dump(model, f)

    print("MODEL FITTING FINISHED")
