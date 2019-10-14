import pandas as pd
import numpy as np
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import warnings
from tqdm import tqdm

def save_simulated(sim_df, save_dir='Tasks/uncertainty_attention_task_shock/data/simulated_data'):

    """
    Take simulated data based on best-fitting values for each subject and turn it into a dataframe per subject

    Args:
        sim_df: Simulated data in dataframe format (i.e. model.sim_df)

    """

    sim_df = sim_df[[c for c in sim_df.columns if 'sim' not in c and 'choices' not in c and c != 'o']]

    A_df = sim_df[sim_df.Run % 2 == 0]
    B_df = sim_df[sim_df.Run % 2 == 1]

    A_df.columns = ['A_' + c if c != 'Subject' else c for c in A_df.columns]
    B_df.columns = ['B_' + c if c != 'Subject' else c for c in B_df.columns]
    A_df['trial_number'] = np.tile(np.arange(len(A_df) / len(A_df.Subject.unique())),len(A_df.Subject.unique()))
    B_df['trial_number'] = np.tile(np.arange(len(B_df) / len(B_df.Subject.unique())),len(B_df.Subject.unique()))

    sim_df_twostim = pd.merge(A_df, B_df, on=['Subject', 'trial_number'])

    for sub in tqdm(sim_df_twostim.Subject.unique()):
        fname = os.path.join(save_dir, "{0}_simulated_data.txt".format(sub))
        sim_df_twostim[sim_df_twostim.Subject == sub].to_csv(fname, index=False)


def combine_behavioural(data_dir, file_regex='uncertainty_attention_shock.+behaviour.csv', subject_regex='(?<=Subject_)[0-9]+',
                        out_file='uncertainty_attention_shock_combined.csv', trials_per_block=40, total_trials=160,
                        break_number=999):

    """
    Function to convert individual subjects' behaviour files into a combined file for modelling

    Args:
        data_dir: Directory containing behavioural data
        file_regex: Regex for identifying behavioural files
        subject_regex: Regex for extracting subject ID from filenames
        out_file: Filename for output file
        trials_per_block: Number of trials per block, default = 40
        total_trials: Expected total number of trials, default = 160
        break_number: Trial number used to indicate breaks, default = 999

    Returns:
        File name of saved file

    """


    data_files = [f for f in os.listdir(data_dir) if re.match(file_regex, f)]
    data_files = sorted(data_files)  # Read in in order

    subject_dfs = []

    shock_levels = []

    exclusions = []

    for n, f in enumerate(tqdm(data_files)):

        try:

            data = pd.read_csv(os.path.join(data_dir, f))
            if data.shock_level.dtype == 'object':
                data.shock_level = data.shock_level.str.replace('_', '').astype(int)
            shock_level = data.shock_level.mean()
            shock_levels.append(shock_level)

            column_names = ['Response', 'Outcome', 'Outcome_2', 'Objective_prob_A', 'Objective_prob_B']
            data_A = data[['A_prob_estimate', 'A_shock', 'B_shock', 'A_shock_prob', 'B_shock_prob']]
            data_B = data[['B_prob_estimate', 'B_shock', 'A_shock', 'B_shock_prob', 'A_shock_prob']]
            data_A.columns = column_names
            data_B.columns = column_names

            break_index = np.where(data.trial_number == break_number)
            break_index = [0] + list(break_index[0]) + [len(data)]

            dfs = []

            for n, i in enumerate(break_index[:-1]):
                temp_df_a = data_A[i:break_index[n + 1]]
                temp_df_a = temp_df_a.assign(Run=pd.Series(n * 2).values.repeat(len(temp_df_a)))
                temp_df_b = data_B[i:break_index[n + 1]]
                temp_df_b = temp_df_b.assign(Run=pd.Series(n * 2 + 1).values.repeat(len(temp_df_b)))
                dfs += [temp_df_a.dropna()[:trials_per_block], temp_df_b.dropna()[:trials_per_block]]

            dfs = pd.concat(dfs)


            try:
                dfs['Subject'] = re.search(subject_regex, f).group()
            except:
                raise ValueError("Could not find subject ID in file name, file = {0}".format(f))

            dfs = dfs.dropna()  # remove NAs, used to remove break trials
            if len(dfs) < total_trials:
                exclusions.append(re.search(subject_regex, f).group())

            prop_extreme_ratings = sum((dfs.Response > 0.95) | (dfs.Response < 0.05) |
                                ((dfs.Response > 0.45) & (dfs.Response < 0.55))) / float(len(dfs))

            if prop_extreme_ratings > 0.8:
                exclusions.append(re.search(subject_regex, f).group())

            else:
                subject_dfs.append(dfs)

        except Exception as e:
            print "Failed on file {0}".format(f)
            print e

    subject_dfs = pd.concat(subject_dfs)

    fname = os.path.join(data_dir, out_file)
    subject_dfs.to_csv(fname, index=False)

    print "Excluded subjects: {0}".format(exclusions)

    print "Saved combined data to {0}".format(fname)

    print "Mean shock level = {0}".format(np.mean(shock_levels))
    #print "Shock level standard deviation = {0}".format(np.std(shock_levels))

    return fname, subject_dfs


def plot_param_values(param_values, out_file='analysis/plots/beta_individual_params'):

    """
    Function to plot parameter estimates with uncertainty

    Args:
        param_values: DMpy parameter table
        out_file: File to save plot to

    """

    param_values_mean = param_values[[i for i in param_values.columns if 'mean' in i or 'Subject' in i]]
    param_values_mean.columns = [i.replace('mean_', '') for i in param_values_mean.columns]
    param_values_sd = param_values[[i for i in param_values.columns if 'sd' in i or 'Subject' in i]]
    param_values_sd.columns = [i.replace('sd_', '') for i in param_values_sd.columns]

    param_values_long = pd.melt(param_values_mean, id_vars=['Subject'], value_name='mean')
    param_values_sd_long = pd.melt(param_values_sd, id_vars=['Subject'])
    param_values_long['sd'] = param_values_sd_long.value
    param_values_long.columns = ['Subject', 'Parameter', 'Mean', 'SD']
    param_values_long['var_name'] = param_values_long.Subject.astype(str).str.cat(param_values_long.Parameter, sep=' ')

    param_values_long = param_values_long.sort_values(["Parameter", 'Subject'])
    param_values_long['test'] = range(0, len(param_values_long))

    pal = sns.color_palette("hls", 7)
    pal_extended = []
    for i in pal:
        pal_extended += [i] * len(param_values_long.Subject.unique())

    sns.set_style("white")
    plt.figure(figsize=[6, 8])

    plt.scatter(param_values_long.Mean, param_values_long['test'][::-1], facecolors=pal_extended, label=None)
    plt.errorbar(param_values_long.Mean, param_values_long['test'][::-1], xerr=param_values_long.SD, fmt='none',
                 ecolor=pal_extended,
                 c=pal_extended[::-1], label=None)
    plt.yticks(range(0, len(param_values_long))[::-1], param_values_long.Subject)

    for n, i in enumerate(param_values_long.Parameter.unique()):
        plt.scatter([], [], alpha=0.8, s=40,
                    label=str(i), c=pal[n])
    plt.legend(scatterpoints=1, frameon=True, title='Parameter')

    plt.xlabel("Estimated value")
    plt.ylabel("Subject")
    plt.tight_layout()

    plt.savefig(out_file + '.svg')
    plt.savefig(out_file + '.pdf')


def construct_equation(predictors, outcome):

    """
    Constructs a regression equation for bambi
    Args:
        predictors: predictor variables, list of strings
        outcome: outcome variable, single string

    Returns:
        Equation of the form y ~ x1 + x2...
    """

    eq_string = '{0} ~ '.format(outcome)

    if isinstance(predictors, dict):
        predictors = predictors.values()

    for n, i in enumerate(predictors):
        if n == 0:
            eq_string += '{0}'.format(i)
        else:
            eq_string += ' + {0}'.format(i)

    return eq_string