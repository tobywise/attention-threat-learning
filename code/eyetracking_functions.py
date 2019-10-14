import pyeparse as pp
import numpy as np
import pandas as pd
import re
import os
from sklearn.preprocessing import scale

def pix2deg(pixel_x, pixel_y, display_size_mm=(409.2, 255.8), display_res_pix=(1024.0, 768.0), eye_distance_mm=740.0):

    """
    Converts pixels to degrees
    """

    display_width = display_size_mm[0]
    display_height = display_size_mm[1]
    display_x_resolution = display_res_pix[0]
    display_y_resolution = display_res_pix[1]
    eye_distance_mm = eye_distance_mm

    sx, sy = pixel_x, pixel_y

    thetaH1 = np.degrees(np.arctan(sx / (eye_distance_mm *
                                         display_x_resolution / display_width)))
    thetaV1 = np.degrees(np.arctan(sy / (eye_distance_mm *
                                         display_y_resolution / display_height)))

    return thetaH1, thetaV1

def get_aoi(x, y, left_aoi, right_aoi):

    """
    Returns 1 if the fixation point is in the right AOI, -1 if in the left
    """

    x, y = pix2deg(x, y)

    if left_aoi.contains_point((x, y)):
        return -1
    elif right_aoi.contains_point((x, y)):
        return 1
    else:
        return np.nan


def process_fixations(sub, n, epoch_epoch_start, epoch_epoch_end, eyetracking_dir,
    behavioural_dir, simulated_dir, left_aoi, right_aoi, phase):

    """
    Produces a dataframe of fixation bias indices for each trial and each subject

    Args:
        epoch_epoch_start: Start of the epoch (post trial phase onset)
        epoch_epoch_end: End of the epoch
        eyetracking_dir: Directory containing eyetracking data
        behavioural_data: Directory containing behavioural data for all subjects
        simulated_data: Directory containing simulated data for all subjects
        left_aoi: matplotlib path containing left aoi
        right_aoi: matplotlib path containing right aoi

    Returns:
        A dataframe

    """

    # Load eyetracking data
    et_path = os.path.join(eyetracking_dir,
                           [i for i in os.listdir(eyetracking_dir) if str(sub) in i and 'eyetracker' in i][0])

    raw = pp.read_raw(et_path)

    # Get trials
    preoutcome_events = raw.find_events('_preoutcome', 1)
    outcome_events = raw.find_events('_outcome', 1)

    for n, i in enumerate(raw.discrete['messages']):
        m = re.search('Trial_[0-9]+$', i[1])
        if m:
            raw.discrete['messages'][n] = (i[0], i[1] + '_start')
    trial_start_events = raw.find_events('_start', 1)


    # Trial numbers
    trial_numbers = [int(re.search('\d+', i[1]).group()) for i in raw.discrete['messages'] if phase in i[1]]
    trial_start_events = trial_start_events[trial_numbers]  # Remove non-existent trials and shift in time

    tmin, tmax, event_id = epoch_epoch_start, epoch_epoch_end, 1
    preoutcome_epochs = pp.Epochs(raw, events=preoutcome_events, event_id=event_id, tmin=tmin, tmax=tmax)

    if len(trial_numbers) > len(preoutcome_epochs):
        trial_numbers = trial_numbers[:len(preoutcome_epochs)]

    preoutcome_fixations = {}

    preoutcome_duration_total = 0
    blink_duration_total = 0


    # The preoutcome phase doesn't have a standard duration so we need to customise this for each trial
    for n in range(len(preoutcome_epochs)):
        if 'preoutcome' in phase:
            preoutcome_duration = raw.times[outcome_events[n][0]] - raw.times[preoutcome_events[n][0]]
        else:
            preoutcome_duration = raw.times[trial_start_events[n][0]] - raw.times[outcome_events[n][0]]
            if preoutcome_duration > 7:
                preoutcome_duration = 4  # durations before breaks can be a longer than they should be so set to minimum expected
        preoutcome_fixations[trial_numbers[n]] = preoutcome_epochs[n].fixations[0][
            preoutcome_epochs[n].fixations[0]['stime'] < preoutcome_duration]
        preoutcome_fixations[trial_numbers[n]]['etime'][
            preoutcome_fixations[trial_numbers[n]]['etime'] > preoutcome_duration] = preoutcome_duration

        preoutcome_blinks = preoutcome_epochs[n].blinks[0][
            preoutcome_epochs[n].blinks[0]['stime'] < preoutcome_duration]
        preoutcome_blinks['etime'][preoutcome_blinks['etime'] > preoutcome_duration] = preoutcome_duration
        blink_duration_total += np.sum(np.array([blink[1] - blink[0] for blink in preoutcome_blinks]))
        preoutcome_duration_total += preoutcome_duration

        if np.any(preoutcome_fixations[trial_numbers[n]]['etime'] - preoutcome_fixations[trial_numbers[n]]['stime'] < 0):
            print preoutcome_fixations[trial_numbers[n]]
            raise ValueError("Fixations of less than 0 seconds on trial {0}".format(n))


    # Load behavioural
    behavioural_path = [i for i in os.listdir(behavioural_dir) if str(sub) in i and 'behaviour' in i][0]
    behavioural = pd.read_csv(os.path.join(behavioural_dir, behavioural_path))
    behavioural = behavioural[(~behavioural.A_shock_prob.isnull()) & (behavioural.trial_number != 999)].reset_index()

    # Load behavioural & simulated data
    simulated = pd.read_csv(os.path.join(simulated_dir, '{0}_simulated_data.txt'.format(sub)))
    simulated.loc[:, 'A_pe'] = simulated.loc[:, 'A_Outcome'] - simulated.loc[:, 'A_True_response']
    simulated.loc[:, 'B_pe'] = simulated.loc[:, 'B_Outcome'] - simulated.loc[:, 'B_True_response']
    simulated.loc[:, 'abs_pe_RL_diff'] = np.abs(simulated.A_pe) - np.abs(simulated.B_pe)
    simulated.loc[:, 'abs_pe_RL_diff'] = np.roll(simulated.loc[:, 'abs_pe_RL_diff'], 1)
    simulated.loc[:, 'prob_estimate_RL_diff'] = simulated.A_True_response - simulated.B_True_response
    simulated.loc[:, 'model_prob_estimate_RL_diff'] = simulated.A_Response - simulated.B_Response
    simulated.loc[:, 'var_RL_diff'] = simulated.A_var - simulated.B_var
    simulated.loc[:, 'objective_prob_RL_diff'] = behavioural.A_shock_prob - behavioural.B_shock_prob

    # trial number correction
    simulated.trial_number += 1

    # Get bias
    bias = []
    # Fixation durations (total for each trial)
    l_durations = []
    r_durations = []
    # First fixation durations for each stimulus
    l_first_durations = []
    r_first_durations = []
    outside_durations = []
    # First fixation location
    first_fix_locations = []

    l_duration_total = 0
    r_duration_total = 0
    outside_duration_total = 0

    for e in range(1, 161):
        if e not in trial_numbers:
            bias.append(np.nan)
            l_durations.append(np.nan)
            r_durations.append(np.nan)
            l_first_durations.append(np.nan)
            r_first_durations.append(np.nan)
            outside_durations.append(np.nan)
            first_fix_locations.append(np.nan)
        else:
            fix_locs = np.array([get_aoi(fix[2], fix[3], left_aoi, right_aoi) for fix in preoutcome_fixations[e]])
            fix_duration = np.array([fix[1] - fix[0] for fix in preoutcome_fixations[e]])

            l_duration = (fix_duration[fix_locs < 0]).sum()
            l_duration_total += l_duration
            r_duration = (fix_duration[fix_locs > 0]).sum()
            r_duration_total += r_duration

            stim_fix_locs = [i for i in fix_locs if not np.isnan(i)]
            if not len(stim_fix_locs):
                first_fix_locations.append(np.nan)
            else:
                first_fix_locations.append(stim_fix_locs[0])

            # First fixations
            if len(fix_duration[fix_locs < 0]):
                l_first_duration = fix_duration[fix_locs < 0][0]
            else:
                l_first_duration = 0

            if len(fix_duration[fix_locs > 0]):
                r_first_duration = fix_duration[fix_locs > 0][0]
            else:
                r_first_duration = 0

            outside_duration = (fix_duration[np.isnan(fix_locs)]).sum()
            outside_duration_total += outside_duration
            if l_duration > 0 or r_duration > 0:
                b = l_duration / (l_duration + r_duration)
            else:
                b = np.nan
            bias.append(b)
            l_durations.append(l_duration)
            r_durations.append(r_duration)
            l_first_durations.append(l_first_duration)
            r_first_durations.append(r_first_duration)
            outside_durations.append(outside_duration)


    bias = np.array(bias)
    bias[np.isnan(bias)] = np.nanmean(bias)  # Impute nans

    bias_df = behavioural[['Subject', 'trial_number', 'Outcome_image_L']].copy()
    bias_df.loc[:, 'bias'] = bias
    bias_df.loc[:, 'l_duration'] = l_durations
    bias_df.loc[:, 'r_duration'] = r_durations
    bias_df.loc[:, 'l_first_duration'] = l_first_durations
    bias_df.loc[:, 'r_first_duration'] = r_first_durations
    bias_df.loc[:, 'first_fixation_location'] = first_fix_locations
    bias_df.loc[:, 'outside_duration'] = outside_durations
    bias_df.loc[:, 'l_prop'] = bias_df.loc[:, 'l_duration'] / (bias_df.loc[:, 'l_duration'] +
                                                               bias_df.loc[:, 'r_duration'] +
                                                               bias_df.loc[:, 'outside_duration'])
    bias_df.loc[:, 'r_prop'] = bias_df.loc[:, 'r_duration'] / (bias_df.loc[:, 'l_duration'] +
                                                               bias_df.loc[:, 'r_duration'] +
                                                               bias_df.loc[:, 'outside_duration'])
    bias_df.loc[:, 'l_prop'] = np.nanmean(bias_df.loc[:, 'l_prop'])
    bias_df.loc[:, 'r_prop'] = np.nanmean(bias_df.loc[:, 'r_prop'])

    bias_df.loc[:, 'preoutcome_duration_total'] = preoutcome_duration_total
    bias_df.loc[:, 'blink_duration_total'] = blink_duration_total
    bias_df.loc[:, 'blink_proportion'] = blink_duration_total / preoutcome_duration_total
    bias_df.loc[:, 'l_duration_total'] = l_duration_total
    bias_df.loc[:, 'r_duration_total'] = r_duration_total
    bias_df.loc[:, 'outside_duration_total'] = outside_duration_total
    bias_df.loc[:, 'left_proportion'] = l_duration_total / (r_duration_total + l_duration_total)
    bias_df.loc[:, 'right_proportion'] = r_duration_total / (r_duration_total + l_duration_total)
    bias_df.loc[:, 'outside_proportion'] = outside_duration_total / (
                r_duration_total + l_duration_total + outside_duration_total)

    bias_df = pd.merge(bias_df,
                       simulated[['trial_number', 'prob_estimate_RL_diff', 'var_RL_diff', 'objective_prob_RL_diff',
                                  'A_var', 'A_True_response', 'B_var', 'B_True_response', 'A_Outcome', 'B_Outcome',
                                  'abs_pe_RL_diff', 'A_pe', 'B_pe', 'model_prob_estimate_RL_diff']],
                       on='trial_number')

    bias_df2 = bias_df[[c for c in bias_df.columns if 'B_' not in c]].copy()
    bias_df3 = bias_df[[c for c in bias_df.columns if 'A_' not in c]].copy()
    bias_df2.loc[bias_df2.Outcome_image_L == 'B', 'l_prop'] = bias_df2['r_prop']
    bias_df3.loc[bias_df2.Outcome_image_L == 'A', 'l_prop'] = bias_df3['r_prop']        

    bias_df2['stimulus'] = 0
    bias_df3['stimulus'] = 1

    bias_df2.columns = [c.replace('A_', '') for c in bias_df2.columns]
    bias_df3.columns = [c.replace('B_', '') for c in bias_df3.columns]

    duration_df = pd.concat([bias_df2, bias_df3])

    return bias_df, duration_df

