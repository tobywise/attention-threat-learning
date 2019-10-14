import theano.tensor as T
import numpy as np
import theano


def dual_lr_qlearning(o, t, o2, v, alpha_p, alpha_n, omega):

    """
    Dual learning rate Q-learning model, from Palminteri et al., 2017, Nature Human Behaviour

    Args:
        o: Trial outcome
        v: Value on previous trial
        alpha_p: Learning rate for positive prediction errors
        alpha_n: Learning rate for negative prediction errors

    Returns:
        value: Value on current trial
        pe: prediction error
        weighted_pe: prediction error weighted by learning rate
    """

    pe = o - v

    weighted_pe = T.switch(T.lt(pe, 0), alpha_n * pe, alpha_p * pe)

    value = v + weighted_pe

    o2 = T.switch(T.lt(omega, 0), 1 - o2, o2)  # If omega is negative, we invert the outcome and use the absolute value of omega
    pe_2 = o2 - value
    value = value + T.abs_(omega) * pe_2

    return (value, pe, weighted_pe)


def pearce_hall_1(o, t, o2, v, alpha, k, omega):

    """
    Dynamic learning rate model where learning rate varies depending on magnitude of squared prediction errors

    Args:
        o: Outcome
        t: Trial
        v: Value from previous trial
        alpha: Previous learning rate
        beta: Free parameter governing the degree to which the learning date changes on each trial

    Returns:

    """

    pe = o - v
    value = v + alpha * pe

    o2 = T.switch(T.lt(omega, 0), 1 - o2, o2)
    pe_2 = o2 - value
    value = value + T.abs_(omega) * pe_2

    alpha_m = alpha + k * (T.pow(pe, 2) - alpha)

    return (value, alpha_m, pe)


def pearce_hall_2(o, t, o2, v, alpha, k, omega):

    """
    Hybrid Rescorla-Wagner / Pearce-Hall model from Tzovara et al., (2018, PloS Computational Biology)

    Args:
        o: Outcome
        t: Trial
        v: Value from previous trial
        alpha: Previous learning rate
        k: Free parameter governing the degree to which the learning date changes on each trial

    Returns:

    """

    pe = o - v
    value = v + alpha * pe

    o2 = T.switch(T.lt(omega, 0), 1 - o2, o2)
    pe_2 = o2 - value
    value = value + T.abs_(omega) * pe_2

    alpha_m = k * T.abs_(pe) + (1 - k) * alpha

    return (value, alpha_m, pe)


def rescorla_wagner(o, t, o2, v, alpha, omega):
    """
    o: outcome
    v: prior value
    alpha: learning rate
    """
    pe = o - v
    value = v + alpha * pe

    return (value, pe)


def leaky_beta(o, t, o2, v, alpha, beta, d, omega, tau):

    """
    Forgetful beta model

    Args:
        o: Trial outcome
        t: Time (not used)
        o2: Outcome of the other stimulus
        v: Previous trial value estimate (not used)
        alpha: Starting alpha
        beta: Starting beta
        d: Decay (forgetting) rate
        omega: Weight of the other stimulus outcome
        tau: Update weight

    Returns:
        Mean: Estimated probability on the current trial (mean of beta distribution)
        Alpha: Alpha value on current trial
        Beta: Beta value on current trial
        Var: Variance of beta distribution

    """

    alpha = (1 - d) * alpha + (o * tau) + (omega * o2)
    beta = (1 - d) * beta + ((1 -o) * tau) + (omega * (1 - o2))

    alpha = T.maximum(T.power(0.1, 10), alpha)
    beta = T.maximum(T.power(0.1, 10), beta)

    value = alpha / (alpha + beta)

    var = (alpha * beta) / (T.pow(alpha + beta, 2) * (alpha + beta + 1))

    return (value, alpha, beta, var)


def leaky_beta_asymmetric(o, t, o2, v, alpha, beta, d, omega, tau_p, tau_n):

    """
    Forgetful beta model with asymmetric updating

    Args:
        o: Trial outcome
        t: Time (not used)
        o2: Outcome of the other stimulus
        v: Previous trial value estimate (not used)
        alpha: Starting alpha
        beta: Starting beta
        d: Decay (forgetting) rate
        omega: Weight of the other stimulus outcome
        tau_p: Positive update weight
        tau_n: Negative update weight

    Returns:
        Mean: Estimated probability on the current trial (mean of beta distribution)
        Alpha: Alpha value on current trial
        Beta: Beta value on current trial
        Var: Variance of beta distribution

    """


    alpha = (1 - d) * alpha + (o * tau_p) + (omega * o2)
    beta = (1 - d) * beta + ((1 - o) * tau_n) + (omega * (1 - o2))

    alpha = T.maximum(T.power(0.1, 10), alpha)
    beta = T.maximum(T.power(0.1, 10), beta)

    value = alpha / (alpha + beta)

    var = (alpha * beta) / (T.pow(alpha + beta, 2) * (alpha + beta + 1))

    return (value, alpha, beta, var)


def leaky_beta_asymmetric_fixation_1(o, t, o2, f1, f2, v, alpha, beta, d, omega, tau_p, tau_n, gamma):

    """
    
    Forgetful beta model with asymmetric updating and fixation weighting of learning
    --------------------------------------------------------------------------------
    
    Fixation weighting is calculated such that the most attended stimulus is multiplied by 1
    and the least weighted is multiplied by 1 minus the difference between the fixation durations
    for stimulus 1 and 2, which is itself weighted by the gamma parameter. 
    
    E.g. If stimulus 1 is fixated 70% of the time and stimulus 2 is fixated 30% of the time, stimulus 1 updates 
    will be weighted by 1 and stimulus 2 by 0.6 if gamma is set to 1. If gamma were 0.5, the update weights
    would be 1 and 0.8.

    Args:
        o: Trial outcome
        t: Time (not used)
        o2: Outcome of the other stimulus
        f1: Fixation duration proportion for this stimulus
        f2: Fixation duration proportion for the other stimulus
        v: Previous trial value estimate (not used)
        alpha: Starting alpha
        beta: Starting beta
        d: Decay (forgetting) rate
        omega: Weight of the other stimulus outcome
        tau_p: Positive update weight
        tau_n: Negative update weight
        gamma: Influence of fixation weighting on updates

    Returns:
        Mean: Estimated probability on the current trial (mean of beta distribution)
        Alpha: Alpha value on current trial
        Beta: Beta value on current trial
        Var: Variance of beta distribution

    """
    
    f_weight = (f1 * gamma) + (1-gamma)

    alpha = (1 - d) * alpha + (o * tau_p * f_weight) + (omega * f2 * o2)
    beta = (1 - d) * beta + ((1 - o) * tau_n * f_weight) + (omega * f2 * (1 - o2))

    alpha = T.maximum(T.power(0.1, 10), alpha)
    beta = T.maximum(T.power(0.1, 10), beta)

    value = alpha / (alpha + beta)

    var = (alpha * beta) / (T.pow(alpha + beta, 2) * (alpha + beta + 1))

    return (value, alpha, beta, var)


def leaky_beta_asymmetric_fixation_1b(o, t, o2, f1, f2, v, alpha, beta, d, omega, tau_p, tau_n, gamma):

    """
    
    Forgetful beta model with asymmetric updating and continuous fixation weighting of learning
    -------------------------------------------------------------------------------------------
    
    Identical to the first fixation model but with weighting updated continuously rather than downweighting the least fixated option.

    Args:
        o: Trial outcome
        t: Time (not used)
        o2: Outcome of the other stimulus
        f1: Fixation duration proportion for this stimulus
        f2: Fixation duration proportion for the other stimulus
        v: Previous trial value estimate (not used)
        alpha: Starting alpha
        beta: Starting beta
        d: Decay (forgetting) rate
        omega: Weight of the other stimulus outcome
        tau_p: Positive update weight
        tau_n: Negative update weight
        gamma: Influence of fixation weighting on updates

    Returns:
        Mean: Estimated probability on the current trial (mean of beta distribution)
        Alpha: Alpha value on current trial
        Beta: Beta value on current trial
        Var: Variance of beta distribution

    """
   
    f_weight = T.switch(T.ge(f1, f2), 1, 1 - (f2 - f1) * gamma)    

    alpha = (1 - d) * alpha + (o * tau_p * f_weight) + (omega * f2 * o2)
    beta = (1 - d) * beta + ((1 - o) * tau_n * f_weight) + (omega * f2 * (1 - o2))

    alpha = T.maximum(T.power(0.1, 10), alpha)
    beta = T.maximum(T.power(0.1, 10), beta)

    value = alpha / (alpha + beta)

    var = (alpha * beta) / (T.pow(alpha + beta, 2) * (alpha + beta + 1))

    return (value, alpha, beta, var)


def leaky_beta_asymmetric_fixation_2(o, t, o2, f1, f2, v, alpha, beta, d, omega, tau_p, tau_n, theta):

    """
    
    Forgetful beta model with asymmetric updating and fixation weighting of value
    --------------------------------------------------------------------------------
    
    Fixation weighting in this model is achieved by giving a bonus to the value of the most attended stimulus, which is
    relative to the difference between the most and least attended fixation proportions. This is achieved by adding this
    difference to alpha of the most attended stimulus, weighted by theta.

    E.g. If stimulus 1 is fixated 70% of the time and stimulus 2 is fixated 30% of the time, alpha of stimulus 1 will receive
    a bonus of 0.4 with theta = 1, or 0.2 if theta = 0.5.

    Args:
        o: Trial outcome
        t: Time (not used)
        o2: Outcome of the other stimulus
        f1: Fixation duration proportion for this stimulus
        f2: Fixation duration proportion for the other stimulus
        v: Previous trial value estimate (not used)
        alpha: Starting alpha
        beta: Starting beta
        d: Decay (forgetting) rate
        omega: Weight of the other stimulus outcome
        tau_p: Positive update weight
        tau_n: Negative update weight
        theta: Weighting on fixation-dependent bonus to alpha

    Returns:
        Mean: Estimated probability on the current trial (mean of beta distribution)
        Alpha: Alpha value on current trial
        Beta: Beta value on current trial
        Var: Variance of beta distribution

    """

    alpha = (1 - d) * alpha + (o * tau_p) + (omega * f2 * o2) + f1 * theta
    beta = (1 - d) * beta + ((1 - o) * tau_n) + (omega * f2 * (1 - o2))

    alpha = T.maximum(T.power(0.1, 10), alpha)
    beta = T.maximum(T.power(0.1, 10), beta)

    value = alpha / (alpha + beta)

    var = (alpha * beta) / (T.pow(alpha + beta, 2) * (alpha + beta + 1))

    return (value, alpha, beta, var)


def leaky_beta_asymmetric_fixation_2b(o, t, o2, f1, f2, v, alpha, beta, d, omega, tau_p, tau_n, theta):

    """
    
    Forgetful beta model with asymmetric updating and continuous fixation weighting of value
    ----------------------------------------------------------------------------------------
    
    Identical to the second fixation model but with weighting updated continuously rather than 
    downweighting the least fixated option.

    Args:
        o: Trial outcome
        t: Time (not used)
        o2: Outcome of the other stimulus
        f1: Fixation duration proportion for this stimulus
        f2: Fixation duration proportion for the other stimulus
        v: Previous trial value estimate (not used)
        alpha: Starting alpha
        beta: Starting beta
        d: Decay (forgetting) rate
        omega: Weight of the other stimulus outcome
        tau_p: Positive update weight
        tau_n: Negative update weight
        theta: Weighting on fixation-dependent bonus to alpha

    Returns:
        Mean: Estimated probability on the current trial (mean of beta distribution)
        Alpha: Alpha value on current trial
        Beta: Beta value on current trial
        Var: Variance of beta distribution

    """

    alpha = (1 - d) * alpha + (o * tau_p) + (omega * f2 * o2) + T.largest(0, f1 - f2) * theta
    beta = (1 - d) * beta + ((1 - o) * tau_n) + (omega * f2 * (1 - o2))

    alpha = T.maximum(T.power(0.1, 10), alpha)
    beta = T.maximum(T.power(0.1, 10), beta)

    value = alpha / (alpha + beta)

    var = (alpha * beta) / (T.pow(alpha + beta, 2) * (alpha + beta + 1))

    return (value, alpha, beta, var)
