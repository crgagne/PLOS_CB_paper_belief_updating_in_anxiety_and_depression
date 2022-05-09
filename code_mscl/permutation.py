import numpy as np
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import pandas as pd

def pearson_permutation_test(x,y, N=10_000, seed=1):
    r_s = []
    p_s = []
    np.random.seed(seed)
    for perm in range(N):
        x_fold = np.random.permutation(x)
        y_fold = np.random.permutation(y)
        r,p = pearsonr(x_fold,y_fold)
        r_s.append(r); p_s.append(p)
    r,p = pearsonr(x,y)
    p_perm = np.mean(np.abs(r_s)>np.abs(r)) # two-sided
    return(p_perm,r,p,r_s,p_s)


def pearson_diff_permutation_test(x,y,z, N=10_000, seed=1):
    r_s = []
    p_s = []
    np.random.seed(seed)
    for perm in range(N):
        x_fold = np.random.permutation(x)
        y_fold = np.random.permutation(y)
        z_fold = np.random.permutation(z)
        r_xy,p_xy = pearsonr(x_fold,y_fold)
        r_xz,p_xz = pearsonr(x_fold,z_fold)
        r_diff = r_xy-r_xz

        r_s.append(r_diff)#; p_s.append(p)
    r_xy,_ = pearsonr(x,y)
    r_xz,_ = pearsonr(x,z)
    r_diff = r_xy-r_xz
    p_perm = np.mean(np.abs(r_s)>np.abs(r_diff)) # two-sided
    return(p_perm,r_diff,r_s,p_s)

def pearson_diff_permutation_test_paired(x,y,z, N=10_000, seed=1):
    r_s = []
    p_s = []
    np.random.seed(seed)
    for perm in range(N):
        i_perm = np.random.permutation(np.arange(len(x)))
        i_perm2 = np.random.permutation(np.arange(len(x)))
        #import pdb; pdb.set_trace()

        x_fold = x[i_perm2]
        y_fold = y[i_perm]
        z_fold = z[i_perm]
        r_xy,p_xy = pearsonr(x_fold,y_fold)
        r_xz,p_xz = pearsonr(x_fold,z_fold)
        r_diff = r_xy-r_xz

        r_s.append(r_diff)#; p_s.append(p)
    r_xy,_ = pearsonr(x,y)
    r_xz,_ = pearsonr(x,z)
    r_diff = r_xy-r_xz
    p_perm = np.mean(np.abs(r_s)>np.abs(r_diff)) # two-sided
    return(p_perm,r_diff,r_s,p_s)


def spearman_diff_with_bootstrap(x,y,z, N=10_000, seed=1):
    r_s = []
    p_s = []
    np.random.seed(seed)
    for perm in range(N):
        i_perm = np.random.choice(np.arange(len(x)), size=len(x),replace=True)
        #import pdb; pdb.set_trace()

        x_fold = x[i_perm]
        y_fold = y[i_perm]
        z_fold = z[i_perm]
        #r_xy,p_xy = pearsonr(x_fold,y_fold)
        #r_xz,p_xz = pearsonr(x_fold,z_fold)
        r_xy,p_xy = spearmanr(x_fold,y_fold)
        r_xz,p_xz = spearmanr(x_fold,z_fold)
        r_diff = r_xy-r_xz

        r_s.append(r_diff)#; p_s.append(p)
    r_xy,_ = pearsonr(x,y)
    r_xz,_ = pearsonr(x,z)
    r_diff = r_xy-r_xz
    p_perm = np.mean(np.abs(r_s)>np.abs(r_diff)) # two-sided
    return(p_perm,r_diff,r_s,p_s)


def regression_diff_permutation_test(trait,bias,prior, N=10_000, seed=1):

    X = sm.add_constant(np.vstack((bias,prior)).T)
    X = pd.DataFrame(X,columns=['constant','bias','prior'])
    results_base = sm.OLS(trait,X).fit()
    param_diff = results_base.params.bias-results_base.params.prior

    diffs = []
    p_s = []
    np.random.seed(seed)
    for perm in range(N):
        trait_fold = np.random.permutation(trait)
        prior_fold = np.random.permutation(prior)
        bias_fold = np.random.permutation(bias)
        X = sm.add_constant(np.vstack((bias_fold,prior_fold)).T)
        X = pd.DataFrame(X,columns=['constant','bias','prior'])
        results_base = sm.OLS(trait_fold,X).fit()
        param_diff_fold = results_base.params.bias-results_base.params.prior

        diffs.append(param_diff_fold)#; p_s.append(p)

    p_perm = np.mean(np.abs(diffs)>np.abs(param_diff)) # two-sided
    return(p_perm,param_diff,diffs,p_s)

def regression_diff_permutation_test2(dep,anx,bias,prior, N=10_000, seed=1):

    X = sm.add_constant(np.vstack((bias,prior)).T)
    X = pd.DataFrame(X,columns=['constant','bias','prior'])
    results_dep = sm.OLS(dep,X).fit()
    results_anx = sm.OLS(anx,X).fit()
    bias_diff = results_anx.params.bias-results_dep.params.bias
    prior_diff = results_dep.params.prior-results_anx.params.prior

    diffs_prior = []
    diffs_bias = []
    p_s = []
    np.random.seed(seed)
    for perm in range(N):
        anx_fold = np.random.permutation(anx)
        dep_fold = np.random.permutation(dep)
        prior_fold = np.random.permutation(prior)
        bias_fold = np.random.permutation(bias)
        X = sm.add_constant(np.vstack((bias_fold,prior_fold)).T)
        X = pd.DataFrame(X,columns=['constant','bias','prior'])
        results_dep = sm.OLS(dep_fold,X).fit()
        results_anx = sm.OLS(anx_fold,X).fit()
        bias_diff_fold = results_anx.params.bias-results_dep.params.bias
        prior_diff_fold = results_dep.params.prior-results_anx.params.prior

        diffs_prior.append(prior_diff_fold)
        diffs_bias.append(bias_diff_fold)

    p_perm_bias = np.mean(np.abs(diffs_bias)>np.abs(bias_diff)) # two-sided
    p_perm_prior = np.mean(np.abs(diffs_prior)>np.abs(prior_diff)) # two-sided
    return(p_perm_bias,p_perm_prior,diffs_bias,diffs_prior, bias_diff, prior_diff)
