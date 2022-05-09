import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as Beta
from scipy.stats import laplace
from scipy.stats import ttest_rel,pearsonr,spearmanr, halfnorm, norm, uniform,t

def replace_bounds(bounds,list_to_check):
    '''replaces params in list that are outside the bounds with the boundaries'''
    for bi,bnd in enumerate(bounds):
        if list_to_check[bi]<bnd[0]:
            list_to_check[bi]=bnd[0]
        if list_to_check[bi]>bnd[1]:
            list_to_check[bi]=bnd[1]
    return(list_to_check)

def new_vs():
    x = np.abs(t(4,50,200).rvs(1))[0]
    if x<100:
        x = np.random.uniform(100,400)
    return(x)

def resample_within_two_indices(params,param_names,modelname):
    name1s = {'wp':'wn','alpha_pos':'alpha_neg'}#,'r_pos':'r_neg'}

    for name1 in param_names: # search through each parameter
        if name1 in name1s.keys(): # check for positive version of parameters
            name2 = name1s[name1]
            param_idx1 = param_names.index(name1)
            param_idx2 = param_names.index(name2)
            gen_set1 = model_specs[modelname]['gen_sets'][param_idx1]
            gen_set2 = model_specs[modelname]['gen_sets'][param_idx2]

            # if so, get value indices for two parameters
            val_idx1 = gen_set1.index(params[param_idx1])
            val_idx2 = gen_set2.index(params[param_idx2])
            if np.abs(val_idx1-val_idx2)<=2: # if within 2 value indices
                new_possible_val_idxs = [i for i in range(len(gen_set1))]
                for rm in [-2,-1,0,1,2]:
                    if val_idx2+rm < len(gen_set1) and val_idx2+rm >0:
                        new_possible_val_idxs.remove(val_idx2+rm)
                new_val = gen_set1[np.random.choice(new_possible_val_idxs)]
                params[param_idx1]=new_val
    return(params)

def transform_params(params, transform_specs, direction='estimation_space_to_model_space'):
    params_transformed = []
    for param,t_specs in zip(params, transform_specs):
        if len(t_specs)>0:
            min, max, trans_type = t_specs[0], t_specs[1], t_specs[2]
            if trans_type=='exp/log':
                if direction=='estimation_space_to_model_space':
                    param = np.exp(param)
                else:
                    param = np.log(param)
        params_transformed.append(param)
    return(params_transformed)

def fit_model(X,y,model,
        modelbasename='None',
        param_names = [],
        bnds = ((),),
        extra_args={},num_starts=10,
        include_trials=None,
        prior=False,
        priorfns=None,
        param_inits=None,
        transform=False,
        transform_specs=[]):

    if 'minfun' not in extra_args:
        extra_args['minfun']='SLSQP'
    if 'report_dist' not in extra_args:
        extra_args['report_dist']='beta;mean'
    if 'model' not in extra_args:
        extra_args['model']='decay_prior'

    # specify neglog lik function
    def loss(params,X,y,prior=prior,priorfns=priorfns,bnds=None,include_trials=include_trials,transform=False,transform_specs=()):

        if transform:
            params=transform_params(params, transform_specs)

        if include_trials is None:
            include_trials = np.arange(len(y))
        if 'report_dist' in extra_args:
            nllk,_ = model(params,X,y,model=extra_args['model'],report_dist=extra_args['report_dist'])
        else:
            nllk,_ = model(params,X,y,model=extra_args['model'],report_dist=extra_args['report_dist'])

        nllk = nllk[include_trials]
        nllk = np.sum(nllk)

        if prior==True:
            for pi,param in enumerate(params):
                if priorfns[pi](param)<0:
                    import pdb; pdb.set_trace()
                priorr = -1.0*np.log(priorfns[pi](param))
                priorr = np.min([100,priorr])
                nllk+=priorr
        return(nllk)

	# start multiple times, find minimium, pick lowest
    results_vec = []
    llk = []
    for starts in range(num_starts):
        params_init = []
        if param_inits is None:
            for bnd in bnds:
                params_init.append(np.random.uniform(bnd[0],bnd[1],size=1))
        else:
            for pi in param_inits:
                params_init.append(np.random.uniform(pi[0],pi[1],size=1))

        results_temp = minimize(loss,params_init, method=extra_args['minfun'],
                                args=(X,y,prior,priorfns,bnds,include_trials,transform,transform_specs),
                                bounds=bnds)
        results_vec.append(results_temp)
        llk.append(results_temp.fun)
    results = results_vec[np.argmin(llk)]
    best_params = list(results.x)

    # fit model again with optimal parameters to get nllk
    if transform:
        best_params=transform_params(best_params, transform_specs)
    nllk,extra_out  = model(best_params,X,y,model=extra_args['model'],report_dist=extra_args['report_dist'])

    # number of params and data
    k = len(params_init)
    n  = len(y)

    # create series for parameters
    params = pd.Series(data=best_params,index=param_names)

    # adjust nllk based on excluding trials (not using usually)
    if include_trials is None:
        include_trials = np.arange(len(y))
    nllk = np.sum(nllk[include_trials])
    nllk_w_prior=nllk

    # adjust nllk for priors
    if prior==True:
        for pi,param in enumerate(params):
            if priorfns[pi](param)<0:
                import pdb; pdb.set_trace()
            priorr = -1.0*np.log(priorfns[pi](param))
            priorr = np.min([100,priorr])
            nllk_w_prior+=priorr

    # BIC/AIC
    AIC = 2*nllk + 2*k
    BIC = 2*nllk + np.log(n)*k
    AIC_w_prior = 2*nllk_w_prior + 2*k
    BIC_w_prior = 2*nllk_w_prior + np.log(n)*k

    # return model info
    out=extra_out

    out['modelbasename']=modelbasename+extra_args['model']
    out['bic']=BIC
    out['aic']=AIC
    out['bic_w_prior']=BIC_w_prior
    out['aic_w_prior']=AIC_w_prior
    out['nllk']=nllk
    out['nllk_w_prior']=nllk_w_prior
    out['params']=params
    out['results']=results

    out['r2']=np.corrcoef(y,out['u_s'])[0,1]**2
    resid = out['u_s']-y # stated belief mean
    sse = np.sum(resid**2)
    mse = np.mean(resid**2)
    out['resid']=resid
    out['sse']=sse
    out['mse']=mse
    out['rmse']=np.sqrt(mse)
    out['llk_vec']=llk

    return(out)

def logit(x):
    return(np.log((x/(1-x))))
def invlogit(x):
    return(1.0/(1.0+np.exp(-1.0*x)))


def model_bayes_w_report(params,X,y,model='decay_prior',report_dist='beta;mean'):
    '''Class for all Bayesian models.
    '''

    # observed feedback / evidence (1=positive, 0=negative)
    feedback = X['feedback'].values

    # vectors across trials
    u_u = np.zeros(len(y))
    v_u = np.zeros(len(y))
    var_u = np.zeros(len(y))
    alpha_u = np.zeros(len(y))
    beta_u = np.zeros(len(y))
    u_s = np.zeros(len(y))
    v_s = np.zeros(len(y))
    var_s = np.zeros(len(y))
    alpha_s = np.zeros(len(y))
    beta_s = np.zeros(len(y))

    neglogpost = np.zeros(len(y))
    a = 1
    gamma = 1
    Ta = 1
    Tb = 1 # not relevant`

    if 'basic_no_reported' in model:
        alpha0 = params[0]
        beta0 = params[1]
        v_s = 0
    elif 'basic' in model and 'no_reported' not in model:
        alpha0 = params[0]
        beta0 = params[1]
        v_s = params[2]
    elif model=='asym_no_reported':
        alpha0 = params[0]
        beta0 = params[1]
        w = params[2]
    elif model=='asym_mean_no_reported':
        alpha0 = params[0]
        beta0 = params[1]
        w = params[2]
        a = params[3] # amplifies
    elif 'w_no_reported' in model:
        alpha0 = params[0]
        beta0 = params[1]
        w = params[2]
        gamma = params[3]
        #v_s = params[4]
    elif ('two_rate' in model) and ('no_reported' in model) and ('no_decay' not in model):
        alpha0 = params[0]
        beta0 = params[1]
        wp = params[2]
        wn = params[3]
        gamma = params[4]
    elif ('two_rate' in model) and ('no_reported' in model) and ('no_decay' in model):
        alpha0 = params[0]
        beta0 = params[1]
        wp = params[2]
        wn = params[3]
    elif ('two_rate' in model) and ('no_reported' not in model) and ('no_decay' not in model):
        alpha0 = params[0]
        beta0 = params[1]
        wp = params[2]
        wn = params[3]
        gamma = params[4]
        v_s = params[5]
    elif ('two_rate' in model) and ('no_reported' not in model) and ('no_decay' in model):
        alpha0 = params[0]
        beta0 = params[1]
        wp = params[2]
        wn = params[3]
        v_s = params[4]
    elif 'two_vs' in model:
        alpha0 = params[0]
        beta0 = params[1]
        w = params[2]
        gamma = params[3]
        v_s1 = params[4]
        v_s2 = params[5]
        v_s = v_s1 # start with v_s1 and switch
    elif 'no_decay' in model:
        alpha0 = params[0]
        beta0 = params[1]
        w = params[2]
        v_s = params[3]
        gamma = 1
        Ta = 1
        Tb = 1 # not relevant
    else:
        alpha0 = params[0]
        beta0 = params[1]
        w = params[2]
        gamma = params[3]
        v_s = params[4]

    ### first trial ###

    # determine implied alpha and beta for 'underlying' beta-distribution
    alpha_u[0] = alpha0
    beta_u[0] = beta0
    u_u[0] = alpha_u[0]/(alpha_u[0]+beta_u[0])
    var_u[0] = (alpha_u[0]*beta_u[0]) / (((alpha_u[0] + beta_u[0])**2)*(alpha_u[0]+beta_u[0]+1))

    # link stated means to underlying belief
    if report_dist=='beta;mean':
        u_s[0] = u_u[0]
    elif report_dist=='beta;median' or report_dist=='beta;50':
        u_s[0] = Beta(alpha_u[0],beta_u[0]).median()
    elif report_dist=='beta;99':
        u_s[0] = Beta(alpha_u[0],beta_u[0]).ppf(0.99)
    elif report_dist=='beta;90':
        u_s[0] = Beta(alpha_u[0],beta_u[0]).ppf(0.90)
    elif report_dist=='beta;75':
        u_s[0] = Beta(alpha_u[0],beta_u[0]).ppf(0.75)
    elif report_dist=='beta;25':
        u_s[0] = Beta(alpha_u[0],beta_u[0]).ppf(0.25)
    elif report_dist=='beta;10':
        u_s[0] = Beta(alpha_u[0],beta_u[0]).ppf(0.1)
    elif report_dist=='beta;1':
        u_s[0] = Beta(alpha_u[0],beta_u[0]).ppf(0.01)
    elif report_dist=='laplace;mean':
        u_s[0] = u_u[0]

    # determine implied alpha and beta for 'stated belief' beta-distribution
    if 'no_reported' in model:
        alpha_s[0] = alpha_u[0]
        beta_s[0] = beta_u[0]
    else:
        alpha_s[0] = u_s[0]*v_s
        beta_s[0] = v_s-alpha_s[0]

    if 'beta' in report_dist:
        neglogpost[0] = -1*Beta.logpdf(y[0],alpha_s[0],beta_s[0])

    var_s[0] = (alpha_s[0]*beta_s[0]) / (((alpha_s[0] + beta_s[0])**2)*(alpha_s[0]+beta_s[0]+1))

    for t in range(1,len(y)):

        ### Underlying Belief Distribution ###

        # posterior mean decays towards 50%
        if 'decay_50' in model:
            Ta=1
            Tb=1
        elif 'decay_prior' in model:
            Ta=alpha_u[0]
            Tb=beta_u[0]

        # update
        if 'two_rate' in model:
            alpha_u[t] = gamma*alpha_u[t-1]+ (1-gamma)*Ta + wp*feedback[t]#*scale
            beta_u[t] = gamma*beta_u[t-1] + (1-gamma)*Tb + wn*(1-feedback[t])#*scale
        elif 'basic' in model:
            alpha_u[t] = alpha_u[t-1] + feedback[t]
            beta_u[t] = beta_u[t-1] + (1-feedback[t])
        else:
            alpha_u[t] = gamma*alpha_u[t-1]+ (1-gamma)*Ta + a*w*feedback[t]#*scale
            beta_u[t] = gamma*beta_u[t-1] + (1-gamma)*Tb + a*1/w*(1-feedback[t])#*scale

        # mean and variance for underlying belief
        u_u[t] = alpha_u[t]/(alpha_u[t]+beta_u[t])
        var_u[t] = (alpha_u[t]*beta_u[t]) / (((alpha_u[t] + beta_u[t])**2)*(alpha_u[t]+beta_u[t]+1))

        # link stated means to underlying belief
        if report_dist=='beta;mean':
            u_s[t] = u_u[t]
        elif report_dist=='beta;median' or report_dist=='beta;50':
            u_s[t] = Beta(alpha_u[t],beta_u[t]).median()
        elif report_dist=='beta;99':
            u_s[t] = Beta(alpha_u[t],beta_u[t]).ppf(0.99)
        elif report_dist=='beta;90':
            u_s[t] = Beta(alpha_u[t],beta_u[t]).ppf(0.90)
        elif report_dist=='beta;75':
            u_s[t] = Beta(alpha_u[t],beta_u[t]).ppf(0.75)
        elif report_dist=='beta;25':
            u_s[t] = Beta(alpha_u[t],beta_u[t]).ppf(0.25)
        elif report_dist=='beta;10':
            u_s[t] = Beta(alpha_u[t],beta_u[t]).ppf(0.1)
        elif report_dist=='beta;1':
            u_s[t] = Beta(alpha_u[t],beta_u[t]).ppf(0.01)
        elif report_dist=='laplace;mean':
            u_s[t] = u_u[t]

        if 'two_vs' in model and t>=10:
            v_s = v_s2

        # determine implied alpha and beta for 'stated belief' beta-distribution
        if 'no_reported' in model:
            alpha_s[t] = alpha_u[t]
            beta_s[t] = beta_u[t]
        else:
            alpha_s[t] = u_s[t]*v_s
            beta_s[t] = v_s-alpha_s[t]

        if 'beta' in report_dist:
            neglogpost[t] = -1*Beta.logpdf(y[t],alpha_s[t],beta_s[t])

        u_s[t] = alpha_s[t] / (alpha_s[t] + beta_s[t])
        var_s[t] = (alpha_s[t]*beta_s[t]) / (((alpha_s[t] + beta_s[t])**2)*(alpha_s[t]+beta_s[t]+1))

    extra_out = {}
    for name,var in zip(['u_u','v_u','var_u','alpha_u','beta_u','u_s','v_s','var_s','alpha_s','beta_s'],
        [u_u,v_u,var_u,alpha_u,beta_u,u_s,v_s,var_s,alpha_s,beta_s]):
        extra_out[name]=var

    # return
    return(neglogpost,extra_out)


def model_RW_update_w_report(params,X,y,model='decay_prior',report_dist='beta;mean'):
    '''Class for all Rescorla-Wagner models.
    '''
    # observed feedback / evidence (1=positive, 0=negative)
    feedback = X['feedback']

    # vectors across trials
    u_u = np.zeros(len(y))
    v_u = np.zeros(len(y))*np.nan # unused
    var_u = np.zeros(len(y))*np.nan # unused
    alpha_u = np.zeros(len(y))*np.nan # unused
    beta_u = np.zeros(len(y))*np.nan # unused
    u_s = np.zeros(len(y))
    v_s = np.zeros(len(y))
    var_s = np.zeros(len(y))
    alpha_s = np.zeros(len(y))
    beta_s = np.zeros(len(y))

    neglogpost = np.zeros(len(y))
    update_total = 1.0
    repeat_scale = 1.0
    if 'full' in model:
        alpha_pos = params[0]
        alpha_neg = params[1]
        r_pos = params[2]
        r_neg = params[3]
        v_s = params[4]
        u_u0 = params[5]
    if 'samelr_fbscaling' in model:
        alpha_pos = alpha_neg = params[0]
        r_pos = params[1]
        r_neg = params[2]
        v_s = params[3]
        u_u0 = params[4]
    if 'difflr_relfbscaling' in model:
        alpha_pos = params[0]
        alpha_neg = params[1]
        r_pos = params[2]
        r_neg = 0
        v_s = params[3]
        u_u0 = params[4]
    if 'difflr_nofbscaling' in model:
        alpha_pos = params[0]
        alpha_neg = params[1]
        r_pos = 1
        r_neg = 0
        v_s = params[2]
        u_u0 = params[3]
    if 'samelr_relfbscaling' in model:
        alpha_pos = alpha_neg = params[0]
        r_pos = params[1]
        r_neg = 0
        v_s = params[2]
        u_u0 = params[3]
    if 'samelr_fbshift' in model:
        alpha_pos = alpha_neg = params[0]
        r_pos = 1+params[1]
        r_neg = 0+params[1]
        v_s = params[2]
        u_u0 = params[3]
    if 'samelr_nofbscaling' in model:
        alpha_pos = alpha_neg = params[0]
        r_pos = 1
        r_neg = 0
        v_s = params[1]
        u_u0 = params[2]
    if ('surprise_extraparam' in model) or ('surprise_binary_extraparam' in model):
        surp_param = params[4]
    else:
        surp_param = None

    if 'surprise' in model:
        surprise = X['surprise']
    else:
        surprise = np.ones_like(feedback)

    if 'surprise_binary' in model:
        surprise = X['surprise_binary']


    if 'relcompfeedback_only' in model:
        f_lambda = 0.
        rel_comp_feedback = X['rel_comp']
    elif 'relcompfeedback_extraparam' in model:
        f_lambda = params[4]
        rel_comp_feedback = X['rel_comp']
    else:
        f_lambda = None

    ### first trial ###
    u_u[0] = u_u0

    # link stated means to underlying belief
    u_s[0] = u_u[0]

    # determine implied alpha and beta for 'stated belief' beta-distribution
    alpha_s[0] = u_s[0]*v_s
    beta_s[0] = v_s-alpha_s[0]

    neglogpost[0] = -1*Beta.logpdf(y[0],alpha_s[0],beta_s[0])
    var_s[0] = (alpha_s[0]*beta_s[0]) / (((alpha_s[0] + beta_s[0])**2)*(alpha_s[0]+beta_s[0]+1))

    for t in range(1,len(y)):

        if surp_param is not None: # multiplicative surprise models
            if feedback[t]==1:
                u_u[t] = u_u[t-1] + alpha_pos*((r_pos-u_u[t-1]))*(1.+surp_param*(surprise[t]-0.5))
            else:
                u_u[t] = u_u[t-1] + alpha_neg*((r_neg-u_u[t-1]))*(1.+surp_param*(surprise[t]-0.5))
        elif f_lambda is not None: # feedback with relative comparison
            combined_feedback = f_lambda*feedback[t] + (1.-f_lambda)*rel_comp_feedback[t]
            if feedback[t]==1.:
                alpha=alpha_pos; r=r_pos
            else:
                alpha=alpha_neg; r=r_neg;
                assert r_neg ==0
            u_u[t] = u_u[t-1] + alpha*((r*combined_feedback-u_u[t-1]))
        else: # all original models 1-13 and no parameter surprise model
            if feedback[t]==1:
                u_u[t] = u_u[t-1] + alpha_pos*((r_pos-u_u[t-1]))*surprise[t]
            else:
                u_u[t] = u_u[t-1] + alpha_neg*((r_neg-u_u[t-1]))*surprise[t]

        u_u[t] = np.max((np.min((u_u[t],0.99)),0.01))

        # link stated means to underlying belief
        if report_dist=='beta;mean':
            u_s[t] = u_u[t]

        # determine implied alpha and beta for 'stated belief' beta-distribution
        alpha_s[t] = u_s[t]*v_s
        beta_s[t] = v_s-alpha_s[t]

        if 'beta' in report_dist:
            neglogpost[t] = -1*Beta.logpdf(y[t],alpha_s[t],beta_s[t])

        u_s[t] = alpha_s[t] / (alpha_s[t] + beta_s[t])
        var_s[t] = (alpha_s[t]*beta_s[t]) / (((alpha_s[t] + beta_s[t])**2)*(alpha_s[t]+beta_s[t]+1))

    extra_out = {}
    for name,var in zip(['u_u','v_u','var_u','alpha_u','beta_u','u_s','v_s','var_s','alpha_s','beta_s'],
        [u_u,v_u,var_u,alpha_u,beta_u,u_s,v_s,var_s,alpha_s,beta_s]):
        extra_out[name]=var

    # return
    return(neglogpost,extra_out)



model_specs = {}

## MAIN MODELS ##

# Model 1
model_specs['model_bayes_wreport_asym_no_reported']={
'modelbasename': 'model_bayes_wreport',
'model': model_bayes_w_report,
'param_names':['alpha0','beta0','w'],
'bnds':((2,100),(2,100),(0,5)),
'pis':((2,20),(2,20),(0.5,2)),
'model_extra':{'model':'asym_no_reported'},
'priorfns':{0:lambda x: halfnorm(0,50).pdf(x),
          1:lambda x: halfnorm(0,50).pdf(x),
          2:lambda x: norm(1,0.5).pdf(x)},
'priorfns_gen':{0:lambda x: np.abs(t(5,1,20).rvs(x)),
          1:lambda x: np.abs(t(5,1,20).rvs(x)),
          2:lambda x: np.abs(norm(1,0.5).rvs(x))},
'gen_sets':[list(np.abs(t(5,1,10).rvs(100))),
            list(np.abs(t(5,1,10).rvs(100))),
            list(np.random.uniform(0.25,0.75,50))+list(np.random.uniform(1.25,3,50)),
            ],
}


# Model 2
model_specs['model_bayes_wreport_basic_no_reported']={
'modelbasename': 'model_bayes_wreport',
'model': model_bayes_w_report,
'param_names':['alpha0','beta0'],
'bnds':((2,100),(2,100)),
'pis':((2,20),(2,20)),
'model_extra':{'model':'basic_no_reported'},
'priorfns':{0:lambda x: halfnorm(0,50).pdf(x),
          1:lambda x: halfnorm(0,50).pdf(x)},
'priorfns_gen':{0:lambda x: np.abs(t(5,1,20).rvs(x)),
          1:lambda x: np.abs(t(5,1,20).rvs(x))},
'gen_sets':[list(np.abs(t(5,1,10).rvs(100))),
            list(np.abs(t(5,1,10).rvs(100))),
            ],
}

# Model 3
model_specs['model_RW_update_w_report_samelr_relfbscaling']={
'modelbasename': 'model_RW_update_w_report',
'model': model_RW_update_w_report,
'param_names':['alpha','r_pos','v_s','u_u0'],
'bnds':((0.01,0.99),(0,5),(2,1000),(0.01,0.99)),
'pis':((0.1,0.5),(0,2),(100,500),(0.1,0.9)),
'model_extra':{'model':'samelr_relfbscaling'},
'priorfns':{0:lambda x: halfnorm(0,0.5).pdf(x),
          1:lambda x: halfnorm(0,2).pdf(x),
          2:lambda x: halfnorm(0,500).pdf(x),
          3:lambda x: norm(0.5,0.5).pdf(x),
          },
'priorfns_gen':{0:lambda x: np.abs(halfnorm(0,0.25).rvs(x)),
          1:lambda x: np.abs(halfnorm(0,1.5).rvs(x)),
          2:lambda x: np.abs(t(4,50,200).rvs(x)),
          3:lambda x: np.abs(norm(0.5,0.25).rvs(x)),},
'gen_sets':[[0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25],
            [0.2,0.4,0.8,1.2,1.4,1.8,2.2,2.5],
            [new_vs() for i in range(100)],
            list(np.random.uniform(0.1,0.9,100)),
            ],
}

# Model 4
model_specs['model_RW_update_w_report_samelr_nofbscaling']={
'modelbasename': 'model_RW_update_w_report',
'model': model_RW_update_w_report,
'param_names':['alpha','v_s','u_u0'],
'bnds':((0.01,0.99),(2,1000),(0.01,0.99)),
'pis':((0.1,0.5),(100,500),(0.1,0.9)),
'model_extra':{'model':'samelr_nofbscaling'},
'priorfns':{0:lambda x: halfnorm(0,0.5).pdf(x),
          1:lambda x: halfnorm(0,500).pdf(x),
          2:lambda x: norm(0.5,0.5).pdf(x),
          },
'priorfns_gen':{0:lambda x: np.abs(halfnorm(0,0.25).rvs(x)),
          1:lambda x: np.abs(t(4,50,200).rvs(x)),
          2:lambda x: np.abs(norm(0.5,0.25).rvs(x)),},
'gen_sets':[[0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25],
            [new_vs() for i in range(100)],
            list(np.random.uniform(0.1,0.9,100)),
            ],
}

## SUPPLEMENTAL MODELS ##

# Model 5
model_specs['model_RW_update_w_report_difflr_nofbscaling']={
'modelbasename': 'model_RW_update_w_report',
'model': model_RW_update_w_report,
'param_names':['alpha_pos','alpha_neg','v_s','u_u0'],
'bnds':((0.01,0.99),(0.01,0.99),(2,1000),(0.01,0.99)),
'pis':((0.1,0.5),(0.1,0.5),(100,500),(0.1,0.9)),
'model_extra':{'model':'difflr_nofbscaling'},
'priorfns':{0:lambda x: halfnorm(0,0.5).pdf(x),
          1:lambda x: halfnorm(0,0.5).pdf(x),
          2:lambda x: halfnorm(0,500).pdf(x),
          3:lambda x: norm(0.5,0.5).pdf(x),
          },
'priorfns_gen':{0:lambda x: np.abs(halfnorm(0,0.25).rvs(x)),
          1:lambda x: np.abs(halfnorm(0,0.25).rvs(x)),
          2:lambda x: np.abs(t(4,50,200).rvs(x)),
          3:lambda x: np.abs(norm(0.5,0.25).rvs(x)),},
'gen_sets':[[0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25],
           [0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2],
            [new_vs() for i in range(100)],
            list(np.random.uniform(0.1,0.9,100)),
            ],
}

# Model 6
model_specs['model_RW_update_w_report_samelr_fbscaling']={
'modelbasename': 'model_RW_update_w_report',
'model': model_RW_update_w_report,
'param_names':['alpha','r_pos','r_neg','v_s','u_u0'],
'bnds':((0.01,0.99),(0,5),(-5,0),(2,1000),(0.01,0.99)), #
'pis':((0.1,0.5),(0.5,2),(-0.5,2),(100,500),(0.1,0.9)),
'model_extra':{'model':'samelr_fbscaling'},
'priorfns':{0:lambda x: halfnorm(0,0.5).pdf(x),
          1:lambda x: halfnorm(0,2).pdf(x),
          2:lambda x: halfnorm(0,2).pdf(-1*x),
          3:lambda x: halfnorm(0,500).pdf(x),
          4:lambda x: norm(0.5,0.5).pdf(x),
          },
'priorfns_gen':{0:lambda x: np.abs(halfnorm(0,0.25).rvs(x)),
          1:lambda x: np.abs(halfnorm(0,1.5).rvs(x)),
          2:lambda x: -1*np.abs(halfnorm(0,1.5).rvs(x)),
          3:lambda x: np.abs(t(4,50,200).rvs(x)),
          4:lambda x: np.abs(norm(0.5,0.25).rvs(x)),},
'gen_sets':[[0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25],
            [0.2,0.4,0.8,1.2,1.1,1.2,1.4,1.5,1.8,2.2,2.8],
            [0,-0.2,-0.3,-0.4,-0.5,-0.7,-1,-1.2,-1.4,-1.8],
            [new_vs() for i in range(100)],
            list(np.random.uniform(0.1,0.9,100)),
            ],
}

# Model 7
model_specs['model_bayes_wreport_w_no_reported_decay_50']={
'modelbasename': 'model_bayes_wreport',
'model': model_bayes_w_report,
'param_names':['alpha0','beta0','w','gamma'],
'bnds':((2,100),(2,100),(0,5),(0,1)),
'pis':((2,20),(2,20),(0.5,2),(0.8,1)),
'model_extra':{'model':'w_no_reported_decay_50'},
'priorfns':{0:lambda x: halfnorm(0,50).pdf(x),
          1:lambda x: halfnorm(0,50).pdf(x),
          2:lambda x: norm(1,0.5).pdf(x),
          3:lambda x: norm(0.9,0.2).pdf(x)},
'priorfns_gen':{0:lambda x: np.abs(t(5,1,20).rvs(x)),
          1:lambda x: np.abs(t(5,1,20).rvs(x)),
          2:lambda x: np.abs(norm(1,0.5).rvs(x)),
          3:lambda x: np.abs(norm(0.9,0.2).rvs(x))},
'gen_sets':[list(np.abs(t(5,1,10).rvs(100))),
            list(np.abs(t(5,1,10).rvs(100))),
            list(np.random.uniform(0.25,0.75,50))+list(np.random.uniform(1.25,3,50)),
            list(np.random.uniform(0.65,0.95,100)),
            ],
}

# Model 8
model_specs['model_bayes_wreport_w_no_reported_decay_prior']={
'modelbasename': 'model_bayes_wreport',
'model': model_bayes_w_report,
'param_names':['alpha0','beta0','w','gamma'],
'bnds':((2,100),(2,100),(0,5),(0,1)),
'pis':((2,20),(2,20),(0.5,2),(0.8,1)),
'model_extra':{'model':'w_no_reported_decay_prior'},
'priorfns':{0:lambda x: halfnorm(0,50).pdf(x),
          1:lambda x: halfnorm(0,50).pdf(x),
          2:lambda x: norm(1,0.5).pdf(x),
          3:lambda x: norm(0.9,0.2).pdf(x)},
'priorfns_gen':{0:lambda x: np.abs(t(5,1,20).rvs(x)),
          1:lambda x: np.abs(t(5,1,20).rvs(x)),
          2:lambda x: np.abs(norm(1,0.5).rvs(x)),
          3:lambda x: np.abs(norm(0.9,0.2).rvs(x))},
'gen_sets':[list(np.abs(t(5,1,10).rvs(100))),
            list(np.abs(t(5,1,10).rvs(100))),
            list(np.random.uniform(0.25,0.75,50))+list(np.random.uniform(1.25,3,50)),
            list(np.random.uniform(0.65,0.95,100)),
            ],
}

# Model 9
model_specs['model_bayes_wreport_basic']={
'modelbasename': 'model_bayes_wreport',
'model': model_bayes_w_report,
'param_names':['alpha0','beta0','v_s'],
'bnds':((2,100),(2,100),(2,1000)),
'pis':((2,20),(2,20),(100,500)),
'model_extra':{'model':'basic'},
'priorfns':{0:lambda x: halfnorm(0,50).pdf(x),
          1:lambda x: halfnorm(0,50).pdf(x),
          2:lambda x: halfnorm(0,500).pdf(x)},
'priorfns_gen':{0:lambda x: np.abs(t(5,1,10).rvs(x)),
          1:lambda x: np.abs(t(5,1,10).rvs(x)),
          2:lambda x: np.abs(t(4,50,200).rvs(x))},
'gen_sets':[list(np.abs(t(5,1,10).rvs(100))),
            list(np.abs(t(5,1,10).rvs(100))),
            [new_vs() for i in range(100)]
            ],
}

# Model 10
model_specs['model_bayes_wreport_no_decay']={
'modelbasename': 'model_bayes_wreport',
'model': model_bayes_w_report,
'param_names':['alpha0','beta0','w','v_s'],
'bnds':((2,100),(2,100),(0,5),(2,1000)),
'pis':((2,20),(2,20),(0.5,2),(100,500)),
'model_extra':{'model':'no_decay'},
'priorfns':{0:lambda x: halfnorm(0,50).pdf(x),
          1:lambda x: halfnorm(0,50).pdf(x),
          2:lambda x: norm(1,1).pdf(x),
          3:lambda x: halfnorm(0,500).pdf(x)},
'priorfns_gen':{0:lambda x: np.abs(t(5,1,10).rvs(x)),
          1:lambda x: np.abs(t(5,1,10).rvs(x)),
          2:lambda x: np.abs(norm(1,0.5).rvs(x)),
          3:lambda x: np.abs(t(4,50,200).rvs(x))},
'gen_sets':[list(np.abs(t(5,1,10).rvs(100))),
            list(np.abs(t(5,1,10).rvs(100))),
            list(np.random.uniform(0.25,0.75,50))+list(np.random.uniform(1.25,3,50)),
            [new_vs() for i in range(100)]
            ],
}


# Model 11
model_specs['model_bayes_wreport_decay_50']={
'modelbasename': 'model_bayes_wreport',
'model': model_bayes_w_report,
'param_names':['alpha0','beta0','w','gamma','v_s'],
'bnds':((2,100),(2,100),(0,5),(0,1),(2,1000)),
'pis':((2,20),(2,20),(0.5,2),(0.8,1),(100,500)),
'model_extra':{'model':'decay_50'},
'priorfns':{0:lambda x: halfnorm(0,50).pdf(x),
          1:lambda x: halfnorm(0,50).pdf(x),
          2:lambda x: norm(1,1).pdf(x),
          3:lambda x: norm(0.9,0.2).pdf(x),
          4:lambda x: halfnorm(0,500).pdf(x)},
'priorfns_gen':{0:lambda x: np.abs(t(5,1,10).rvs(x)),
          1:lambda x: np.abs(t(5,1,10).rvs(x)),
          2:lambda x: np.abs(norm(1,0.5).rvs(x)),
          3:lambda x: np.abs(norm(0.9,0.2).rvs(x)),
          4:lambda x: np.abs(t(4,50,200).rvs(x))},
'gen_sets':[list(np.abs(t(5,1,10).rvs(100))),
            list(np.abs(t(5,1,10).rvs(100))),
            list(np.random.uniform(0.25,0.75,50))+list(np.random.uniform(1.25,3,50)),
            list(np.random.uniform(0.65,0.95,100)),
            [new_vs() for i in range(100)]
            ],
}

# Model 12
model_specs['model_bayes_wreport_decay_prior']={
'modelbasename': 'model_bayes_wreport',
'model': model_bayes_w_report,
'param_names':['alpha0','beta0','w','gamma','v_s'],
'bnds':((2,100),(2,100),(0,5),(0,1),(2,1000)),
'pis':((2,20),(2,20),(0.5,2),(0.8,1),(100,500)),
'model_extra':{'model':'decay_prior'},
'priorfns':{0:lambda x: halfnorm(0,50).pdf(x),
          1:lambda x: halfnorm(0,50).pdf(x),
          2:lambda x: norm(1,1).pdf(x),
          3:lambda x: norm(0.9,0.2).pdf(x),
          4:lambda x: halfnorm(0,500).pdf(x)},
'priorfns_gen':{0:lambda x: np.abs(t(5,1,10).rvs(x)),
          1:lambda x: np.abs(t(5,1,10).rvs(x)),
          2:lambda x: np.abs(norm(1,0.5).rvs(x)),
          3:lambda x: np.abs(norm(0.9,0.2).rvs(x)),
          4:lambda x: np.abs(t(4,50,200).rvs(x))},
'gen_sets':[list(np.abs(t(5,1,10).rvs(100))),
            list(np.abs(t(5,1,10).rvs(100))),
            list(np.random.uniform(0.25,0.75,50))+list(np.random.uniform(1.25,3,50)),
            list(np.random.uniform(0.65,0.95,100)),
            [new_vs() for i in range(100)]
            ],
}

# Model 13
model_specs['model_bayes_wreport_two_rates_no_decay']={
'modelbasename': 'model_bayes_wreport',
'model': model_bayes_w_report,
'param_names':['alpha0','beta0','wp','wn','v_s'],
'bnds':((2,100),(2,100),(0,10),(0,10),(2,1000)),
'pis':((2,20),(2,20),(0.5,2),(0.5,2),(100,500)),
'model_extra':{'model':'two_rates_no_decay'},
'priorfns':{0:lambda x: halfnorm(0,50).pdf(x),
          1:lambda x: halfnorm(0,50).pdf(x),
          2:lambda x: norm(1,1).pdf(x),
          3:lambda x: norm(1,1).pdf(x),
          4:lambda x: halfnorm(0,500).pdf(x)},
'priorfns_gen':{0:lambda x: np.abs(t(5,1,10).rvs(x)),
          1:lambda x: np.abs(t(5,1,10).rvs(x)),
          2:lambda x: np.abs(norm(1,1).rvs(x)),
          3:lambda x: np.abs(norm(1,1).rvs(x)),
          4:lambda x: np.abs(t(4,50,200).rvs(x))},
'gen_sets':[list(np.abs(t(5,1,10).rvs(100))),
            list(np.abs(t(5,1,10).rvs(100))),
            [0.2,0.4,0.6,0.8,1.2,1.4,1.8,2.2,2.6,3,3.5],
            [0.2,0.4,0.6,0.8,1.2,1.4,1.8,2.2,2.6,3,3.5],
            [new_vs() for i in range(100)]
            ],
}


## SUPPLEMENTAL SURPRISE MODELS ##

# Model 3a
model_specs['model_RW_update_w_report_samelr_relfbscaling_surprise_extraparam']={
'modelbasename': 'model_RW_update_w_report',
'model': model_RW_update_w_report,
'param_names':['alpha','r_pos','v_s','u_u0','surp'],
'bnds':((0.01,0.99),(0,5),(2,1000),(0.01,0.99),(0, 2)),
'pis':((0.1,0.5),(0,2),(100,500),(0.1,0.9),(0,2)),
'model_extra':{'model':'samelr_relfbscaling_surprise_extraparam'},
'priorfns':{0:lambda x: halfnorm(0,0.5).pdf(x),
          1:lambda x: halfnorm(0,2).pdf(x),
          2:lambda x: halfnorm(0,500).pdf(x),
          3:lambda x: norm(0.5,0.5).pdf(x),
          4:lambda x: halfnorm(0,2).pdf(x),
          },
'priorfns_gen':{0:lambda x: np.abs(halfnorm(0,0.25).rvs(x)),
          1:lambda x: np.abs(halfnorm(0,1.5).rvs(x)),
          2:lambda x: np.abs(t(4,50,200).rvs(x)),
          3:lambda x: np.abs(norm(0.5,0.25).rvs(x)),
          4:lambda x: np.abs(halfnorm(0,2).rvs(x))},
'gen_sets':[[0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25],
            [0.2,0.4,0.8,1.2,1.4,1.8,2.2,2.5],
            [new_vs() for i in range(100)],
            list(np.random.uniform(0.1,0.9,100)),
            ],
}

# Model 3b
model_specs['model_RW_update_w_report_samelr_relfbscaling_surprise_binary_extraparam']={
'modelbasename': 'model_RW_update_w_report',
'model': model_RW_update_w_report,
'param_names':['alpha','r_pos','v_s','u_u0','surp'],
'bnds':((0.01,0.99),(0,5),(2,1000),(0.01,0.99),(0, 2)),
'pis':((0.1,0.5),(0,2),(100,500),(0.1,0.9),(0,2)),
'model_extra':{'model':'samelr_relfbscaling_surprise_binary_extraparam'},
'priorfns':{0:lambda x: halfnorm(0,0.5).pdf(x),
          1:lambda x: halfnorm(0,2).pdf(x),
          2:lambda x: halfnorm(0,500).pdf(x),
          3:lambda x: norm(0.5,0.5).pdf(x),
          4:lambda x: halfnorm(0,2).pdf(x),
          },
'priorfns_gen':{0:lambda x: np.abs(halfnorm(0,0.25).rvs(x)),
          1:lambda x: np.abs(halfnorm(0,1.5).rvs(x)),
          2:lambda x: np.abs(t(4,50,200).rvs(x)),
          3:lambda x: np.abs(norm(0.5,0.25).rvs(x)),
          4:lambda x: np.abs(halfnorm(0,2).rvs(x))},
'gen_sets':[[0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25],
            [0.2,0.4,0.8,1.2,1.4,1.8,2.2,2.5],
            [new_vs() for i in range(100)],
            list(np.random.uniform(0.1,0.9,100)),
            ],
}

# Model 3c
model_specs['model_RW_update_w_report_samelr_relfbscaling_relcompfeedback_extraparam']={
'modelbasename': 'model_RW_update_w_report',
'model': model_RW_update_w_report,
'param_names':['alpha','r_pos','v_s','u_u0','f_lambda'],
'bnds':((0.01,0.99),(0,5),(2,1000),(0.01,0.99),(0, 1)),
'pis':((0.1,0.5),(0,2),(100,500),(0.1,0.9),(0,1)),
'model_extra':{'model':'samelr_relfbscaling_relcompfeedback_extraparam'},
'priorfns':{0:lambda x: halfnorm(0,0.5).pdf(x),
          1:lambda x: halfnorm(0,2).pdf(x),
          2:lambda x: halfnorm(0,500).pdf(x),
          3:lambda x: norm(0.5,0.5).pdf(x),
          4:lambda x: halfnorm(0,1).pdf(x),
          },
'priorfns_gen':{0:lambda x: np.abs(halfnorm(0,0.25).rvs(x)),
          1:lambda x: np.abs(halfnorm(0,1.5).rvs(x)),
          2:lambda x: np.abs(t(4,50,200).rvs(x)),
          3:lambda x: np.abs(norm(0.5,0.25).rvs(x)),
          4:lambda x: np.abs(halfnorm(0,1).rvs(x))},
'gen_sets':[[0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25],
            [0.2,0.4,0.8,1.2,1.4,1.8,2.2,2.5],
            [new_vs() for i in range(100)],
            list(np.random.uniform(0.1,0.9,100)),
            ],
}

# Model 3d
model_specs['model_RW_update_w_report_samelr_relfbscaling_relcompfeedback_only']={
'modelbasename': 'model_RW_update_w_report',
'model': model_RW_update_w_report,
'param_names':['alpha','r_pos','v_s','u_u0'],
'bnds':((0.01,0.99),(0,5),(2,1000),(0.01,0.99)),
'pis':((0.1,0.5),(0,2),(100,500),(0.1,0.9)),
'model_extra':{'model':'samelr_relfbscaling_relcompfeedback_only'},
'priorfns':{0:lambda x: halfnorm(0,0.5).pdf(x),
          1:lambda x: halfnorm(0,2).pdf(x),
          2:lambda x: halfnorm(0,500).pdf(x),
          3:lambda x: norm(0.5,0.5).pdf(x),
          4:lambda x: halfnorm(0,1).pdf(x),
          },
'priorfns_gen':{0:lambda x: np.abs(halfnorm(0,0.25).rvs(x)),
          1:lambda x: np.abs(halfnorm(0,1.5).rvs(x)),
          2:lambda x: np.abs(t(4,50,200).rvs(x)),
          3:lambda x: np.abs(norm(0.5,0.25).rvs(x)),
          4:lambda x: np.abs(halfnorm(0,1).rvs(x))},
'gen_sets':[[0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25],
            [0.2,0.4,0.8,1.2,1.4,1.8,2.2,2.5],
            [new_vs() for i in range(100)],
            list(np.random.uniform(0.1,0.9,100)),
            ],
}
