import pickle
import numpy as np
import pandas as pd

def get_model_fit_data(modelname, df, selfother='self', modelfits_folder='../model_fits/individual_fits_w_priors', MIDs=None):

    model_fits = {}
    stats = ['r2','mse','rmse','aic','bic','nllk','alpha_s','beta_s']
    for stat in stats:
        model_fits[stat]=[]
    model_fits['MID']=[]
    model_fits['modelname']=[]
    model_fits['u_last']=[]
    model_fits['starting_belief']=[]
    model_fits['ending_belief']=[]
    model_fits['self_feedback_cb']=[]
    model_fits['session3_cb']=[]
    model_fits['u_avg_abs_diff']=[]

    if selfother=='self':
        prefix2='self'
        suffix1 = '_self'
    else:
        prefix2='other'
        suffix1 = '_others'

    if MIDs == None:
        MIDs = df.MID

    for MID in MIDs:

        # get model fit
        fit = pickle.load( open(modelfits_folder+suffix1+\
                        '/'+MID+'_'+modelname+".p", "rb" ),encoding='latin1')

        # add stats
        for stat in stats:
            model_fits[stat].append(np.round(fit[stat],3))

        # add params and transformations
        params = fit['params']
        if 'alpha0' in params:
            params['alpha0+beta0']=params['alpha0']+params['beta0']
            params['alpha0/alpha0+beta0']=params['alpha0']/(params['alpha0']+params['beta0'])
            params['logalpha0']=np.log(params['alpha0'])
            params['logbeta0']=np.log(params['beta0'])
            params['log(alpha0+beta0)']=np.log(params['alpha0']+params['beta0'])
        if 'wp' in params:
            params['wp+wn']=params['wp']+params['wn']
            params['wp-wn']=params['wp']-params['wn']
        if 'update_pos' in params:
            params['update_pos+update_neg']=params['update_pos']+params['update_neg']
            params['update_pos-update_neg']=params['update_pos']-params['update_neg']
            params['update_pos/update_neg']=params['update_pos']/params['update_neg']
        if 'alpha_pos' in params and 'alpha_neg' in params:
            params['alpha_pos/alpha_neg']=params['alpha_pos']/params['alpha_neg']
            params['alpha_pos+alpha_neg']=params['alpha_pos']+params['alpha_neg']
            params['alpha_pos-alpha_neg']=params['alpha_pos']-params['alpha_neg']
        if 'r_pos' in params and 'r_neg' in params:
            params['r_pos-|r_neg|']=params['r_pos']-np.abs(params['r_neg'])
            params['r_pos-r_neg']=params['r_pos']-params['r_neg']

        for param in fit['params'].keys():
            if param not in model_fits:
                model_fits[param]=[]
            model_fits[param].append(fit['params'][param])

        # add last belief predicted by the model
        model_fits['u_last'].append(fit['u_s'][-1])
        model_fits['u_avg_abs_diff'].append(np.mean(np.abs(np.diff(fit['u_s']))))

        # add other stuff
        model_fits['MID'].append(MID)
        model_fits['modelname'].append(modelname)
        model_fits['starting_belief'].append(df.loc[df.MID==MID,'starting_beliefs_self'].values[0])
        model_fits['ending_belief'].append(df.loc[df.MID==MID,'self_estimate_flipped'].values[0][-1])
        model_fits['self_feedback_cb'].append(df.loc[df.MID==MID,'self_feedback_cb'].values[0])
        model_fits['session3_cb'].append(df.loc[df.MID==MID,'session3_cb'].values[0])

    model_fits_df = pd.DataFrame(model_fits)

    traits1= [
    'stai_state_sess3','stai_trait_anx_sess2',
                  'stai_trait_anx_sess3','stai_trait_dep_sess2','stai_trait_dep_sess3',
                  'cesd_dep_sess2', 'cesd_dep_sess3',
                  'cesd_anh_sess2','cesd_anh_sess3',
                  'cesd_som_sess2','cesd_som_sess3',
                 'masq_aa','masq_ad','masq_as','masq_ds','pswq']
    traits2 =  ['item_clinical.fa.omega3.anh','item_clinical.fa.omega3.g','item_clinical.fa.omega3.cog_anx']

    for trait in traits1+traits2:
        for MID in df['MID']:
            model_fits_df.loc[(model_fits_df.MID==MID),trait]=df.loc[df.MID==MID,trait].values
    return(model_fits_df,params)


def get_multiple_models(modelnamelist, df):
    model_fits = {}
    stats = ['r2','mse','rmse','aic','bic','nllk']
    for stat in stats:
        model_fits[stat]=[]
    model_fits['MID']=[]
    model_fits['modelname']=[]
    model_fits['starting_belief']=[]
    model_fits['ending_belief'] = []
    model_fits['self_v_others'] = []
    for MID in df.MID:

        for selfother in ['self','other']:

            if selfother=='self':
                prefix2='self'
                suffix1 = '_self'
            else:
                prefix2='other'
                suffix1 = '_others'

            # get params and plot
            for modelname in modelnamelist:
                fit = pickle.load( open('../model_fits/individual_fits_w_priors'+suffix1+\
                                        '/'+MID+'_'+modelname+".p", "rb" ),encoding='latin1')
                for stat in stats:
                    model_fits[stat].append(np.round(fit[stat],3))

                model_fits['MID'].append(MID)
                model_fits['modelname'].append(modelname)

                # adding in other potentially useful things
                model_fits['starting_belief'].append(df.loc[df['MID']==MID,prefix2+'_estimate_flipped'].values[0][0])
                model_fits['ending_belief'].append(df.loc[df['MID']==MID,prefix2+'_estimate_flipped'].values[0][-1])

                model_fits['self_v_others'].append(selfother)
    model_fits_df = pd.DataFrame(model_fits)
    return(model_fits_df)
