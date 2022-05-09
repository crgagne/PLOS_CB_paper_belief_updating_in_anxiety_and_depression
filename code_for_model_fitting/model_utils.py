
import numpy as np
from scipy.stats import beta

from loading_models import get_multiple_models

def save_out_models_for_BMS(modelnamelist, df, name='full_set'):

    model_fits_df = get_multiple_models(modelnamelist, df)
    model_fits_df_self = model_fits_df.loc[model_fits_df['self_v_others']=='self'].copy()
    model_fits_df_other = model_fits_df.loc[model_fits_df['self_v_others']=='other'].copy()
    BIC_self = model_fits_df_self[['modelname','bic','MID']].pivot(index='MID',columns='modelname')
    BIC_other = model_fits_df_other[['modelname','bic','MID']].pivot(index='MID',columns='modelname')
    BIC_self.to_csv('../model_fits/bics_for_model_comparison/'+name+'_self.csv',index=False)
    BIC_other.to_csv('../model_fits/bics_for_model_comparison/'+name+'_other.csv',index=False)



# Sample new beliefs for all participants
def generate_data_from_model(model_fits_df, df, S=100):

    N=66
    new_reported_beliefs = np.empty((N,S,21))
    new_updates = np.empty((N,S,20))
    new_pos_updates = np.empty((N,S,10))
    new_neg_updates = np.empty((N,S,10))

    feedbacks = np.empty((N,21))
    estimates = np.empty((N,21))
    updates = np.empty((N,20))
    pos_updates = np.empty((N,10))
    neg_updates = np.empty((N,10))

    mu = np.empty((N,10))

    for i,MID in enumerate(model_fits_df.MID):

        # get participants fitted reported distribution parameters for each trial (alphas and betas)
        alpha_s = model_fits_df.loc[model_fits_df.MID==MID,'alpha_s'].values[0]
        beta_s = model_fits_df.loc[model_fits_df.MID==MID,'beta_s'].values[0]

        # generate 100 new belief trajectories by sampling from those reported distribution
        new_reported_beliefs[i,:,:] = beta(alpha_s,beta_s).rvs((S,21))

        # store feedbacks
        feedbacks[i,:]=df.loc[df.MID==MID,'self_feedback'].values[0]
        estimates[i,:]=df.loc[df.MID==MID,'self_estimate_flipped'].values[0]
        updates[i,:]=np.diff(estimates[i,:])
        pos_updates[i,:]=updates[i,feedbacks[i,1:]==1]
        neg_updates[i,:]=updates[i,feedbacks[i,1:]==0]

        # calculate new updates
        new_updates[i,:,:] = np.diff(new_reported_beliefs[i,:,:])
        new_pos_updates[i,:,:] = new_updates[i,:,feedbacks[i,1:]==1].T
        new_neg_updates[i,:,:] = new_updates[i,:,feedbacks[i,1:]==0].T

    return(new_reported_beliefs,new_updates,new_pos_updates,new_neg_updates,
           feedbacks,estimates,updates,pos_updates,neg_updates)
