import glob
import pandas as pd
import sys
import numpy as np
import pickle
import json
import argparse

from models import *


def main():
    '''Example of Fitting Model #3:
            python fit_one_model.py --seed 1 --modelname model_RW_update_w_report_samelr_relfbscaling --subjblock all --prior True --save True

    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--modelname', '-m', type=str, default=None)
    parser.add_argument('--subjblock', '-sb', type=str, default=None)
    parser.add_argument('--prior', '-p', type=str, default='False')
    parser.add_argument('--other', '-o', type=str, default='False')
    parser.add_argument('--save', type=str, default='True')

    args = parser.parse_args()
    print(args.modelname)
    print(args.subjblock)
    modelname=args.modelname
    prior = eval(args.prior) # it's a string
    other = eval(args.other)

    ##### Load Participants Datasets ####
    if other==False:
        MIDS = np.load('../data_for_model_fitting/self_MIDS.npy',allow_pickle=True)
        estimates_flipped = np.load('../data_for_model_fitting/self_estimates_flipped.npy')
        feedback = np.load('../data_for_model_fitting/self_feedback.npy')
        surprise = np.load('../data_for_model_fitting/extra_data_for_surprise_models/self_surprise.npy')
        rel_comp = np.load('../data_for_model_fitting/extra_data_for_surprise_models/self_rel_comp.npy')
    elif other==True:
        MIDS = np.load('../data_for_model_fitting/other_MIDS.npy',allow_pickle=True)
        estimates_flipped = np.load('../data_for_model_fitting/other_estimates_flipped.npy')
        feedback = np.load('../data_for_model_fitting/other_feedback.npy')
        surprise = np.load('../data_for_model_fitting/extra_data_for_surprise_models/other_surprise.npy')
        rel_comp = np.load('../data_for_model_fitting/extra_data_for_surprise_models/other_rel_comp.npy')

    sls = {'test':[MIDS[0]],
            'all':MIDS,
    }

    # calculate normalized surprise
    surprise_normalized = (surprise - np.nanmin(surprise))
    surprise_normalized = surprise_normalized/np.nanmax(surprise_normalized)

    # calculate binarized surprise
    surprise_binarized = (surprise > 0).astype('float')

    #####
    # loop through subjects
    ####
    for MID in sls[args.subjblock]:
        print(MID)
        y = estimates_flipped[list(MIDS).index(MID),:]
        data = {'feedback': feedback[list(MIDS).index(MID),:],
                'surprise': surprise_normalized[list(MIDS).index(MID),:],
                'surprise_binary': surprise_binarized[list(MIDS).index(MID),:],
                'rel_comp': rel_comp[list(MIDS).index(MID),:]}
        X = pd.DataFrame(data)


        if prior==True:
            priorfns = model_specs[modelname]['priorfns']
        else:
            priorfns = None

        if 'transform_specs' in model_specs[modelname].keys():
            transform=True
            transform_specs = model_specs[modelname]['transform_specs']
        else:
            transform=False
            transform_specs = ()

        fit = fit_model(X,y,
                model_specs[modelname]['model'],
                modelbasename=model_specs[modelname]['modelbasename'],
                param_names=model_specs[modelname]['param_names'],
                bnds=model_specs[modelname]['bnds'],
                param_inits=model_specs[modelname]['pis'],
                extra_args=model_specs[modelname]['model_extra'],
                prior=prior,
                priorfns=priorfns,
                num_starts=10,
                transform=transform,
                transform_specs=transform_specs)

        if prior==True:
            basefolder='model_fits/individual_fits_w_priors'
        else:
            basefolder='model_fits/individual_fits'

        if other==True:
            basefolder = basefolder+'_others'
        else:
            basefolder = basefolder+'_self'

        # save out parameters
        if eval(args.save):
            pickle.dump(fit, open('../'+basefolder+'/'+MID+'_'+model_specs[modelname]['modelbasename']+'_'+model_specs[modelname]['model_extra']['model']+".p", "wb" ))


if __name__=='__main__':
    main()
