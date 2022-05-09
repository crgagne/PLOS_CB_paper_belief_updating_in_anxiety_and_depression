renaming = {'em0001':'8243',
            'em0002':'1284',
            'em0003':'2478',
            'em0004':'8116',
            'em0006':'3827',
            'em0010':'3910',
            'emr0001':'8346',
            'emr0002':'8345',
            'emr0003':'8344',
            'emr0004':'8347'}

name_replace_clinical = {
    'item_clinical.fa.omega3.g':'general factor',
    'item_clinical.fa.omega3.anh':'depression-specific factor',
    'item_clinical.fa.omega3.cog_anx':'anxiety-specific factor',
    'stai_state_sess3':'STAI state (sess3)',
    'stai_trait_anx_sess2': 'STAI anxiety (sess2)',
    'stai_trait_anx_sess3': 'STAI anxiety (sess3)',
    'stai_trait_dep_sess2': 'STAI depression (sess2)',
    'stai_trait_dep_sess3': 'STAI depression (sess3)',
    'cesd_dep_sess2':'CESD depression (sess2)',
    'cesd_dep_sess3':'CESD depression (sess3)',
    'cesd_anh_sess2':'CESD anhedonia (sess2)',
    'cesd_anh_sess3':'CESD anhedonia (sess3)',
    'cesd_som_sess2':'CESD somatic (sess2)',
    'cesd_som_sess3':'CESD somatic (sess3)',
    'masq_aa':'MASQ anxious arousal (sess2)',
    'masq_ad':'MASQ anhedonia (sess2)',
    'masq_as':'MASQ anxiety general (sess2)',
    'masq_ds':'MASQ depression general (sess2)',
    'pswq':'PSWQ worry (sess1)',
}

name_replace_stat = {
    'r_pos':'$b$',
    'u_u0':r'$\mu_0$',
    'alpha':r'$\eta$',
    'v_s':r'$\nu$',
    'r2':r'R-squared',
    'aic':r'AIC',
    'bic':r'BIC',
    'nllk':'negative loglik',
    'surp':'surp',
    'alpha_pos':r'$\eta_+$',
    'alpha_pos-alpha_neg':r'$\eta_+ - \eta_-$',
    'alpha_pos\alpha_neg':r'$\eta_+ / \eta_-$',
    'alpha_neg':r'$\eta_-$',
    'alpha0/alpha0+beta0':r'$alpha0 / alpha0+beta0$',
    'w':r'$\omega$',
    'f_lambda': '$\lambda_f$'
}

name_replace_modelsA = {
    # base models
    'model_bayes_wreport_asym_no_reported':'biased Bayesian (Model 1)',
    'model_bayes_wreport_basic_no_reported':'Bayesian (Model 2)',
    'model_RW_update_w_report_samelr_relfbscaling':'biased RW (Model 3)',
    'model_RW_update_w_report_samelr_nofbscaling':'RW (Model 4)',
}

name_replace_modelsB = {

    # base models
    'model_bayes_wreport_asym_no_reported':'Model 1',
    'model_bayes_wreport_basic_no_reported':'Model 2',
    'model_RW_update_w_report_samelr_relfbscaling':'Model 3',
    'model_RW_update_w_report_samelr_nofbscaling':'Model 4',

    # supplemental RW
    'model_RW_update_w_report_difflr_nofbscaling':'Model 5',
    'model_RW_update_w_report_samelr_fbscaling':'Model 6',

    # supplemental Bayes
    'model_bayes_wreport_w_no_reported_decay_50':'Model 7',
    'model_bayes_wreport_w_no_reported_decay_prior':'Model 8',
    'model_bayes_wreport_basic':'Model 9',
    'model_bayes_wreport_no_decay':'Model 10',
    'model_bayes_wreport_decay_50':'Model 11',
    'model_bayes_wreport_decay_prior':'Model 12',
    'model_bayes_wreport_two_rates_no_decay': 'Model 13',

    # supplemental surprise models
    'model_RW_update_w_report_samelr_relfbscaling_surprise_extraparam': 'Model 3a',
    'model_RW_update_w_report_samelr_relfbscaling_surprise_binary_extraparam': 'Model 3b',
    'model_RW_update_w_report_samelr_relfbscaling_relcompfeedback_extraparam': 'Model 3c',
    'model_RW_update_w_report_samelr_relfbscaling_relcompfeedback_only': 'Model 3d'

}
