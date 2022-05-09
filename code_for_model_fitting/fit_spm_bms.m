clear all;

addpath('~/Documents/spm12/')

% specify the aic's or bic's
input_file = 'full_set_self.csv';
output_file = 'full_set_self_results.csv';

lme_table = readtable(['../model_fits/bics_for_model_comparison/',input_file],'HeaderLines',1);
lme_matrix = table2array(lme_table)*-1;

%keyboard;

modelnames = lme_table.Properties.VariableNames;


[alpha,exp_r,xp,pxp,bor] = spm_BMS(lme_matrix);

% ploy
%figure; bar(exp_r);
figure; bar(pxp);
set(gca,'XTickLabel',modelnames,'XTick',1:numel(modelnames))
alpha = alpha';
exp_r = exp_r';
xp = xp';
pxp=pxp';
bor=bor;
display('bayesian omnibus risk')
display(bor)
results = table(alpha,exp_r,xp,pxp);
%writetable(results,['../model_fits/bics_for_model_comparison/',output_file])

% family wise exceedance approach
% sum the alpha's I think.
%- xp = spm_dirichlet_exceedance(alpha,Nsamp);
