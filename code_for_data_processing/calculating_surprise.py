import pandas as pd
import numpy as np

from participants import get_all_participants_df

def calculate_percent_selected_dict():
    df_everyone = pd.read_csv('../data_processed/data_everyone_from_session2.csv')
    selected_ev = []
    unselected_ev = []

    # get two lists of MIDs for all (selected) and (unselected) profiles
    for i in range(df_everyone.shape[0]):
        sel = df_everyone.loc[i,'selected']
        unsel =df_everyone.loc[i,'unselected']
        if type(sel)==str:
            selected_ev.extend(eval(sel))
            unselected_ev.extend(eval(unsel))
    selected_ev = np.array(selected_ev)
    unselected_ev = np.array(unselected_ev)

    # for each MID in those lists, calculate percent of times chosen
    all_mids = list(set(list(selected_ev)+list(unselected_ev)))
    percent_selected_dict = {}
    for mid in all_mids:
        percent_selected = (selected_ev==mid).sum() / ((selected_ev==mid).sum()+(unselected_ev==mid).sum())
        percent_selected_dict[mid]=percent_selected

    return(percent_selected_dict)


def saving_out_surprise(self_other='self'):

    _ , df=get_all_participants_df(exclude=True)

    percent_selected_dict = calculate_percent_selected_dict()

    # get data that I fit models to
    MIDS = np.load('../data_processed/'+self_other+'_MIDS.npy',allow_pickle=True)
    estimates_flipped = np.load('../data_processed/'+self_other+'_estimates_flipped.npy')
    feedback = np.load('../data_processed/'+self_other+'_feedback.npy')

    fb_pair0_list = []
    fb_pair1_list = []
    fb_pair0_percent_chosen_list = []
    fb_pair1_percent_chosen_list = []
    surprise_list = []
    rel_comp_list = []

    for i, MID in enumerate(MIDS):


        # get feedback sequence per participants
        fb = df.loc[df.MID==MID][self_other+'_feedback'].values[0][1::]
        fb_pair0 = df.loc[df.MID==MID][self_other+'_feedback_pair0'].values[0][1::]
        fb_pair1 = df.loc[df.MID==MID][self_other+'_feedback_pair1'].values[0][1::]
        fb_pair0_list.append(fb_pair0)
        fb_pair1_list.append(fb_pair1)
        assert np.all(fb==feedback[i][1::]) # check that this dataset matches saved one
        assert len(np.unique(fb_pair0))==1 # check that fb_pair0 is the person themselves

        # for each pair in the sequence
        diff_percent_chosen_unchosen = []
        fb_pair0_percent_chosen = []
        fb_pair1_percent_chosen = []
        surprise = []
        rel_comp = []
        p0_orig = fb_pair0[0]
        for p0, p1, sel in zip(fb_pair0, fb_pair1, fb):
            assert p0_orig ==p0

            # get % times chosen
            p0_percent = percent_selected_dict[p0]
            p1_percent = percent_selected_dict[p1]
            fb_pair0_percent_chosen.append(p0_percent)
            fb_pair1_percent_chosen.append(p1_percent)

            # calculate difference between unselected - selected (this will be negative surprise)
            if sel==0:
                diff = p0_percent-p1_percent
            elif sel==1:
                diff = p1_percent-p0_percent

            # store
            diff_percent_chosen_unchosen.append(diff)
            surprise.append(-1*diff)
            rel_comp.append(p0_percent>p1_percent) # self > other

        fb_pair0_percent_chosen_list.append(fb_pair0_percent_chosen)
        fb_pair1_percent_chosen_list.append(fb_pair1_percent_chosen)
        surprise_list.append(surprise)
        rel_comp_list.append(rel_comp)

    fb_pair0 = np.array(fb_pair0_list)
    fb_pair1 = np.array(fb_pair1_list)
    fb_pair0_percent_chosen = np.array(fb_pair0_percent_chosen_list)
    fb_pair1_percent_chosen = np.array(fb_pair1_percent_chosen_list)
    surprise = np.array(surprise_list)
    rel_comp = np.array(rel_comp_list)

    # add column of nans
    nan_column = np.ones(fb_pair0.shape[0])[:,np.newaxis]*np.nan
    fb_pair0 = np.hstack((nan_column, fb_pair0))
    fb_pair1 = np.hstack((nan_column, fb_pair1))
    fb_pair0_percent_chosen = np.hstack((nan_column, fb_pair0_percent_chosen))
    fb_pair1_percent_chosen = np.hstack((nan_column, fb_pair1_percent_chosen))
    surprise = np.hstack((nan_column, surprise))
    rel_comp = np.hstack((nan_column, rel_comp))

    np.save('../data_processed/'+self_other+'_fb_pair0.npy', fb_pair0)
    np.save('../data_processed/'+self_other+'_fb_pair1.npy', fb_pair1)
    np.save('../data_processed/'+self_other+'_fb_pair0_percent_chosen.npy', fb_pair0_percent_chosen)
    np.save('../data_processed/'+self_other+'_fb_pair1_percent_chosen.npy', fb_pair1_percent_chosen)
    np.save('../data_processed/'+self_other+'_surprise.npy', surprise)
    np.save('../data_processed/'+self_other+'_rel_comp.npy', rel_comp)
