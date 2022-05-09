
import json
import pandas as pd
import sys
import re
import numpy as np
import os
import glob
from sklearn.preprocessing import scale

sys.path.append('../functions/')

import questionnaires.pswq as q_pswq
import questionnaires.stai_state as q_stai_state
import questionnaires.stai_trait as q_stai_trait
import questionnaires.masq as q_masq
import questionnaires.cesd as q_cesd

import renaming

def logit(x):
    return(np.log((x/(1-x))))
def invlogit(x):
    return(1.0/(1.0+np.exp(-1.0*x)))


class Participant:

    def __init__(self,code_id,batch,threshold=0.85,ex_count = True,
        dirr1='../data/session1/',
        dirr2='../data/session2/',
        dirr3='../data/session3/'):

        self.code_id = code_id
        self.code_id2 = code_id
        self.SESSION_1_DATA_FILE = dirr1
        self.SESSION_2_DATA_FILE = dirr2
        self.SESSION_3_DATA_FILE = dirr3
        self.batch=batch

        # loading session2 data doesn't have em's
        if 'fall' not in self.SESSION_1_DATA_FILE:
            if 'em' in self.code_id:
                self.code_id2 = renaming.renaming[self.code_id]

        # if passing a session 2-3 id that don't have em's..
        if self.code_id in renaming.renaming.values():
            self.code_id = list(renaming.renaming.keys())[list(renaming.renaming.values()).index(self.code_id)]

        self.threshold = threshold # % similar responses (ie. >85% the same answer)
        self.ex_count = ex_count # whether to use % similar respnses in addition to catch questions ...

        # session 1 stuff
        if os.path.exists(self.SESSION_1_DATA_FILE+self.code_id):
            self._load_sats_and_grades()
            self._load_pswq()
            self.all_pass_q = self.pswq_pass
            self._load_consents()
        else:
            #pass
            print('session1 doesnt exist for '+self.code_id)

        # session 2 stuff
        try:
            self._load_stai_state()
            self._load_stai_trait()
            self._load_masq()
            self._load_cesd()
            self._load_sess2_responses()
            self._load_sess2_event_durs()
            self._load_sess2_how_rated()
        except:
            #pass
            print('session2 doesnt exist for '+self.code_id)
            import pdb; pdb.set_trace()

        # session 3 stuff
        try:
            self._load_session3_estimates()
            self._load_stai_state_sess3()
            self._load_stai_trait_sess3()
            self._load_cesd_sess3()
            self._load_sess3_how_rated()
        except:
            pass
            print('session3 doesnt exist for '+self.code_id)


    def _load_sats_and_grades(self):

        with open(self.SESSION_1_DATA_FILE +self.code_id+'/sat_grades_entry_responses.json') as scores:
            s = json.load(scores)
            self.sat = [int(s['Q0']), int(s['Q1']), int(s['Q2'])]
            self.grades = [s['Q3'], s['Q4'], s['Q5']]

    def _load_pswq(self):
        with open (self.SESSION_1_DATA_FILE +self.code_id+'/survey_pswq_responses.json') as pswq_responses:
            p = json.load(pswq_responses)
            raw_scores = [p['Q'+str(q)] for q in range(len(p))]
            scores = [''.join(r.split()) for r in raw_scores]
            self.pswq_a = [int(re.sub("[^0-9]", "", x)) for x in scores]

        self.pswq, self.pswq_pass,self.pswq_forward, self.pswq_reverse = q_pswq.PSWQ_score(self.pswq_a,self.threshold,True)

    def _load_stai_state(self):
        if 'round2' in self.SESSION_2_DATA_FILE or 'round3' in self.SESSION_2_DATA_FILE:
            filee='/survey-multi-choicesurvey_stai_state_responses0.json'
        else:
            filee='/survey-multi-choicesurvey_stai_state_responses.json'

        with open (self.SESSION_2_DATA_FILE +self.code_id2+filee) as responses:
            p = json.load(responses)
            raw_scores = [p['Q'+str(q)] for q in range(len(p))]
            scores = [r.replace("<br>", "").replace("</br>", "").replace(" ", "") for r in raw_scores]
            self.stai_state_a = [q_stai_state.choices_dict[x] for x in scores]
            self.stai_state, self.stai_state_pass,self.stai_state_forward, self.stai_state_reverse = q_stai_state.STAI_state_score(self.stai_state_a,self.threshold,self.ex_count)

        df_e = pd.read_csv(self.SESSION_2_DATA_FILE +self.code_id2+'/event_dict.csv')

    def _load_stai_trait(self):
        if 'round2' in self.SESSION_2_DATA_FILE or 'round3' in self.SESSION_2_DATA_FILE:
            filee='/survey-multi-choicesurvey_stai_trait_responses0.json'
        else:
            filee='/survey-multi-choicesurvey_stai_trait_responses.json'
        with open (self.SESSION_2_DATA_FILE +self.code_id2+filee) as responses:
            p = json.load(responses)
            raw_scores = [p['Q'+str(q)] for q in range(len(p))]
            scores = [r.replace("<br>", "").replace("</br>", "").replace(" ", "") for r in raw_scores]
            self.stai_trait_a = [q_stai_trait.choices_dict[x] for x in scores]
            self.stai_trait, self.stai_trait_pass, self.stai_trait_forward, self.stai_trait_reverse,self.stai_trait_anx,self.stai_trait_dep = q_stai_trait.STAI_trait_score(self.stai_trait_a,self.threshold,self.ex_count)

    def _load_masq(self):
        if 'round2' in self.SESSION_2_DATA_FILE or 'round3' in self.SESSION_2_DATA_FILE:
            filee0='/survey-multi-choicesurvey_masq0_responses0.json'
            filee1='/survey-multi-choicesurvey_masq1_responses0.json'
        else:
            filee0='/survey-multi-choicesurvey_masq0_responses.json'
            filee1='/survey-multi-choicesurvey_masq1_responses.json'
        with open (self.SESSION_2_DATA_FILE +self.code_id2+filee0) as responses:
            p0 = json.load(responses)

        with open (self.SESSION_2_DATA_FILE +self.code_id2+filee1) as responses:
            p1 = json.load(responses)

        # change the second half's names
        p1new = {}
        for i in range(len(p1)):
            p1new['Q'+str(len(p0)+i)]=p1['Q'+str(i)]

        p=dict([item for item in p0.items()] + [item for item in p1new.items()])

        raw_scores = [p['Q'+str(q)] for q in range(len(p))]
        scores = [r.replace("<br>", "").replace("</br>", "").replace(" ", "") for r in raw_scores]
        self.masq_a = [q_masq.choices_dict[x] for x in scores]
        self.masq_aa,self.masq_ad,self.masq_ds,self.masq_as,self.masq_ms, self.masq_pass,self.masq_pass_catch0,self.masq_pass_catch1,self.masq_a_wo_catch = q_masq.MASQ_score(self.masq_a,self.threshold,self.ex_count)

    def _load_cesd(self):
        if 'round2' in self.SESSION_2_DATA_FILE or 'round3' in self.SESSION_2_DATA_FILE:
            filee='/survey-multi-choicesurvey_cesd_responses0.json'
        else:
            filee='/survey-multi-choicesurvey_cesd_responses.json'

        with open (self.SESSION_2_DATA_FILE +self.code_id2+filee) as responses:
            p = json.load(responses)
            raw_scores = [p['Q'+str(q)] for q in range(len(p))]
            scores = [r.replace("<br>", "").replace("</br>", "").replace(" ", "") for r in raw_scores]
            self.cesd_a = [q_cesd.choices_dict[x] for x in scores]
            self.cesd, self.cesd_pass,self.cesd_forward, self.cesd_reverse,self.cesd_dep_sess2,self.cesd_anh_sess2,self.cesd_som_sess2 = q_cesd.CESD_score(self.cesd_a,self.threshold,self.ex_count)

    def _load_consents(self):

        # session 1 consents
        with open((self.SESSION_1_DATA_FILE +self.code_id+'/instructions_backstory_tags.txt')) as f:
            consents_p1 = f.readlines()[0].split(',') # list ['I agree','I disagree','Iagree']

        with open((self.SESSION_1_DATA_FILE +self.code_id+'/consent_part1_2b_tags.txt')) as f:
            consents_p2b = f.readlines()

        with open((self.SESSION_1_DATA_FILE +self.code_id+'/consent_part1_2_tags.txt')) as f:
            consents_p2 = f.readlines()

        self.consent1=consents_p1[0] # consent to participate
        self.consent2=consents_p1[1] # agree for key to be linked after study is complete
        self.consent3=consents_p1[2] # agree to be contacted
        self.consent4=consents_p2[0] # my answers to be provided for next session
        self.consent5=consents_p2b[0] # I would like to participate in parts 2-3


    def _load_sess2_event_durs(self):

        df_e = pd.read_csv(self.SESSION_2_DATA_FILE +self.code_id2+'/event_dict.csv')

        # opening instructions
        instruction_acum_start = df_e.loc[df_e['event_names']=='NaN0','event_accum_times'].values[0]
        instruction_acum_end = df_e.loc[df_e['event_types']=='instructions19','event_accum_times'].values[0]
        self.dur_openning_instructions = instruction_acum_end-instruction_acum_start

        # choosing pair_responses
        first_decision = df_e.loc[df_e['event_names']=='choosing_resp'].iloc[0]['event_accum_times']
        last_decision = df_e.loc[df_e['event_names']=='choosing_resp'].iloc[-1]['event_accum_times']
        self.dur_choosing = last_decision-first_decision

        # questionnaire instructions (end time minus end time of previous event)
        i = df_e.loc[df_e['event_names']=='survey_stai_state_instr0'].index
        self.dur_stai_state_instruct = df_e.iloc[i]['event_accum_times'].values[0]-df_e.iloc[i-1]['event_accum_times'].values[0]

        i = df_e.loc[df_e['event_names']=='survey_stai_trait_instr0'].index
        self.dur_stai_trait_instruct = df_e.iloc[i]['event_accum_times'].values[0]-df_e.iloc[i-1]['event_accum_times'].values[0]

        i = df_e.loc[df_e['event_names']=='survey_cesd_instr0'].index
        self.dur_cesd_instruct = df_e.iloc[i]['event_accum_times'].values[0]-df_e.iloc[i-1]['event_accum_times'].values[0]

        i = df_e.loc[df_e['event_names']=='survey_masq_instr0'].index
        self.dur_masq_instruct = df_e.iloc[i]['event_accum_times'].values[0]-df_e.iloc[i-1]['event_accum_times'].values[0]

        # filling out questionnaires
        self.dur_stai_state = df_e.loc[df_e['event_names']=='survey_stai_state','event_durs'].values[0]
        self.dur_stai_trait = df_e.loc[df_e['event_names']=='survey_stai_trait','event_durs'].values[0]
        self.dur_cesd = df_e.loc[df_e['event_names']=='survey_cesd','event_durs'].values[0]
        self.dur_masq0 = df_e.loc[df_e['event_names']=='survey_masq0','event_durs'].values[0]
        self.dur_masq1 = df_e.loc[df_e['event_names']=='survey_masq1','event_durs'].values[0]
        self.dur_masq=self.dur_masq0+self.dur_masq1
        self.dur_survey_on_what_used = df_e.loc[(df_e['event_types']=='survey-likert'),'event_durs'].values[0]

    def _load_sess2_responses(self):

        df = pd.read_csv(self.SESSION_2_DATA_FILE +self.code_id2+'/pair_responses.csv')
        df_e = pd.read_csv(self.SESSION_2_DATA_FILE +self.code_id2+'/event_dict.csv')

        # selected, unselected, number of flip flops, and duration of choice.
        selected = []
        unselected = []
        self.choice_durs = list(df_e.loc[df_e['event_types']=='alt-choice-gagne'].event_durs.values)
        self.choice_durs = [int(n) for n in self.choice_durs]
        assert len(self.choice_durs)>=24

        self.number_flipflops = [len(eval(l)) for l in df.button_pressed_list.values]

        for i,press in enumerate(df.last_button_pressed.values):
            if press==1:
                selected.append(df.pair0.values[i])
                unselected.append(df.pair1.values[i])
            elif press==2:
                selected.append(df.pair1.values[i])
                unselected.append(df.pair0.values[i])

        self.selected=selected
        self.unselected=unselected

    def _load_sess2_how_rated(self):
        if 'round2' in self.SESSION_2_DATA_FILE or 'round3' in self.SESSION_2_DATA_FILE:
            filee='/survey-likertNaN_responses0.json'
        else:
            filee='/survey-likertNaN_responses.json'

        with open (self.SESSION_2_DATA_FILE +self.code_id2+filee) as responses:
            p = json.load(responses)
            p['whether group vs. independent worker']=p.pop('Q0')
            p['whether analytic vs. intuitive thinker']=p.pop('Q1')
            p['whether strategic vs. adaptive planner']=p.pop('Q2')
            p["the personal description"]=p.pop('Q3')
            p["SAT scores"]=p.pop('Q4')
            p["grades"]=p.pop('Q5')
        self.considered_during_rating=p
        self.considered_gi=p['whether group vs. independent worker']
        self.considered_ai=p['whether analytic vs. intuitive thinker']
        self.considered_sa=p['whether strategic vs. adaptive planner']
        self.considered_persd=p["the personal description"]
        self.considered_sat=p["SAT scores"]
        self.considered_grade=p["grades"]

    def _load_sess2_feedback(self):
        pass

        # get a list of all other subjects that have rated me...
        # this I might tweak, if we want to exclude feedback ##
        #MIDs_w_session2_data = [f.split('/')[-1] for f in glob.glob('../data/session2/*')]

        # instantiate those participants
        #for idd in MIDs_w_session2_data:
        #    p = participants.Participant(idd)

    def _load_session3_estimates(self):
        df = pd.read_csv(self.SESSION_3_DATA_FILE+self.code_id2+'/self_estimates.csv',index_col=0)
        self.session3_cb = df['cb'].values[0]
        self.session3_truth = df['truth'].values[0]
        self.self_estimate = df['estimate'].values/100.0
        self.self_feedback = df['feedback'].values
        self.self_feedback_pair0 = df['pair0'].values
        self.self_feedback_pair1 = df['pair1'].values
        self.self_rt = df['rt'].values

        df = pd.read_csv(self.SESSION_3_DATA_FILE+self.code_id2+'/other_estimates.csv',index_col=0)
        self.other_estimate = df['estimate'].values/100.0
        self.other_feedback = df['feedback'].values
        self.other_feedback_pair0 = df['pair0'].values
        self.other_feedback_pair1 = df['pair1'].values
        self.other_rt = df['rt'].values

        # flip Feedback
        if self.session3_cb==0:
            self.self_estimate_flipped = 1.0-self.self_estimate
            self.other_estimate_flipped = 1.0-self.other_estimate
        elif self.session3_cb==1:
            self.self_estimate_flipped = self.self_estimate
            self.other_estimate_flipped = self.other_estimate

        # logit transformed
        self.self_estimate_flipped_logit = np.log((self.self_estimate_flipped/(1.0-self.self_estimate_flipped)))
        self.other_estimate_flipped_logit = np.log((self.other_estimate_flipped/(1.0-self.other_estimate_flipped)))


    def _load_stai_state_sess3(self):

        with open (self.SESSION_3_DATA_FILE +self.code_id2+'/survey-multi-choicesurvey_stai_state_responses0.json') as responses:
            p = json.load(responses)
            raw_scores = [p['Q'+str(q)] for q in range(len(p))]
            scores = [r.replace("<br>", "").replace("</br>", "").replace(" ", "") for r in raw_scores]
            self.stai_state_a_sess3 = [q_stai_state.choices_dict[x] for x in scores]
            self.stai_state_sess3, self.stai_state_pass_sess3,self.stai_state_forward_sess3, self.stai_state_reverse_sess3 = q_stai_state.STAI_state_score(self.stai_state_a_sess3,self.threshold,self.ex_count)

    def _load_stai_trait_sess3(self):

        with open (self.SESSION_3_DATA_FILE +self.code_id2+'/survey-multi-choicesurvey_stai_trait_responses0.json') as responses:
            p = json.load(responses)
            raw_scores = [p['Q'+str(q)] for q in range(len(p))]
            scores = [r.replace("<br>", "").replace("</br>", "").replace(" ", "") for r in raw_scores]
            self.stai_trait_a_sess3 = [q_stai_trait.choices_dict[x] for x in scores]
            self.stai_trait_sess3, self.stai_trait_pass_sess3, self.stai_trait_forward_sess3, self.stai_trait_reverse_sess3,self.stai_trait_anx_sess3,self.stai_trait_dep_sess3 = q_stai_trait.STAI_trait_score(self.stai_trait_a_sess3,self.threshold,self.ex_count)

    def _load_cesd_sess3(self):
        with open (self.SESSION_3_DATA_FILE +self.code_id2+'/survey-multi-choicesurvey_cesd_responses0.json') as responses:
            p = json.load(responses)
            raw_scores = [p['Q'+str(q)] for q in range(len(p))]
            scores = [r.replace("<br>", "").replace("</br>", "").replace(" ", "") for r in raw_scores]
            self.cesd_a_sess3 = [q_cesd.choices_dict[x] for x in scores]
            self.cesd_sess3, self.cesd_pass_sess3,self.cesd_forward_sess3, self.cesd_reverse_sess3,self.cesd_dep_sess3,self.cesd_anh_sess3,self.cesd_som_sess3 = q_cesd.CESD_score(self.cesd_a_sess3,self.threshold,self.ex_count)

    def _load_sess3_how_rated(self):
        with open (self.SESSION_3_DATA_FILE +self.code_id2+'/survey-likertNaN_responses0.json') as responses:
            p = json.load(responses)
        self.sess3_self_considered_total_times=p['Q0']
        self.sess3_self_considered_better_grades=p['Q1']
        self.sess3_self_considered_worse_grades=p['Q2']
        self.sess3_self_considered_better_sats=p['Q3']
        self.sess3_self_considered_worse_sats=p['Q4']

        with open (self.SESSION_3_DATA_FILE +self.code_id2+'/survey-likertNaN_responses1.json') as responses:
            p = json.load(responses)
        self.sess3_other_considered_total_times=p['Q0']
        self.sess3_other_considered_better_grades=p['Q1']
        self.sess3_other_considered_worse_grades=p['Q2']
        self.sess3_other_considered_better_sats=p['Q3']
        self.sess3_other_considered_worse_sats=p['Q4']

        with open (self.SESSION_3_DATA_FILE +self.code_id2+'/survey-multi-choiceNaN_responses0.json') as responses:
            p = json.load(responses)
        self.sess3_wanted_to_see_what_people_used = p['Q0']

        with open (self.SESSION_3_DATA_FILE +self.code_id2+'/survey-multi-choiceNaN_responses1.json') as responses:
            p = json.load(responses)
        self.sess3_wanted_to_see_truth = p['Q0']



def get_all_participants_df(exclude=True):

    # spring 2018
    MIDs_w_session3_data_spring = [f.split('/')[-1] for f in glob.glob('../data/session3_round1/*')]

    # fall 2018
    MIDs_w_session3_data_fall = [f.split('/')[-1] for f in glob.glob('../data/session3_round2/*')]

    # spring 2019
    MIDs_w_session3_data_spring19 = [f.split('/')[-1] for f in glob.glob('../data/session3_round3/*')]

    # combine data
    MIDs_w_session3_data=MIDs_w_session3_data_spring+MIDs_w_session3_data_fall+MIDs_w_session3_data_spring19

    ## Exclude ##
    if exclude:
        MIDs_w_session3_data.remove('1284')
        MIDs_w_session3_data.remove('1746')
        MIDs_w_session3_data.remove('7668')
        MIDs_w_session3_data.remove('8322')
        MIDs_w_session3_data.remove('8116')
        MIDs_w_session3_data.remove('8243')
        MIDs_w_session3_data.remove('7558')
        MIDs_w_session3_data.remove('8407')
        MIDs_w_session3_data.remove('7665')


    all_Part_dict = []

    for MID in MIDs_w_session3_data:

        if MID in MIDs_w_session3_data_fall:
            SESSION_1_DATA_FILE = '../data/session1_round2/'
            SESSION_2_DATA_FILE = '../data/session2_round2/'
            SESSION_3_DATA_FILE = '../data/session3_round2/'
            term='fall18'
        elif MID in MIDs_w_session3_data_spring19:
            SESSION_1_DATA_FILE = '../data/session1_round3/'
            SESSION_2_DATA_FILE = '../data/session2_round3/'
            SESSION_3_DATA_FILE = '../data/session3_round3/'
            term='spring19'
        else:
            SESSION_1_DATA_FILE = '../data/session1_round1/'
            SESSION_2_DATA_FILE = '../data/session2_round1/'
            SESSION_3_DATA_FILE = '../data/session3_round1/'
            term='spring18'

        # get data
        P = Participant(MID,term,dirr1=SESSION_1_DATA_FILE,
                                         dirr2=SESSION_2_DATA_FILE,
                                         dirr3=SESSION_3_DATA_FILE)

        # get attribubtes of participant
        attr = [a for a in dir(P) if a[0]!='_']
        P_dict = {att:getattr(P,att) for att in attr}

        # create list of dicts
        all_Part_dict.append(P_dict)

    # make pandas dataframe to return
    df = pd.DataFrame(all_Part_dict)

    # Some easy preprocessing #
    df['MID']=df['code_id2']

    for MID in df['MID']:
        df.loc[df.MID==MID,'starting_beliefs_self']= df.loc[df.MID==MID,'self_estimate_flipped'].values[0][0]
        df.loc[df.MID==MID,'starting_beliefs_other']=df.loc[df.MID==MID,'other_estimate_flipped'].values[0][0]

    # number of times selected, sat total.
    selected = []
    unselected = []
    for i in range(df.shape[0]):
        selected.extend(df.loc[i,'selected'])
        unselected.extend(df.loc[i,'unselected'])
    selected = np.array(selected)
    unselected = np.array(unselected)

    df_everyone = pd.read_csv('../data_for_model_fitting/data_everyone_from_session2.csv')
    selected_ev = []
    unselected_ev = []
    for i in range(df_everyone.shape[0]):
        sel = df_everyone.loc[i,'selected']
        unsel =df_everyone.loc[i,'unselected']
        if type(sel)==str:
            selected_ev.extend(eval(sel))
            unselected_ev.extend(eval(unsel))
    selected_ev = np.array(selected_ev)
    unselected_ev = np.array(unselected_ev)

    for MID in df.MID:
        df.loc[df.MID==MID,'num_times_selected'] = np.sum(selected==int(MID))
        df.loc[df.MID==MID,'num_times_not_selected'] = np.sum(unselected==int(MID))
        df.loc[df.MID==MID,'%_selected'] = np.sum(selected==int(MID)) / (np.sum(selected==int(MID))+np.sum(unselected==int(MID)))
        df.loc[df.MID==MID,'sat_total']=np.sum(df.loc[df.MID==MID,'sat'].values[0])
        df.loc[df.MID==MID,'num_times_selected_everyone'] = np.sum(selected_ev.astype('str')==MID)
        df.loc[df.MID==MID,'num_times_not_selected_everyone'] = np.sum(unselected_ev.astype('str')==MID)
        df.loc[df.MID==MID,'%_selected_everyone'] = np.sum(selected_ev.astype('str')==MID) / (np.sum(selected_ev.astype('str')==MID)+np.sum(unselected_ev.astype('str')==MID))

    # average/median positive/negative adjustment
    df['self_updates']=0
    df['self_updates']=df['self_updates'].astype(object)
    df['self_updates_logit']=df['self_updates'].astype(object)
    for MID in df.MID:
        estimates = df.loc[df.MID==MID,'self_estimate_flipped'].values[0]
        logit_estimates = logit(df.loc[df.MID==MID,'self_estimate_flipped'].values[0])
        feedback = df.loc[df.MID==MID,'self_feedback'].values[0][1:]

        # calculate adjustments #
        adj = np.diff(estimates)
        adj_logit = np.diff(logit_estimates)
        i = df.loc[(df.MID==MID)].index[0]
        df.at[i,'self_updates']=adj #
        #import pdb; pdb.set_trace()
        df.at[i,'self_updates_logit']=adj_logit #

        # get adjustments after positive v/s negative feedback
        adj_pos = adj[feedback==1]
        adj_neg = adj[feedback==0]
        adj_pos_logit = adj_logit[feedback==1]
        adj_neg_logit = adj_logit[feedback==0]
        df.loc[df.MID==MID,'adj_pos_mean']=np.mean(adj_pos)
        df.loc[df.MID==MID,'adj_neg_mean']=np.mean(adj_neg)
        df.loc[df.MID==MID,'adj_pos_logit_mean']=np.mean(adj_pos_logit)
        df.loc[df.MID==MID,'adj_neg_logit_mean']=np.mean(adj_neg_logit)
        df.loc[df.MID==MID,'adj_pos_median']=np.median(adj_pos)
        df.loc[df.MID==MID,'adj_neg_median']=np.median(adj_neg)

    # add in sequence type
    for MID in df.MID:
        if df.loc[df.MID==MID,'self_feedback'].values[0][1]==0:
            df.loc[df.MID==MID,'self_feedback_cb']=0
        elif df.loc[df.MID==MID,'self_feedback'].values[0][1]==1:
            df.loc[df.MID==MID,'self_feedback_cb']=1

    # collapse session 2 and session 3
    df.rename(columns={'stai_state':'stai_state_sess2',
                        'stai_trait':'stai_trait_sess2',
                        'stai_trait_anx':'stai_trait_anx_sess2',
                        'stai_trait_dep':'stai_trait_dep_sess2',
                        'cesd':'cesd_sess2'},inplace=True)

    df['stai_state']=(df['stai_state_sess2']+df['stai_state_sess3'])/2
    df['stai_trait']=(df['stai_trait_sess2']+df['stai_trait_sess3'])/2
    df['stai_trait_anx']=(df['stai_trait_anx_sess2']+df['stai_trait_anx_sess3'])/2
    df['stai_trait_dep']=(df['stai_trait_dep_sess2']+df['stai_trait_dep_sess3'])/2
    df['cesd']=(df['cesd_sess2']+df['cesd_sess3'])/2

    # subscale summations
    df['scale_anhedonia']=(scale(df['stai_trait_dep_sess2'].values)+\
        scale(df['stai_trait_dep_sess3'].values)+\
        scale(df['masq_ad'].values)+\
        scale(df['cesd_anh_sess3'].values)+\
        scale(df['cesd_anh_sess2'].values))/5
    df['scale_dysphoria']=(scale(df['masq_ds'].values)+\
        scale(df['cesd_dep_sess3'].values)+\
        scale(df['cesd_dep_sess2'].values))/3
    df['scale_cog_anx']=(scale(df['stai_trait_anx_sess2'].values)+\
        scale(df['stai_trait_anx_sess3'].values)+\
        scale(df['pswq'].values))/3
    df['scale_phys_anx']=(scale(df['masq_as'].values)+\
        scale(df['masq_aa'].values))/2

    df['scale_depression']=(df['scale_anhedonia']+df['scale_dysphoria'])/2
    df['scale_anxiety']=(df['scale_cog_anx']+df['scale_phys_anx'])/2
    df['scale_neg_affect']=(df['scale_depression']+df['scale_anxiety'])/2

    # Factor analysis from Elife paper
    scores = pd.read_csv('../factor_scores/items_clinical.omega2.csv',index_col=0)
    scores.MID=scores.MID.astype('str')
    scores.rename(columns={'g':'item_clinical.fa.omega3.g',
                          'F1.':'item_clinical.fa.omega3.anh',
                          'F2.':'item_clinical.fa.omega3.cog_anx',},inplace=True)
    scores['item_clinical.fa.omega3.anh']=-1*scores['item_clinical.fa.omega3.anh']
    df= df.merge(scores,on='MID')

    return(MIDs_w_session3_data,df)


def generate_data_matrices(df):
    data_matrices = {}
    estimates = []
    estimates_centered = []
    feedback = []
    updates = []
    for i in range(len(df)):
        estimates.append(df['self_estimate_flipped'][i])
        estimates_centered.append(df['self_estimate_flipped'][i]-df['self_estimate_flipped'][i][0])
        feedback.append(df['self_feedback'][i][1:])
        updates.append(df['self_updates'][i])

    data_matrices['self_estimates']=np.array(estimates)
    data_matrices['self_estimates_deprior']=np.array(estimates_centered)
    data_matrices['self_feedback']=np.array(feedback)
    data_matrices['self_updates']=np.array(updates)

    return(data_matrices)
