
catch_questions = [55,76]
reverse_score_items = []

choices = ["not at all", "a little bit", "moderately", "quite a bit", "extremely",'(prefer not to answer)']

choices_dict = {"notatall": 1,"alittlebit":2,"moderately":3,"quiteabit":4,"extremely":5,'(prefernottoanswer)': 0}

# contains catch questions
questions = [
      "Felt cheerful",
      "Felt afraid",
      "Startled easily",
      "Felt confused",
      "Slept very well",
      "Felt sad",
      "Felt very alert",
      "Felt discouraged",
      "Felt nauseous",
      "Felt like crying",
      "Felt successful",
       "Had diarrhea",
       "Felt worthless",
       "Felt really happy",
       "Felt nervous", "Felt depressed",
      "Felt irritable",
      "Felt optimistic",
      "Felt faint",
      "Felt uneasy",
      "Felt really bored",
      "Felt hopeless",
      "Felt like I was having a lot of fun",
      "Blamed myself for a lot of things",
      "Felt numbness or tingling in my body",
      "Felt withdrawn from other people",
      "Seemed to move quickly and easily",
      "Was afraid I was going to lose control",
      "Felt dissatisfied with everything",
      "Looked forward to things with enjoyment",
      "Had trouble remembering things",
      "Felt like I didn't need much sleep",
      "Felt like nothing was very enjoyable",
      "Felt like something awful was going to happen",
      "Felt like I had accomplished a lot",
      "Felt like I had a lot of interesting things to do",
      "Did not have much of an appetite",
      "Felt like being with other people",
      "Felt like it took extra effort to get started",
      "Felt like I had a lot to look forward to",
      "Thoughts and ideas came to me very easily",
      "Felt pessimistic about the future",
      "Felt like I could do everything I needed to do",
      "Felt like there wasn't anything interesting or fun to do",
      "Had pain in my chest",
      "Felt really talkative",
      "Felt like a failure",
      "Had hot or cold spells",
      "Was proud of myself",
      "Felt very restless",
      "Had trouble falling asleep",
      "Felt dizzy or lightheaded",
      "Felt unattractive",
      "Felt very clearheaded",
      "Was short of breath",
      "Select extremely", ## catch question
      "Felt sluggish or tired",
      "Hands were shaky",
      "Felt really 'up' or lively",
      "Was unable to relax",
      "Felt like being by myself",
      "Felt like I was choking",
      "Was able to laugh easily",
      "Had an upset stomach",
      "Felt inferior to others",
      "Had a lump in my throat",
      "Felt really slowed down",
      "Had a very dry mouth",
      "Felt confident about myself",
      "Muscles twitched or trembled",
      "Had trouble making decisions",
      "Felt like I was going crazy",
      "Felt like I had a lot of energy",
      "Was afraid I was going to die",
      "Was disappointed in myself",
      "Heart was racing or pounding",
      "Select moderately", # catch question
      "Had trouble concentrating",
      "Felt tense or high-strung",
      "Felt hopeful about the future",
      "Was trembling or shaking",
      "Had trouble paying attention",
      "Muscles were tense or sore",
      "Felt keyed up, on edge",
      "Had trouble staying asleep",
      "Worried a lot about things",
      "Had to urinate frequently",
      "Felt really good about myself",
      "Had trouble swallowing",
      "Hands were cold or sweaty",
      "Thought about death or suicide",
      "Got tired or fatigued easily",
]

questions_wo_catch = [
      "Felt cheerful",
      "Felt afraid",
      "Startled easily",
      "Felt confused",
      "Slept very well",
      "Felt sad",
      "Felt very alert",
      "Felt discouraged",
      "Felt nauseous",
      "Felt like crying",
      "Felt successful",
       "Had diarrhea",
       "Felt worthless",
       "Felt really happy",
       "Felt nervous", "Felt depressed",
      "Felt irritable",
      "Felt optimistic",
      "Felt faint",
      "Felt uneasy",
      "Felt really bored",
      "Felt hopeless",
      "Felt like I was having a lot of fun",
      "Blamed myself for a lot of things",
      "Felt numbness or tingling in my body",
      "Felt withdrawn from other people",
      "Seemed to move quickly and easily",
      "Was afraid I was going to lose control",
      "Felt dissatisfied with everything",
      "Looked forward to things with enjoyment",
      "Had trouble remembering things",
      "Felt like I didn't need much sleep",
      "Felt like nothing was very enjoyable",
      "Felt like something awful was going to happen",
      "Felt like I had accomplished a lot",
      "Felt like I had a lot of interesting things to do",
      "Did not have much of an appetite",
      "Felt like being with other people",
      "Felt like it took extra effort to get started",
      "Felt like I had a lot to look forward to",
      "Thoughts and ideas came to me very easily",
      "Felt pessimistic about the future",
      "Felt like I could do everything I needed to do",
      "Felt like there wasn't anything interesting or fun to do",
      "Had pain in my chest",
      "Felt really talkative",
      "Felt like a failure",
      "Had hot or cold spells",
      "Was proud of myself",
      "Felt very restless",
      "Had trouble falling asleep",
      "Felt dizzy or lightheaded",
      "Felt unattractive",
      "Felt very clearheaded",
      "Was short of breath",
      "Felt sluggish or tired",
      "Hands were shaky",
      "Felt really 'up' or lively",
      "Was unable to relax",
      "Felt like being by myself",
      "Felt like I was choking",
      "Was able to laugh easily",
      "Had an upset stomach",
      "Felt inferior to others",
      "Had a lump in my throat",
      "Felt really slowed down",
      "Had a very dry mouth",
      "Felt confident about myself",
      "Muscles twitched or trembled",
      "Had trouble making decisions",
      "Felt like I was going crazy",
      "Felt like I had a lot of energy",
      "Was afraid I was going to die",
      "Was disappointed in myself",
      "Heart was racing or pounding",
      "Had trouble concentrating",
      "Felt tense or high-strung",
      "Felt hopeful about the future",
      "Was trembling or shaking",
      "Had trouble paying attention",
      "Muscles were tense or sore",
      "Felt keyed up, on edge",
      "Had trouble staying asleep",
      "Worried a lot about things",
      "Had to urinate frequently",
      "Felt really good about myself",
      "Had trouble swallowing",
      "Hands were cold or sweaty",
      "Thought about death or suicide",
      "Got tired or fatigued easily",
]

list_aa = [3,19,25,45,48,52,55,57,61,67,69,73,75,79,85,87,88]
list_aa =[l-1 for l in list_aa]

list_ad_add = [21,26,33,39,44,53,66,89]
list_ad_add = [l-1 for l in list_ad_add]

list_ad_sub = [1,14,18,23,27,30,35,36,40,49,58,72,78,86]
list_ad_sub = [l-1 for l in list_ad_sub]

list_ds = [6,8,10,13,16,22,24,42,47,56,64]
list_ds = [l-1 for l in list_ds]

list_as = [2,9,12,15,20,59,63,65,77,81,82]
list_as = [l-1 for l in list_as]

list_ms_add = [4,17,29,31,34,37,50,51,70,76,80,83,84,90]
list_ms_add = [l-1 for l in list_ms_add]

list_ms_sub =[5]
list_ms_sub = [l-1 for l in list_ms_sub]

def MASQ_score(answers, threshold=.75,ex_count=False):
    assert len(answers) == 92 and max(answers) <= 5 and min(answers) >= 0
    score = 0
    d_pass = True

    if ex_count:
        count_mode = max([answers.count(x) for x in range(0,5)]) #Finds the count of the mode
        if count_mode >= threshold * len(answers): #Checks if someone just chose the same answer for N times
            d_pass = False

    # deal with catch questions
    if answers[catch_questions[0]]!=5: # select extremely
        pass_catch0 =False
    else:
        pass_catch0 =True

    if answers[catch_questions[1]]!=3: # select moderately
        pass_catch1 =False
    else:
        pass_catch1 =True
    if pass_catch0==False or pass_catch1==False:
        d_pass = False

    # remove catch question answers
    answers_wo_catch = [answers[l] for l in range(len(answers)) if l not in catch_questions]

    # anxious arousal
    subscale_aa = 0
    listadd = list_aa
    for l in listadd:
        subscale_aa+=answers_wo_catch[l]

    # anhedonic depression
    summ1 = 0
    listadd = list_ad_add
    for l in listadd:
        summ1+=answers_wo_catch[l]
    listsub= list_ad_sub
    summ2 = 0
    for l in listsub:
        summ2+=answers_wo_catch[l]

    subscale_ad=summ1+(84-summ2)

    # depressive symptoms
    subscale_ds = 0
    listadd = list_ds
    for l in listadd:
        subscale_ds+=answers_wo_catch[l]

    # anxious symptoms
    subscale_as = 0
    listadd = list_as
    for l in listadd:
        subscale_as+=answers_wo_catch[l]

    # mixed symptoms
    subscale_ms = 0
    listadd = list_ms_add

    for l in listadd:
        subscale_ms+=answers_wo_catch[l]
    listsub= list_ms_sub
    for l in listsub:
        subscale_ms+=6-answers_wo_catch[l]

    return subscale_aa,subscale_ad,subscale_ds,subscale_as,subscale_ms, d_pass, pass_catch0,pass_catch1,answers_wo_catch
