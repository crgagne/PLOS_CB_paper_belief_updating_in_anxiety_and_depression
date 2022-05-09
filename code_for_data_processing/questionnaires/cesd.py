
reverse_score_items = [l-6 for l in  [21,17,9,13]]

choices = ["Rarely or none of the time (less than one day)",
"Some or a little of the time (1-2 days)",
"Occasionally or a moderate amount of time (3-4 days)",
"All of the time (5-7 days)",
'(prefer not to answer)'];

choices_dict = {"Rarelyornoneofthetime(lessthanoneday)": 0,
"Someoralittleofthetime(1-2days)":1,
"Occasionallyoramoderateamountoftime(3-4days)":2,
"Allofthetime(5-7days)":3,
'(prefernottoanswer)': 0}

# contains catch questions
questions = [
  "I was bothered by things that usually don’t bother me",
  "I did not feel like eating; my appetite was poor",
  "I felt that I could not shake off the blues even with help from my family",
  "I felt that I was just as good as other people",
  "I had trouble keeping my mind on what I was doing",
  "I felt depressed",
  "I felt that everything I did was an effort",
  "I felt hopeful about the future",
  "I thought my life had been a failure",
  "I felt fearful",
  "My sleep was restless",
  "I was happy",
  "I talked less than usual",
  "I felt lonely",
  "People were unfriendly",
  "I enjoyed life",
  "I had crying spells",
  "I felt sad",
  "I felt that people disliked me",
  "I could not 'get going'"
]

# depressed
list_dep_add = [3,6,14,18]
list_dep_add =[l-1 for l in list_dep_add]

# anhedonic (all positively coded)
list_anh_add = [4,8,12,16]
list_anh_add =[l-1 for l in list_anh_add]

# somatic
list_som_add = [1,2,5,7,11,20]
list_som_add =[l-1 for l in list_som_add]

def CESD_score(answers, threshold=.75,ex_count=False):
    assert len(answers) == 20 and max(answers) <= 3 and min(answers) >= 0
    score = 0
    score_forward = 0
    score_reverse = 0
    d_pass = True

    if ex_count:
        count_mode = max([answers.count(x) for x in range(0,3)]) #Finds the count of the mode
        if count_mode >= threshold * len(answers): #Checks if someone just chose the same answer for N times
            d_pass = False

    for a in range(len(answers)):
        if a in reverse_score_items:
            score += (3 - answers[a])
            score_reverse += (3 - answers[a])
        else:
            score += answers[a]
            score_forward += answers[a]

    # depressed symptoms
    subscale_dep = 0
    listadd = list_dep_add
    for l in listadd:
        subscale_dep+=answers[l]

    # anh symptoms
    subscale_anh = 0
    listadd = list_anh_add
    for l in listadd:
        subscale_anh+=(3-answers[l])

    # som symptoms
    subscale_som = 0
    listadd = list_som_add
    for l in listadd:
        subscale_som+=answers[l]

    return score, d_pass, score_forward, score_reverse,subscale_dep,subscale_anh,subscale_som
