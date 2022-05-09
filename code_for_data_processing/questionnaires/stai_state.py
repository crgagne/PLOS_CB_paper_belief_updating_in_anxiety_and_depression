
reverse_score_items = [0,1,4,7,9,10,14,15,18,19]

choices = ["Not at all", "Somewhat", "Moderately so", "Very much so",'(prefer not to answer)'];

choices_dict = {"Notatall": 1,"Somewhat":2,"Moderatelyso":3,"Verymuchso":4,'(prefernottoanswer)': 0}

# contains catch questions
questions = ["I feel calm",
"I feel secure",
"I am tense",
"I feel strained",
"I feel at ease",
"I feel upset",
"I am presently worrying over possible misfortunes",
"I feel satisfied",
"I feel frightened",
"I feel comfortable",
"I feel self-confident",
"I feel nervous",
"I feel jittery",
"I feel indecisive",
"I am relaxed",
"I feel content",
"I am worried",
"I feel confused",
"I feel steady",
"I feel pleasant"]

def STAI_state_score(answers, threshold=.75,ex_count=False):
    assert len(answers) == 20 and max(answers) <= 4 and min(answers) >= 0
    score = 0
    score_forward = 0
    score_reverse = 0
    d_pass = True

    if ex_count:
        count_mode = max([answers.count(x) for x in range(0,4)]) #Finds the count of the mode
        if count_mode >= threshold * len(answers): #Checks if someone just chose the same answer for N times
            d_pass = False

    for a in range(len(answers)):
        if answers[a] == 0: #If their answer was "Prefer Not to Answer"
            continue
        elif a in reverse_score_items:
            score += (5 - answers[a])
            score_reverse += (5 - answers[a])
        else:
            score += answers[a]
            score_forward += answers[a]
    return score, d_pass, score_forward, score_reverse
