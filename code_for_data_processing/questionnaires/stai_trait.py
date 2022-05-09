
reverse_score_items = [0,2,5,6,9,12,13,15,18]

choices = ["Almost never", "Sometimes", "Often", "Almost always",'(prefer not to answer)']

choices_dict = {"Almostnever": 1,"Sometimes":2,"Often":3,"Almostalways":4,'(prefernottoanswer)': 0}

# contains catch questions
questions = [
      "I feel pleasant",
      "I feel nervous and restless", # anx 1
      "I feel satisfied with myself",
      "I wish I could be as happy as others seem to be",
      "I feel like a failure",
      "I feel rested",
      "I am calm, cool and collected",
      "I feel that difficulties are piling up so that I cannot overcome them", #anx 7
      "I worry too much over something that really doesn’t matter", # anx 8
      "I am happy",
      "I have disturbing thoughts", # anx 10
      "I lack self-confidence",
      "I feel secure",
      "I make decisions easily",
      "I feel inadequate",
      "I am content",
      "Some unimportant thought runs through my mind and bothers me", # anx 16
      "I take disappointments so keenly that I can’t put them out of my mind", #anx 17
      "I am a steady person",
      "I get in a state of tension or turmoil as I think over my recent concerns and interests" #anx 19
    ]

anx_list = [1,7,8,10,16,17,19]

def STAI_trait_score(answers, threshold=.75,ex_count=False):
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

    score_anx = 0
    for a in range(len(answers)):
        if a in anx_list:
            if a in reverse_score_items:
                score_anx+=(5-answers[a])
            else:
                score_anx+=answers[a]

    score_dep = score - score_anx



    return score, d_pass, score_forward, score_reverse, score_anx, score_dep
