
#There are no catch questions and choices are already enumerated as their score

reverse_score_items = [0, 2, 7, 9, 10]

questions = [
    "If I don’t have enough time to do everything, I don’t worry about it.",
    "My worries overwhelm me.",
    "I don’t tend to worry about things.",
    "Many situations make me worry.",
    "I know I shouldn’t worry about things, but I just can’t help it.",
    "When I am under pressure, I worry a lot.",
    "I am always worrying about something.",
    "I find it easy to dismiss worrisome thoughts.",
    "As soon as I finish one task, I start to worry about everything else I have to do.",
    "I never worry about anything.",
    "When there is nothing more I can do about a concern, I don’t worry about it anymore.",
    "I’ve been a worrier all my life.",
    "I notice that I have been worrying about things.",
    "Once I start worrying, I can’t stop.",
    "I worry all the time.",
    "I worry about projects until they are done."
]

def PSWQ_score(answers, threshold=.75,ex_count=False):
    assert len(answers) == 16 and max(answers) <= 5 and min(answers) >= 1
    score = 0
    score_forward = 0
    score_reverse = 0
    p_pass = True

    if ex_count:
        count_mode = max([answers.count(x) for x in range(0,8)]) #Finds the count of the mode
        if count_mode >= threshold * len(answers): #Checks if someone just chose the same answer for N times
            p_pass = False

    for a in range(len(answers)):
        if a in reverse_score_items:
            score += 6 - answers[a]
            score_reverse += (6 - answers[a])
        else:
            score += answers[a]
            score_forward += answers[a]
    return score, p_pass,score_forward, score_reverse
