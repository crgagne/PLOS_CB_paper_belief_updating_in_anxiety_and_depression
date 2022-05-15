The data for session 1 consists of:
* SAT scores and grades ('sat_grades_entry_responses.json'); Q0-Q2 correspond to scores for reading, math, and writing, respectively.
* responses to the PSWQ worry questionnaire (see 'code_for_data_processing/questionnaires/pswq.py' to link the question ID to the question text)
* responses to the following questions about the participants' working style. Participants answered using a Likert scale with possible responses from 1-9.
  - Q0: I prefer to work: on my own / in a group
  - Q1: When approaching a problem, I tend to focus on: the details / the big picture
  - Q2: When working through a problem, I tend to think: intuitively / analytically
  - Q3: When stuck on a problem, I prefer to seek advice: immediately / after I've worked on it for awhile
  - Q4: When presenting my work, I prefer to present in an order that is: sequential / thematic,
  - Q5: When starting a new project, I prefer to plan: the entire project in one go / one step at a time
* a profile image ('working_style_profile.png'); participants' responses to questions about their working style were normalized across participants and used to create this image.
* responses to consent forms ('consent_part1_2_tags', 'consent_part1_2b_tags', 'instructions_backstory_tags')

The data for session 2 consists of:
* responses to the CESD, STAI, and MASQ questionnaires (see 'code_for_data_processing/questionnaires' to link the question ID to the question text)
* responses to the following multi-part question: "When choosing who to work with, how much did you consider each of the following aspects of the profiles?":
  - whether group vs. independent worker
  - whether analytic vs. intuitive thinker
  - whether strategic vs. adaptive planner
  - the personal description
  - SAT scores
  - grades
  - (Participants could respond with: 'did not consider','considered a little','considered a lot','considered exclusively')
* participants' choices for which candidate in each pair they would rather work with ('pair_responses.csv'). The four digit number is the participant ID and the responses (1 or 2) correspond to choosing the first or second ID in the pair.
* responses to consent forms (instructionsNaN_tags.txt)

The data for session 3 consists of:
* participants' trial-by-trial beliefs that they were in the more/less popular half of participants (self-estimates.csv); their beliefs for the 'other' participant were in ('other-estimates.csv'). The columns in this file are:
  - cb: whether they were asked to report beliefs about being in the top (1) or bottom (0) half of participants
  - estimate: their reported beliefs
  - feedback: whether they were chosen (1) or not (0)
  - pair0: the participant's ID
  - pair1: the ID of the participant they were compared to
  - rt: how long the participant viewed the feedback
  - truth: whether participants were in the top (1) or bottom (0) half of participants
* responses to CESD and STAI questionnaires, administered for a second time (see 'code_for_data_processing/questionnaires' to link the question ID to the question text)
* responses to the following multi-part question about what information participants used to update their beliefs ('survey-likertNaN_responses0'): 'Here, we'd like you to indicate how much each the following things influenced your belief about whether you are in the more popular half of students as a partner.
  - 'The total number of times I was chosen to work with'
  - 'How often someone with better grades was chosen over me'
  - 'How often someone with worse grades was chosen over me'
  - 'How often someone with better SAT scores was chosen over me'
  - 'How often someone with worse SAT scores was chosen over me'
* responses to the same multi-part question, but answered in reference to the 'other' participant ('survey-likertNaN_responses1')
* responses to the question (survey-multi-choiceNaN_responses0): 'Would you like to know what type of information people most often used while selecting who to work with? Yes/No'
* responses to the question (survey-multi-choiceNaN_responses1): 'Would you like to know whether you were in the top or bottom half in terms of how many times you were chosen to work with? Yes/No'
