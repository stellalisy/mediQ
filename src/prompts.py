expert_system = {
    "meditron_system_msg_old": "You are a medical doctor answering real-world medical entrance exam questions. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following multiplechoice question. Base your answer on the current and standard practices referenced in medical guidelines.\nTask: You will be asked to reason through the current patient's information and either ask an information seeking question or choose an option.",

    "meditron_system_msg": "You are a medical doctor answering real-world medical entrance exam questions. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following multiple choice question. Base your answer on the current and standard practices referenced in medical guidelines.",

    "basic_system_msg": "You are an experienced doctor trying to make a medical decision about a patient.",

    "empty_system_msg": "",

    "only_choice": "Please answer with ONLY the correct letter choice (JUST ONE LETTER and NOTHING ELSE): A, B, C, or D.",

    "system": "You are an experienced doctor trying to make a medical decision about a patient.",
    
    "starter": """A patient comes into the clinic presenting with a symptom as described in the conversation log below:\n\nCONVERSATION LOG:\n""",

    "question_word": "Doctor Question",
    "answer_word": "Patient Response",

    "task": "Given the information from above, your task is to choose one of four options that best answers the inquiry.",
    
    "prompt": """\nMedical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. How confident are you to pick the correct option to the inquiry factually using the conversation log? In the first line of your response, generate the probability as a float from 0 to 1.\n\nIf there are missing features that prevent you from picking a confident and factual answer to the inquiry, consider which features are not yet asked about in the conversation log; then, consider which missing feature is the most important to ask the patient in order to provide the most helpful information toward a correct medical decision. Ask ONE SPECIFIC ATOMIC QUESTION to address this feature. The question should be bite-sized, and NOT ask for too much at once. In the second line of your response, generate the atomic question and nothing else.\n\nHowever, if you feel like you already have enough information from the above question-answer pairs to answer the patient inquiry, use the above information to produce a factual conclusion. In this case, answer with ONLY the correct letter choice and nothing else.""",
    
    "yes_no": "Now, are you confident to pick the correct option to the inquiry factually using the conversation log? Answer with YES or NO and NOTHING ELSE.",

    
    "implicit": "Given the information so far, if you are confident to pick an option correctly and factually, respond with the letter choice and NOTHING ELSE. Otherwise, if you are not confident to pick an option and need more information, ask ONE SPECIFIC ATOMIC QUESTION to the patient. The question should be bite-sized, NOT ask for too much at once, and NOT repeat what has already been asked. In this case, respond with the atomic question and NOTHING ELSE.",

    "implicit_RG": "Given the information so far, if you are confident to pick an option correctly and factually, respond in the format:\nREASON: a one-sentence explanation of why you are choosing a particular option.\nANSWER: the letter choice and NOTHING ELSE. Otherwise, if you are not confident to pick an option and need more information, ask ONE SPECIFIC ATOMIC QUESTION to the patient. The question should be bite-sized, NOT ask for too much at once, and NOT repeat what has already been asked. In this case, respond in the format:\nREASON: a one-sentence explanation of why you should ask the particular question.\nQUESTION: the atomic question and NOTHING ELSE.",

    "binary": "Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. Now, are you confident to pick the correct option to the inquiry factually using the conversation log? Answer with YES or NO and NOTHING ELSE.",
    
    "binary_RG": "Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. Up to this point, are you confident to pick the correct option to the inquiry factually using the conversation log? Answer in the following format:\nREASON: a one-sentence explanation of why you are or are not confident and what other information is needed.\nDECISION: YES or NO.",

    "numcutoff": "Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. What is your confidence score to pick the correct option to the inquiry factually using the conversation log? Answer with the probability as a float from 0.0 to 1.0 and NOTHING ELSE.",

    "numcutoff_RG": "Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. What is your confidence score to pick the correct option to the inquiry factually using the conversation log? Answer strictly in the following format:\nREASON: a one-sentence explanation of why you are or are not confident and what other information is needed.\nSCORE: your confidence score written as a float from 0.0 to 1.0.",

    "numerical": "Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. What is your confidence score to pick the correct option to the inquiry factually using the conversation log? Answer with the probability as a float from 0.0 to 1.0 and NOTHING ELSE.",

    "numerical_RG": "Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. What is your confidence score to pick the correct option to the inquiry factually using the conversation log? Answer strictly in the following format:\nREASON: a one-sentence explanation of why you are or are not confident and what other information is needed.\nSCORE: your confidence score written as a float from 0.0 to 1.0.",

    "scale": """Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. How confident are you to pick the correct option to the problem factually using the conversation log? Choose between the following ratings:
"Very Confident" - The correct option is supported by all evidence, and there is enough evidence to eliminate the rest of the answers, so the option can be confirmed conclusively.
"Somewhat Confident" - I have reasonably enough information to tell that the correct option is more likely than other options, more information is helpful to make a conclusive decision.
"Neither Confident or Unconfident" - There are evident supporting the correct option, but further evidence is needed to be sure which one is the correct option.
"Somewhat Unconfident" - There are evidence supporting more than one options, therefore more questions are needed to further distinguish the options.
"Very Unconfident" - There are not enough evidence supporting any of the options, the likelihood of picking the correct option at this point is near random guessing.\n\nThink carefully step by step, respond with the chosen confidence rating ONLY and NOTHING ELSE.""",

    "scale_RG": """Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. How confident are you to pick the correct option to the problem factually using the conversation log? Choose between the following ratings:
"Very Confident" - The correct option is supported by all evidence, and there is enough evidence to eliminate the rest of the answers, so the option can be confirmed conclusively.
"Somewhat Confident" - I have reasonably enough information to tell that the correct option is more likely than other options, more information is helpful to make a conclusive decision.
"Neither Confident or Unconfident" - There are evident supporting the correct option, but further evidence is needed to be sure which one is the correct option.
"Somewhat Unconfident" - There are evidence supporting more than one options, therefore more questions are needed to further distinguish the options.
"Very Unconfident" - There are not enough evidence supporting any of the options, the likelihood of picking the correct option at this point is near random guessing.\n\nAnswer in the following format:\nREASON: a one-sentence explanation of why you are or are not confident and what other information is needed.\nDECISION: chosen rating from the above list.""",

    "yes_no": "Now, are you confident to pick the correct option to the inquiry factually using the conversation log? Answer with YES or NO and NOTHING ELSE.",

    "verbal_abstain_llama": "Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. Up to this point, are you confident to pick the correct option to the inquiry factually using the conversation log? Answer in the following format:\nDECISION: YES or NO.",

    "implicit_abstain": "Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. In the following cases, either answer the question or ask another information-seeking question:\n1. If you are confident to pick the correct option to the inquiry factually using the conversation log, answer with ONLY the correct letter choice and NOTHING ELSE.\n2. If you are not confident to pick the correct option to the inquiry factually using the conversation log, consider what are the missing information that would help you differenciate among the options. Ask ONE SPECIFIC ATOMIC QUESTION to address the missing feature. The question should be bite-sized, and NOT ask for too much at once. Make sure to NOT repeat any questions from the above conversation log. Generate the atomic question and NOTHING ELSE.",
 
    "atomic_question": "If there are missing features that prevent you from picking a confident and factual answer to the inquiry, consider which features are not yet asked about in the conversation log; then, consider which missing feature is the most important to ask the patient in order to provide the most helpful information toward a correct medical decision. Ask ONE SPECIFIC ATOMIC QUESTION to address this feature. The question should be bite-sized, and NOT ask for too much at once. Generate the atomic question and NOTHING ELSE.",
    
    "atomic_question_improved": "If there are missing features that prevent you from picking a confident and factual answer to the inquiry, consider which features are not yet asked about in the conversation log; then, consider which missing feature is the most important to ask the patient in order to provide the most helpful information toward a correct medical decision. You can ask about any relevant information about the patient’s case, such as family history, tests and exams results, treatments already done, etc. Consider what are the common questions asked in the specific subject relating to the patient’s known symptoms, and what the best and most intuitive doctor would ask. Ask ONE SPECIFIC ATOMIC QUESTION to address this feature. The question should be bite-sized, and NOT ask for too much at once. Make sure to NOT repeat any questions from the above conversation log. Answer in the following format:\nATOMIC QUESTION: the atomic question and NOTHING ELSE.\nATOMIC QUESTION: ",

    "answer": "Assume that you already have enough information from the above question-answer pairs to answer the patient inquiry, use the above information to produce a factual conclusion. Respond with the correct letter choice (A, B, C, or D) and NOTHING ELSE.\nLETTER CHOICE: ",
    
    "non_interactive": {
        "starter": "A patient comes into the clinic presenting with a symptom as described in the statements below:",
        "question_prompt": "Given the information from above, your task is to choose one of four options that best answers the following question: ",
        "response": "To the best of your ability, answer with ONLY the correct letter choice and nothing else."
    },

    "curr_template": """A patient comes into the clinic presenting with a symptom as described in the conversation log below:
    
PATIENT INFORMATION: {}
CONVERSATION LOG:
{}
QUESTION: {}
OPTIONS: {}
YOUR TASK: {}"""
    
}

patient_system = {
    "system": "You are a truthful assistant that understands the patient's information, and you are trying to answer questions from a medical doctor about the patient. ",
    "header": "Below is a list of factual statements about the patient:\n",
    "prompt": 'Which of the above atomic factual statements answers the question? If no statement answers the question, simply say "The patient cannot answer this question, please do not ask this question again." Answer only what the question asks for. Do not provide any analysis, inference, or implications. Respond by selecting all statements that answer the question from above ONLY and NOTHING ELSE.',

    "prompt_new": """Below is a list of factual statements about the patient:\n
{}\n
Which of the above atomic factual statements answers the question? If no statement answers the question, simply say "The patient cannot answer this question, please do not ask this question again." Answer only what the question asks for. Do not provide any analysis, inference, or implications. Respond with all statements that directly answer the question from above verbatim ONLY and NOTHING ELSE, with one statement on each line.

Example:
Question from the doctor: [some question]
STATEMENTS:\n[example statement: she reports that...]\n[example statement: she has a history of...]

Question from the doctor: {}
""",

    "system_first_person": "You are a patient with a list of symptoms, and you task is to truthfully answer questions from a medical doctor. ",
    "header_first_person": "Below is a list of atomic facts about you, use ONLY the information in this list and answer the doctor's question.",
    "prompt_first_person": """Which of the above atomic factual statements are the best answer to the question? Select at most two statements. If no statement answers the question, simply say "The patient cannot answer this question, please do not ask this question again." Do not provide any analysis, inference, or implications. Respond by reciting the matching statements, then convert the selected statements into first person perspective as if you are the patient but keep the same information. Generate your answer in this format:

STATEMENTS: 
FIRST PERSON: """
}

conformal_scores = {
    "prompt_score": "Given the information from above, your task is to assign a likelihood score to each option. Respond with the probability as a float from 0 to 1 and NOTHING ELSE. Respond in the following format:\nA: 0.0\nB: 0.0\nC: 0.0\nD: 0.0",
}