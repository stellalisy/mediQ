import sys
import logging
import random
import re
from helper import get_response

global history_logger
try: history_logger = logging.getLogger("history_logger")
except: history_logger = None

LIKERT_THRESHOLD = 4
PROB_THRESHOLD = 0.8

def log_info(message, logger=history_logger):
    logger = logging.getLogger(logger) if type(logger) == str else logger
    if logger: logger.info(message)
    sys.stdout.write(message + "\n")

def expert_response_choice_or_question(message, options_dict, self_consistency=1, model_name="gpt-3.5-turbo", **kwargs):
    """
    Implicit Abstain
    """
    answers = []
    answer_responses = []
    questions = []
    question_responses = []
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    choice_logprobs = []
    for i in range(self_consistency):

        log_info("[<PROMPT>]: " + message[-1]["content"])
        response_text, log_probs, num_tokens = get_response(message, model_name, top_logprobs=20, max_tokens=60, **kwargs)
        total_tokens["input_tokens"] += num_tokens["input_tokens"]
        total_tokens["output_tokens"] += num_tokens["output_tokens"]
        if not response_text: 
            log_info("[<LM RES>]: " + "No response --> Re-prompt")
            continue
        log_info("[<LM RES>]: " + response_text)
        response_text = response_text.replace("Confident --> Answer: ", "").replace("Not confident --> Doctor Question: ", "")

        if "?" not in response_text:
            letter_choice = parse_choice(response_text, options_dict)
            if letter_choice:
                log_info("[<PARSED>]: " + letter_choice)
                answers.append(letter_choice)
                answer_responses.append(response_text)
                choice_logprobs.append(log_probs)
        else:
            # not a choice, parse as question
            atomic_question = parse_atomic_question(response_text)
            if atomic_question:
                log_info("[<PARSED>]: " + atomic_question)
                questions.append(atomic_question)
                question_responses.append(response_text)
            
            else:
                log_info("[<PARSED>]: " + "FAILED TO PARSE --> Re-prompt")

    if len(answers) + len(questions) == 0:
        log_info("[<PARSED>]: " + "No response.")
        return "No response.", None, None, 0.0, {}, total_tokens

    conf_score = len(answers) / (len(answers) + len(questions))
    log_info(f"[<IMPLICIT ABSTAIN RETURN>]: answers: {answers}, questions: {questions}, conf_score: {conf_score} ([{len(answers)}, {len(questions)}])")
    if len(answers) > len(questions): 
        final_answer = max(set(answers), key = answers.count)
        response_text = answer_responses[answers.index(final_answer)]
        top_logprobs = choice_logprobs[answers.index(final_answer)]
        atomic_question = None
    else:
        final_answer = None
        rand_id = random.choice(range(len(questions)))
        response_text = question_responses[rand_id]
        atomic_question = questions[rand_id]
        top_logprobs = None
    return response_text, atomic_question, final_answer, conf_score, top_logprobs, total_tokens



def expert_response_yes_no(messages, model_name="gpt-3.5-turbo", self_consistency=1, **kwargs):
    """
    Binary Abstain
    """
    log_info("[<PROMPT>]: " + messages[-1]["content"])

    yes_no_responses, log_probs_list = [], []
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    for i in range(self_consistency):
        response_text, log_probs, num_tokens = get_response(messages, model_name, **kwargs)
        total_tokens["input_tokens"] += num_tokens["input_tokens"]
        total_tokens["output_tokens"] += num_tokens["output_tokens"]
        if not response_text: 
            log_info("[<LM RES>]: " + "No response.")
        log_info("[<LM RES>]: " + response_text)
        log_probs_list.append(log_probs)

        yes_choice = parse_yes_no(response_text)
        yes_no_responses.append(yes_choice)
    if yes_no_responses.count("YES") > yes_no_responses.count("NO"):
        yes_choice = "YES"
        log_probs = log_probs_list[yes_no_responses.index("YES")]
    else:
        yes_choice = "NO"
        log_probs = log_probs_list[yes_no_responses.index("NO")]

    sys.stdout.write("[YES_NO]Response text: {}\n[YES_NO]Extracted: {}\n".format(response_text, yes_choice))
    sys.stdout.write("----------------------------------------------------\n")
    log_info("[<PARSED>]: " + yes_choice)
    return response_text, yes_choice, yes_no_responses.count("YES")/len(yes_no_responses), log_probs, total_tokens



def expert_response_confidence_score(messages, model_name="gpt-3.5-turbo", self_consistency=1, **kwargs):
    """
    Numerical Abstain
    """
    log_info("[<PROMPT>]: " + messages[-1]["content"])

    conf_scores = []
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    for _ in range(self_consistency):
        response_text, log_probs, num_tokens = get_response(messages, model_name, **kwargs)
        total_tokens["input_tokens"] += num_tokens["input_tokens"]
        total_tokens["output_tokens"] += num_tokens["output_tokens"]
        if not response_text: 
            log_info("[<LM RES>]: " + "No response.")
            # return "No response.", 0.0, num_tokens
        log_info("[<LM RES>]: " + response_text)

        conf_score = parse_confidence_score(response_text)
        conf_scores.append(conf_score)
        log_info(f"[<PARSED>]: {conf_score}")
    
    avg_conf_score = sum(conf_scores) / len(conf_scores) if len(conf_scores) > 0 else 0.0
    response_text = "CONFIDENCE SCORE: " + str(avg_conf_score)
    sys.stdout.write("[CONFIDENCE]Response text: {}\n[CONFIDENCE]Extracted (average): {}\n".format(response_text, avg_conf_score))
    sys.stdout.write("----------------------------------------------------\n")
    log_info(f"[<PARSED>] [average conf score] {avg_conf_score}")
    return response_text, avg_conf_score, log_probs, total_tokens


def expert_response_scale(messages, model_name="gpt-3.5-turbo", abstain_threshold=LIKERT_THRESHOLD, self_consistency=1, **kwargs):
    """
    Scale Abstain
    """
    log_info("[<PROMPT>]: " + messages[-1]["content"])

    conf_scores, log_probs_list = [], []
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    for i in range(self_consistency):
        response_text, log_probs, num_tokens = get_response(messages, model_name, **kwargs)
        total_tokens["input_tokens"] += num_tokens["input_tokens"]
        total_tokens["output_tokens"] += num_tokens["output_tokens"]
        if not response_text:
            log_info("[<LM RES>]: " + "No response.")
            # return "No response.", False, num_tokens
        log_info("[<LM RES>]: " + response_text)
        log_probs_list.append(log_probs)

        conf_score = parse_likert_scale(response_text)
        conf_scores.append(conf_score)
        log_info("[<PARSED>]: " + str(conf_score))
    
    avg_conf_score = sum(conf_scores) / len(conf_scores) if len(conf_scores) > 0 else 0
    if avg_conf_score >= abstain_threshold:
        log_info(f"[<PARSED>]: line 416: AVERAGE CONF SCORE = {str(avg_conf_score)} ([{', '.join([str(s) for s in conf_scores])}]) >= {abstain_threshold}, return \"YES\"")
        return response_text, "YES", avg_conf_score, log_probs, total_tokens
    else:
        log_info(f"[<PARSED>]: line 419: AVERAGE CONF SCORE = {str(avg_conf_score)} ([{', '.join([str(s) for s in conf_scores])}]) < {abstain_threshold}, return \"NO\"")
        return response_text, "NO", avg_conf_score, log_probs, total_tokens


def expert_response_choice(messages, options_dict, model_name="gpt-3.5-turbo", **kwargs):
    """
    Get intermediate answer choice regardless of abstention decision
    """
    log_info("[<GET ANSWER PROMPT>]: " + messages[-1]["content"])
    response_text, log_probs, num_tokens = get_response(messages, model_name, **kwargs)
    if not response_text: 
        log_info("[<LM RES>]: " + "No response.")
        return "No response.", None, num_tokens
    log_info("[<LM RES>]: " + response_text)

    letter_choice = parse_choice(response_text, options_dict)
    if letter_choice:
        log_info("[<PARSED>]: " + letter_choice)
    else:
        log_info("[<PARSED>]: " + "FAILED TO PARSE.")
    
    return response_text, letter_choice, num_tokens


def expert_response_question(messages, model_name="gpt-3.5-turbo", **kwargs):
    """
    Get follow-up question
    """
    log_info("[<GET QUESTION PROMPT>]: " + messages[-1]["content"])
    response_text, log_probs, num_tokens = get_response(messages, model_name, **kwargs)
    if not response_text: 
        log_info("[<LM RES>]: " + "No response.")
        return "No response.", None, num_tokens
    log_info("[<LM RES>]: " + response_text)

    atomic_question = parse_atomic_question(response_text)
    if atomic_question:
        log_info("[<PARSED>]: " + atomic_question)
    else:
        log_info("[<PARSED>]: " + "FAILED TO PARSE.")
    
    return response_text, atomic_question, num_tokens


def parse_atomic_question(response_text):
    questions = []
    for line in response_text.split("\n"):
        if '?' in line:
            questions.append(line.split(":")[-1].strip())
        
    if len(questions) == 0:
        logging.error("can't find question in answer: {}".format(response_text))
        log_info("[<PARSED>]: " + "FAILED TO PARSE.")
        return None
            
    atomic_question = questions[-1].replace("'", "").replace('"', "").strip()
    # sys.stdout.write("[QUESTION LM Response text]: {}\n[QUESTION Extracted]: {}\n".format(response_text, atomic_question))
    # sys.stdout.write("----------------------------------------------------\n")
    # if atomic_question: log_info("[<PARSED>]: " + atomic_question)
    # else: log_info("[<PARSED>]: " + "FAILED TO PARSE.")
    return atomic_question

def parse_choice(response_text, options_dict):
    choice = None
    for response_line in response_text.split("\n"):
        for op_letter, op_text in options_dict.items():
            if op_text.lower() in response_line.lower():
                print(f"Found {op_text} in response line: {response_line}")
                return op_letter
        for op_letter in options_dict.keys():
            if op_letter in [token for token in re.sub(r"[,.;@#()?!'/&:$]+\ *", " ", response_line).split(' ')]:
                op_letter_str = op_letter if op_letter else "none"
                response_line_str = response_line if response_line else "none"
                print(f"Found {op_letter_str} in response line: {response_line_str}")
                return op_letter
    return choice

def parse_yes_no(response_text):
    temp_processed_response = response_text.lower().replace('.','').replace(',','').replace(';','').replace(':','').split()
    temp_processed_response = temp_processed_response.split("DECISION:")[-1].strip()
    yes_answer = "yes" in temp_processed_response
    no_answer = "no" in temp_processed_response
    if yes_answer and no_answer:
        yes_choice = "NO"
        logging.error("can't parse RG abstain answer: {}".format(response_text))
    if yes_answer == False and no_answer == False:
        yes_choice = "NO"
        logging.error("can't parse RG abstain answer: {}".format(response_text))
    if yes_answer: yes_choice = "YES"
    elif no_answer: yes_choice = "NO"
    logging.error("can't parse yes/no answer: {}".format(response_text))
    return yes_choice

def parse_confidence_score(response_text):
    # parse the probability
    float_regex = re.compile(r'\d+\.\d+')
    scores = re.findall(float_regex, response_text)

    if len(scores) == 0:
        logging.error("can't parse confidence score - answer: {}".format(response_text))
        score = round(0.2 + (random.random() - random.random()) * 0.2, 4)
        return score
    
    prob = float(scores[-1])
    if len(scores) > 1: logging.warning("more than one confidence score - using last: {}".format(response_text))
    if prob > 1: logging.warning("confidence score > 1: {}".format(response_text))
    return prob

def parse_likert_scale(response_text):
    temp_processed_response = response_text.lower().replace('.','').replace(',','').replace(';','').replace(':','')
    if "very confident" in temp_processed_response:
        conf_score = 5
    elif "somewhat confident" in temp_processed_response:
        conf_score = 4
    elif "neither confident nor unconfident" in temp_processed_response:
        conf_score = 3
    elif "neither confident or unconfident" in temp_processed_response:
        conf_score = 3
    elif "somewhat unconfident" in temp_processed_response:
        conf_score = 2
    elif "very unconfident" in temp_processed_response:
        conf_score = 1
    else:
        conf_score = 0
        logging.error("can't parse likert confidence score: {}".format(response_text))
    return conf_score