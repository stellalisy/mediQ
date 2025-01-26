import logging
import random
import re
from helper import get_response


def log_info(message, logger_name="detail_logger", print_to_std=False, type="info"):
    # if type(logger) == str and logger in logging.getLogger().manager.loggerDict:
    logger = logging.getLogger(logger_name)
    if type == "error": return logger.error(message)
    if logger: logger.info(message)
    if print_to_std: print(message + "\n")


def expert_response_choice_or_question(messages, options_dict, self_consistency=1, **kwargs):
    """
    Implicit Abstain
    """
    log_info(f"++++++++++++++++++++ Start of Implicit Abstention [expert_basics.py:expert_response_choice_or_question()] ++++++++++++++++++++")
    log_info(f"[<IMPLICIT ABSTAIN PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")
    answers, questions, response_texts = [], [], {}
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    choice_logprobs = []
    for i in range(self_consistency):
        log_info(f"-------------------- Self-Consistency Iteration {i+1} --------------------")
        response_text, log_probs, num_tokens = get_response(messages, **kwargs)
        total_tokens["input_tokens"] += num_tokens["input_tokens"]
        total_tokens["output_tokens"] += num_tokens["output_tokens"]
        if not response_text: 
            log_info("[<IMPLICIT ABSTAIN LM RES>]: " + "No response --> Re-prompt")
            continue
        log_info("[<IMPLICIT ABSTAIN LM RES>]: " + response_text)
        response_text = response_text.replace("Confident --> Answer: ", "").replace("Not confident --> Doctor Question: ", "")

        if "?" not in response_text:
            letter_choice = parse_choice(response_text, options_dict)
            if letter_choice:
                log_info("[<IMPLICIT ABSTAIN PARSED>]: " + letter_choice)
                answers.append(letter_choice)
                response_texts[letter_choice] = response_text
                choice_logprobs.append(log_probs)
        else:
            # not a choice, parse as question
            atomic_question = parse_atomic_question(response_text)
            if atomic_question:
                log_info("[<IMPLICIT ABSTAIN PARSED>]: " + atomic_question)
                questions.append(atomic_question)
                response_texts[atomic_question] = response_text
            
            else:
                log_info("[<IMPLICIT ABSTAIN PARSED>]: " + "FAILED TO PARSE --> Re-prompt")

    if len(answers) + len(questions) == 0:
        log_info("[<IMPLICIT ABSTAIN SC-PARSED>]: " + "No response.")
        return "No response.", None, None, 0.0, {}, total_tokens

    conf_score = len(answers) / (len(answers) + len(questions))
    if len(answers) > len(questions): 
        final_answer = max(set(answers), key = answers.count)
        response_text = response_texts[final_answer]
        top_logprobs = choice_logprobs[answers.index(final_answer)]
        atomic_question = None
    else:
        final_answer = None
        rand_id = random.choice(range(len(questions)))
        atomic_question = questions[rand_id]
        response_text = response_texts[atomic_question]
        top_logprobs = None
    log_info(f"[<IMPLICIT ABSTAIN RETURN>]: atomic_question: {atomic_question}, final_answer: {final_answer}, conf_score: {conf_score} ([{len(answers)} : {len(questions)}])")
    return response_text, atomic_question, final_answer, conf_score, top_logprobs, total_tokens



def expert_response_yes_no(messages, self_consistency=1, **kwargs):
    """
    Binary Abstain
    """
    log_info(f"++++++++++++++++++++ Start of YES/NO Decision [expert_basics.py:expert_response_yes_no()] ++++++++++++++++++++")
    log_info(f"[<YES/NO PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")

    yes_no_responses, log_probs_list, response_texts = [], [], {}
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    for i in range(self_consistency):
        log_info(f"-------------------- Self-Consistency Iteration {i+1} --------------------")
        response_text, log_probs, num_tokens = get_response(messages, **kwargs)
        total_tokens["input_tokens"] += num_tokens["input_tokens"]
        total_tokens["output_tokens"] += num_tokens["output_tokens"]
        if not response_text: 
            log_info("[<YES/NO LM RES>]: " + "No response.")
        log_info("[<YES/NO LM RES>]: " + response_text)
        log_probs_list.append(log_probs)

        yes_choice = parse_yes_no(response_text)
        log_info("[<YES/NO PARSED>]: " + yes_choice)
        yes_no_responses.append(yes_choice)
        response_texts[yes_choice] = response_text
    
    if yes_no_responses.count("YES") > yes_no_responses.count("NO"):
        yes_choice = "YES"
        log_probs = log_probs_list[yes_no_responses.index("YES")]
    else:
        yes_choice = "NO"
        log_probs = log_probs_list[yes_no_responses.index("NO")]
    log_info(f"[<YES/NO RETURN>]: yes_choice: {yes_choice}, confidence: {yes_no_responses.count('YES')/len(yes_no_responses)}")
    return response_texts[yes_choice], yes_choice, yes_no_responses.count("YES")/len(yes_no_responses), log_probs, total_tokens



def expert_response_confidence_score(messages, self_consistency=1, **kwargs):
    """
    Numerical Abstain
    """
    log_info(f"++++++++++++++++++++ Start of Numerical Confidence Score [expert_basics.py:expert_response_confidence_score()] ++++++++++++++++++++")
    log_info(f"[<CONF SCORE PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")

    conf_scores, log_probs_list, response_texts = [], {}, {}
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    for i in range(self_consistency):
        log_info(f"-------------------- Self-Consistency Iteration {i+1} --------------------")
        response_text, log_probs, num_tokens = get_response(messages, **kwargs)
        total_tokens["input_tokens"] += num_tokens["input_tokens"]
        total_tokens["output_tokens"] += num_tokens["output_tokens"]
        if not response_text: 
            log_info("[<CONF SCORE LM RES>]: " + "No response.")
            continue
        log_info("[<CONF SCORE LM RES>]: " + response_text)

        conf_score = parse_confidence_score(response_text)
        conf_scores.append(conf_score)
        log_probs_list[conf_score] = log_probs
        response_texts[conf_score] = response_text
        log_info(f"[<CONF SCORE PARSED>]: {conf_score}")
    
    if len(conf_scores) > 0:
        avg_conf_score = sum(conf_scores) / len(conf_scores)
        # response_text = "CONFIDENCE SCORE: " + str(avg_conf_score)
        temp = [abs(r-avg_conf_score) for r in conf_scores]
        response_text = response_texts[conf_scores[temp.index(min(temp))]]
        log_probs = log_probs_list[conf_scores[temp.index(min(temp))]]
    else:
        avg_conf_score, response_text, log_probs = 0, "No response.", None
    log_info(f"[<CONF SCORE RETURN>] (average conf score): {avg_conf_score}")
    return response_text, avg_conf_score, log_probs, total_tokens



def expert_response_scale_score(messages, self_consistency=1, **kwargs):
    """
    Scale Abstain
    """
    log_info(f"++++++++++++++++++++ Start of Scale Confidence Score [expert_basics.py:expert_response_scale_score()] ++++++++++++++++++++")
    log_info(f"[<SCALE SCORE PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")

    conf_scores, log_probs_list, response_texts = [], {}, {}
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    for i in range(self_consistency):
        log_info(f"-------------------- Self-Consistency Iteration {i+1} --------------------")
        response_text, log_probs, num_tokens = get_response(messages, **kwargs)
        total_tokens["input_tokens"] += num_tokens["input_tokens"]
        total_tokens["output_tokens"] += num_tokens["output_tokens"]
        if not response_text:
            log_info("[<SCALE SCORE LM RES>]: " + "No response.")
            continue
        log_info("[<SCALE SCORE LM RES>]: " + response_text)

        conf_score = parse_likert_scale(response_text)
        conf_scores.append(conf_score)
        log_probs_list[conf_score] = log_probs
        response_texts[conf_score] = response_text
        log_info("[<SCALE SCORE PARSED>]: " + str(conf_score))
    
    if len(conf_scores) > 0:
        avg_conf_score = sum(conf_scores) / len(conf_scores)
        temp = [abs(r-avg_conf_score) for r in conf_scores]
        response_text = response_texts[conf_scores[temp.index(min(temp))]]
        log_probs = log_probs_list[conf_scores[temp.index(min(temp))]]
    else:
        avg_conf_score, response_text, log_probs = 0, "No response.", None
    log_info(f"[<SCALE SCORE RETURN>] (average conf score]): {avg_conf_score}")
    return response_text, avg_conf_score, log_probs, total_tokens



def expert_response_choice(messages, options_dict, **kwargs):
    """
    Get intermediate answer choice regardless of abstention decision
    """
    log_info(f"++++++++++++++++++++ Start of Multiple Chocie Decision [expert_basics.py:expert_response_choice()] ++++++++++++++++++++")
    log_info(f"[<CHOICE PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")
    response_text, log_probs, num_tokens = get_response(messages, **kwargs)
    if not response_text: 
        log_info("[<CHOICE LM RES>]: " + "No response.")
        return "No response.", None, num_tokens
    log_info("[<CHOICE LM RES>]: " + response_text)

    letter_choice = parse_choice(response_text, options_dict)
    if letter_choice:
        log_info("[<CHOICE PARSED>]: " + letter_choice)
    else:
        log_info("[<CHOICE PARSED>]: " + "FAILED TO PARSE.")
    
    return response_text, letter_choice, num_tokens



def expert_response_question(messages, **kwargs):
    """
    Get follow-up question
    """
    log_info(f"++++++++++++++++++++ Start of Question Generator [expert_basics.py:expert_response_question()] ++++++++++++++++++++")
    log_info(f"[<QUESTION GENERATOR PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")
    response_text, log_probs, num_tokens = get_response(messages, **kwargs)
    if not response_text: 
        log_info("[<QUESTION GENERATOR LM RES>]: " + "No response.")
        return "No response.", None, num_tokens
    log_info("[<QUESTION GENERATOR LM RES>]: " + response_text)

    atomic_question = parse_atomic_question(response_text)
    if atomic_question:
        log_info("[<QUESTION GENERATOR PARSED>]: " + atomic_question)
    else:
        log_info("[<QUESTION GENERATOR PARSED>]: " + "FAILED TO PARSE.")
    
    return response_text, atomic_question, num_tokens



############################
# Helper Functions for Parsing Responses
############################

def parse_atomic_question(response_text):
    questions = []
    for line in response_text.split("\n"):
        if '?' in line:
            questions.append(line.split(":")[-1].strip())
        
    if len(questions) == 0:
        log_info("can't find question in answer: {}".format(response_text), type="error")
        return None
            
    atomic_question = questions[-1].replace("'", "").replace('"', "").strip()
    return atomic_question

def parse_choice(response_text, options_dict):
    if response_text.strip() in ["A", "B", "C", "D"]:
        return response_text.strip()
    for response_line in response_text.split("\n"):
        for op_letter, op_text in options_dict.items():
            if op_text.lower() in response_line.lower():
                log_info(f"....Found {op_text} in response line: {response_line}")
                return op_letter
        for op_letter in options_dict.keys():
            if op_letter in [token for token in re.sub(r"[,.;@#()?!'/&:$]+\ *", " ", response_line).split(' ')]:
                # op_letter_str = str(op_letter) if op_letter else "none"
                # response_line_str = str(response_line) if response_line else "none"
                log_info(f"....Found {op_letter} in response line: {response_line}")
                return op_letter
    log_info("can't parse choice: {}".format(response_text), type="error")
    return None

def parse_yes_no(response_text):
    temp_processed_response = response_text.lower().replace('.','').replace(',','').replace(';','').replace(':','').split("DECISION:")[-1].strip()
    yes_answer = "yes" in temp_processed_response
    no_answer = "no" in temp_processed_response
    if yes_answer == no_answer:
        yes_choice = "NO"
        log_info("can't parse yes/no abstain answer: {}".format(response_text), type="error")
    if yes_answer: yes_choice = "YES"
    elif no_answer: yes_choice = "NO"
    return yes_choice

def parse_confidence_score(response_text):
    # parse the probability
    float_regex = re.compile(r'\d+\.\d+')
    scores = re.findall(float_regex, response_text)

    if len(scores) == 0:
        log_info("can't parse confidence score - answer: {}".format(response_text), type="error")
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
        log_info("can't parse likert confidence score: {}".format(response_text), type="error")
    return conf_score