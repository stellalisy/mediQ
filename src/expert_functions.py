import prompts
import expert_basics
import logging

PROB_THRESHOLD = 0.8
SCALE_THRESHOLD = 4.0

def answer_to_idx(answer):
    return ord(answer) - ord("A")

def log_info(message, logger="detail_logger", print_to_std=False):
    if type(logger) == str and logger in logging.getLogger().manager.loggerDict:
        logger = logging.getLogger(logger)
    if logger: logger.info(message)
    if print_to_std: print(message + "\n")



def fixed_abstention_decision(max_depth, patient_state, inquiry, options_dict, **kwargs):
    """
    Fixed abstention strategy based on the current interaction length.
    If the interaction length is less than the max depth, abstain, otherwise answer.
    """
    # first get the model's abstention decision
    log_info(f"++++++++++++++++++++ Start of Fixed Abstention [expert_functions.py:fixed_abstention_decision()] ++++++++++++++++++++")
    abstain_decision = len(patient_state['interaction_history']) < max_depth
    conf_score = 1 if abstain_decision else 0
    log_info(f"[ABSTENTION RESPONSE]: {abstain_decision}\n")

    # second, no matter what the model's abstention decision is, get an intermediate answer for evaluation and analysis
    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'

    prompt_answer = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, prompts.expert_system["answer"])
    messages_answer = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_answer}
    ]
    response_text, letter_choice, num_tokens = expert_basics.expert_response_choice(messages_answer, options_dict, **kwargs)

    log_info(f"[FIXED ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, letter_choice: {letter_choice}, usage: {num_tokens}\n")
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages_answer,
        "letter_choice": letter_choice,
    }



def implicit_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, **kwargs):
    """
    Implicit abstention strategy based on the current patient state.
    This function uses the expert system to make a decision on whether to abstain or not based on the current patient state.
    """
    # Get the response from the expert system
    prompt_key = "implicit_RG" if rationale_generation else "implicit"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
    
    # first get the model's abstention decision
    prompt_abstain = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, abstain_task_prompt)

    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    response_text, atomic_question, letter_choice, conf_score, top_logprobs, num_tokens = expert_basics.expert_response_choice_or_question(messages, options_dict, **kwargs)
    log_info(f"[ABSTENTION PROMPT]: {messages}")
    log_info(f"[ABSTENTION RESPONSE]: {response_text}\n")
    messages.append({"role": "assistant", "content": response_text})

    if atomic_question != None: abstain_decision = True  # if the model generates a question, it is abstaining from answering, therefore abstain decision is True
    elif letter_choice != None: abstain_decision = False  # if the model generates an answer, it is not abstaining from answering, therefore abstain decision is False
    else: abstain_decision = True  # if the model generates neither an answer nor a question, it is abstaining from answering, therefore abstain decision is True

    # second, no matter what the model's abstention decision is, get an intermediate answer for evaluation and analysis
    # note that we get this for free if implicit abstain already chooses an answer instead of a question
    if letter_choice == None:
        prompt_answer = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, prompts.expert_system["answer"])
        messages_answer = [
            {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
            {"role": "user", "content": prompt_answer}
        ]
        response_text, letter_choice, num_tokens_answer = expert_basics.expert_response_choice(messages_answer, options_dict, **kwargs)
        num_tokens["input_tokens"] += num_tokens_answer["input_tokens"]
        num_tokens["output_tokens"] += num_tokens_answer["output_tokens"]

    log_info(f"[IMPLICIT ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, letter_choice: {letter_choice}, usage: {num_tokens}, atomic_question: {atomic_question}\n")
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages,
        "letter_choice": letter_choice,
        "atomic_question": atomic_question,
    }



def binary_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, **kwargs):
    """
    Binary abstention strategy based on the current patient state.
    This function prompts the user to make a binary decision on whether to abstain or not based on the current patient state.
    """
    # Get the response from the expert system
    prompt_key = "binary_RG" if rationale_generation else "binary"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
    
    # first get the model's abstention decision
    prompt_abstain = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, abstain_task_prompt)

    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    response_text, abstain_decision, conf_score, log_probs, num_tokens = expert_basics.expert_response_yes_no(messages, **kwargs)
    abstain_decision = abstain_decision.lower() == 'no'
    log_info(f"[ABSTENTION PROMPT]: {messages}")
    log_info(f"[ABSTENTION RESPONSE]: {response_text}\n")
    messages.append({"role": "assistant", "content": response_text})

    # second, no matter what the model's abstention decision is, get an intermediate answer for evaluation and analysis
    prompt_answer = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, prompts.expert_system["answer"])
    messages_answer = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_answer}
    ]
    response_text, letter_choice, num_tokens_answer = expert_basics.expert_response_choice(messages_answer, options_dict, **kwargs)
    num_tokens["input_tokens"] += num_tokens_answer["input_tokens"]
    num_tokens["output_tokens"] += num_tokens_answer["output_tokens"]

    log_info(f"[BINARY ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, letter_choice: {letter_choice}, usage: {num_tokens}\n")
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages,
        "letter_choice": letter_choice,
    }



def numerical_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, **kwargs):
    """
    Numerical abstention strategy based on the current patient state.
    This function prompts the model to produce a numerical confidence score of how confident it is in its decision, then ask whether it wants to proceed
    """

    # Get the response from the expert system
    prompt_key = "numerical_RG" if rationale_generation else "numerical"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
    
    # first get the model's abstention decision
    prompt_abstain = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, abstain_task_prompt)

    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    response_text, conf_score, log_probs, num_tokens = expert_basics.expert_response_confidence_score(messages, **kwargs)
    messages.append({"role": "assistant", "content": response_text})
    
    messages.append({"role": "user", "content": prompts.expert_system["yes_no"]})
    # third return is supposed to be the conf_score in the binary setup, but we don't use it here because has conf score from last turn.
    response_text, abstain_decision, _, log_probs, num_tokens_2 = expert_basics.expert_response_yes_no(messages, **kwargs)
    abstain_decision = abstain_decision.lower() == 'no'
    num_tokens["input_tokens"] += num_tokens_2["input_tokens"]
    num_tokens["output_tokens"] += num_tokens_2["output_tokens"]
    log_info(f"[ABSTENTION PROMPT]: {messages}")
    log_info(f"[ABSTENTION RESPONSE]: {response_text}\n")
    messages.append({"role": "assistant", "content": response_text})


    # second, no matter what the model's abstention decision is, get an intermediate answer for evaluation and analysis
    prompt_answer = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, prompts.expert_system["answer"])
    messages_answer = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_answer}
    ]
    response_text, letter_choice, num_tokens_answer = expert_basics.expert_response_choice(messages_answer, options_dict, **kwargs)
    num_tokens["input_tokens"] += num_tokens_answer["input_tokens"]
    num_tokens["output_tokens"] += num_tokens_answer["output_tokens"]

    log_info(f"[NUMERICAL ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, letter_choice: {letter_choice}, usage: {num_tokens}\n")
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages,
        "letter_choice": letter_choice,
    }



def numcutoff_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, abstain_threshold, **kwargs):
    """
    Numcutoff abstention strategy based on the current patient state.
    This function prompts the model to produce a numerical confidence score of how confident it is in its decision, then decide abstention based on arbitrarily set threshold
    """
    if not abstain_threshold: abstain_threshold = PROB_THRESHOLD
    
    # Get the response from the expert system
    prompt_key = "numcutoff_RG" if rationale_generation else "numcutoff"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
    
    # first get the model's abstention decision
    prompt_abstain = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, abstain_task_prompt)

    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    response_text, conf_score, log_probs, num_tokens = expert_basics.expert_response_confidence_score(messages, abstain_threshold=abstain_threshold, **kwargs)
    abstain_decision = conf_score < abstain_threshold
    log_info(f"[ABSTENTION PROMPT]: {messages}")
    log_info(f"[ABSTENTION RESPONSE]: {response_text}\n")
    messages.append({"role": "assistant", "content": response_text})

    # second, no matter what the model's abstention decision is, get an intermediate answer for evaluation and analysis
    prompt_answer = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, prompts.expert_system["answer"])
    messages_answer = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_answer}
    ]
    response_text, letter_choice, num_tokens_answer = expert_basics.expert_response_choice(messages_answer, options_dict, **kwargs)
    num_tokens["input_tokens"] += num_tokens_answer["input_tokens"]
    num_tokens["output_tokens"] += num_tokens_answer["output_tokens"]

    log_info(f"[NUMCUTOFF ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, letter_choice: {letter_choice}, usage: {num_tokens}\n")
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages,
        "letter_choice": letter_choice,
    }



def scale_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, abstain_threshold, **kwargs):
    """
    Likert abstention strategy based on the current patient state.
    This function prompts the model to produce a likert scale confidence score of how confident it is in its decision, then decide abstention based on a cutoff
    """
    if not abstain_threshold: abstain_threshold = SCALE_THRESHOLD

    # Get the response from the expert system
    prompt_key = "scale_RG" if rationale_generation else "scale"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
    
    # first get the model's abstention decision
    prompt_abstain = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, abstain_task_prompt)

    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    response_text, conf_score, log_probs, num_tokens = expert_basics.expert_response_scale_score(messages, abstain_threshold=abstain_threshold, **kwargs)
    abstain_decision = conf_score < abstain_threshold
    log_info(f"[ABSTENTION PROMPT]: {messages}")
    log_info(f"[ABSTENTION RESPONSE]: {response_text}\n")
    messages.append({"role": "assistant", "content": response_text})

    # second, no matter what the model's abstention decision is, get an intermediate answer for evaluation and analysis
    prompt_answer = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, prompts.expert_system["answer"])
    messages_answer = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_answer}
    ]
    response_text, letter_choice, num_tokens_answer = expert_basics.expert_response_choice(messages_answer, options_dict, **kwargs)
    num_tokens["input_tokens"] += num_tokens_answer["input_tokens"]
    num_tokens["output_tokens"] += num_tokens_answer["output_tokens"]

    log_info(f"[SCALE ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, letter_choice: {letter_choice}, usage: {num_tokens}\n")
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages,
        "letter_choice": letter_choice,
    }



def question_generation(patient_state, inquiry, options_dict, messages, independent_modules, **kwargs):
    task_prompt = prompts.expert_system["atomic_question_improved"]

    if independent_modules:
        patient_info = patient_state["initial_info"]
        conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
        options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
        prompt = prompts.expert_system["curr_template"].format(patient_info, conv_log, inquiry, options_text, task_prompt)

        messages = [
            {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
            {"role": "user", "content": prompt}
        ]
    else:
        messages.append({"role": "user", "content": task_prompt})

    response_text, atomic_question, num_tokens = expert_basics.expert_response_question(messages, **kwargs)
    log_info(f"[ATOMIC QUESTION PROMPT]: {messages}")
    log_info(f"[ATOMIC QUESTION RESPONSE]: {atomic_question}\n")
    messages.append({"role": "assistant", "content": atomic_question})

    log_info(f"[ATOMIC QUESTION RETURN]: {atomic_question}, usage: {num_tokens}\n")
    return {
        "atomic_question": atomic_question,
        "messages": messages,
        "usage": num_tokens,
    }
