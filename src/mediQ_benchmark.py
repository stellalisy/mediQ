import json
import os
import time
import logging
from args import get_args
from patient import Patient
import importlib

def setup_logger(name, file):
    if not file: return None
    logger = logging.getLogger(name)
    handler = logging.FileHandler(file, mode='a')
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def log_info(message, print_to_std=False):
    if history_logger: history_logger.info(message)
    if detail_logger: detail_logger.info(message)
    if print_to_std: print(message + "\n")

def load_data(filename):
    with open(filename, "r") as json_file:
        json_list = list(json_file)
    data = [json.loads(line) for line in json_list]
    data = {item['id']: item for item in data}
    return data

def main():
    if os.path.exists(args.output_filename):
        with open(args.output_filename, "r") as f:
            lines = f.readlines()
        output_data = [json.loads(line) for line in lines]
        if len(lines) == 0: processed_ids = []
        else: processed_ids = {sample["id"]: {"correct": sample["interactive_system"]["letter_choice"] == sample["info"]["correct_answer_idx"],
                                              "timeout": len(sample["interactive_system"]["intermediate_choices"]) > args.max_questions,
                                              "turns": sample["interactive_system"]["num_questions"]}
                                for sample in output_data}
    else:
        processed_ids = []

    expert_module = importlib.import_module(args.expert_module)
    expert_class = getattr(expert_module, args.expert_class)
    patient_module = importlib.import_module(args.patient_module)
    patient_class = getattr(patient_module, args.patient_class)
    
    patient_data_path = os.path.join(args.data_dir, args.dev_filename)
    patient_data = load_data(patient_data_path)

    num_processed = 0
    correct_history, timeout_history, turn_lengths = [], [], []

    for pid, sample in patient_data.items():
        if pid in processed_ids:
            print(f"Skipping patient {pid} as it has already been processed.")
            correct_history.append(processed_ids[pid]["correct"])
            timeout_history.append(processed_ids[pid]["timeout"])
            turn_lengths.append(processed_ids[pid]["turns"])
            continue

        log_info(f"|||||||||||||||||||| PATIENT #{pid} ||||||||||||||||||||")
        letter_choice, questions, answers, temp_choice_list, temp_additional_info, sample_info = run_patient_interaction(expert_class, patient_class, sample)
        log_info(f"|||||||||||||||||||| Interaction ended for patient #{pid} ||||||||||||||||||||\n\n\n")

        output_dict = {
            "id": pid,
            "interactive_system": {
                "correct": letter_choice == sample["answer_idx"],
                "letter_choice": letter_choice,
                "questions": questions,
                "answers": answers,
                "num_questions": len(questions),
                "intermediate_choices": temp_choice_list,
                "temp_additional_info": temp_additional_info
            },
            "info": sample_info,
            # TODO: add additional evaluation metrics for analysis, some metrics can be found in src/evaluate.py
            # "eval": {
            #     "confidence_scores": [],
            #     "repeat_question_score": [],
            #     "repeat_answer_score": [],
            #     "relevancy_score": [],
            #     "delta_confidence_score": [],
            #     "specificity_score": []
            # }
        }

        # create the directory if it does not exist
        os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)
        with open(args.output_filename, 'a+') as f:
            f.write(json.dumps(output_dict) + '\n')

        correct_history.append(letter_choice == sample["answer_idx"])
        timeout_history.append(len(temp_choice_list) > args.max_questions)
        turn_lengths.append(len(temp_choice_list))
        num_processed += 1
        accuracy = sum(correct_history) / len(correct_history) if len(correct_history) > 0 else None
        timeout_rate = sum(timeout_history) / len(timeout_history) if len(timeout_history) > 0 else None
        avg_turns = sum(turn_lengths) / len(turn_lengths) if len(turn_lengths) > 0 else None

        results_logger.info(f'Processed {num_processed}/{len(patient_data)} patients | Accuracy: {accuracy}')
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processed {num_processed}/{len(patient_data)} patients | Accuracy: {accuracy} | Timeout Rate: {timeout_rate} | Avg. Turns: {avg_turns}")
    print(f"Accuracy: {sum(correct_history)} / {len(correct_history)} = {accuracy}")
    print(f"Timeout Rate: {sum(timeout_history)} / {len(timeout_history)} = {timeout_rate}")
    print(f"Avg. Turns: {avg_turns}")
    

def run_patient_interaction(expert_class, patient_class, sample):
    expert_system = expert_class(args, sample["question"], sample["options"])
    patient_system = patient_class(args, sample)  # Assuming the patient_system is initialized with the sample which includes necessary context
    temp_choice_list = []
    temp_additional_info = []  # To store optional data like confidence scores

    while len(patient_system.get_questions()) < args.max_questions:
        log_info(f"==================== Turn {len(patient_system.get_questions()) + 1} ====================")
        patient_state = patient_system.get_state()
        response_dict = expert_system.respond(patient_state)
        log_info(f"[Expert System]: {response_dict}")
        
        # Optional return values for analysis, e.g., confidence score, logprobs
        temp_additional_info.append({k: v for k, v in response_dict.items() if k not in ["type", "letter_choice", "question"]})

        if response_dict["type"] == "question":
            # still make the Expert generate a choice based on the current state for intermediate evaluation, log the question as an intermediate choice
            temp_choice_list.append(response_dict["letter_choice"])
            # Patient generates an answer based on the last question asked, and add to memory
            patient_response = patient_system.respond(response_dict["question"])
            log_info(f"[Patient System]: {patient_response}")

        elif response_dict["type"] == "choice":
            expert_decision = response_dict["letter_choice"]
            temp_choice_list.append(expert_decision)
            sample_info = {
                "initial_info": patient_system.initial_info,
                "correct_answer": sample["answer"],
                "correct_answer_idx": sample["answer_idx"],
                "question": sample["question"],
                "options": sample["options"],
                "context": sample["context"],
                "facts": patient_system.facts, # if the FactSelectPatient patient module is used, this will store the atomic facts the patient used to answer questions for reproducibility
            }
            return expert_decision, patient_system.get_questions(), patient_system.get_answers(), temp_choice_list, temp_additional_info, sample_info
        
        else:
            raise ValueError("Invalid response type from expert_system.")
    
    # If max questions are reached and no final decision has been made
    log_info(f"==================== Max Interaction Length ({args.max_questions} turns) Reached --> Force Final Answer ====================")
    patient_state = patient_system.get_state()
    response_dict = expert_system.respond(patient_state)
    log_info(f"[Expert System]: {response_dict}")
    stuck_response = response_dict["letter_choice"]
    # Optional return values for analysis, e.g., confidence score, logprobs
    temp_additional_info.append({k: v for k, v in response_dict.items() if k != "letter_choice"})

    sample_info = {
        "initial_info": patient_system.initial_info,
        "correct_answer": sample["answer"],
        "correct_answer_idx": sample["answer_idx"],
        "question": sample["question"],
        "options": sample["options"],
        "context": sample["context"],
        "facts": patient_system.facts, # if the FactSelectPatient patient module is used, this will store the atomic facts the patient used to answer questions for reproducibility
    }
    
    return stuck_response, patient_system.get_questions(), patient_system.get_answers(), temp_choice_list + [stuck_response], temp_additional_info, sample_info


if __name__ == "__main__":
    args = get_args()
    results_logger = setup_logger('results_logger', args.log_filename)
    history_logger = setup_logger('history_logger', args.history_log_filename)
    detail_logger = setup_logger('detail_logger', args.detail_log_filename)
    message_logger = setup_logger('message_logger', args.message_log_filename)
    main()