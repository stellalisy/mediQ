import json
import os
import time
import logging
from args import get_args
from patient import Patient
import importlib

def setup_logger(name, file):
    logger = logging.getLogger(name)
    handler = logging.FileHandler(file, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

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
        else: processed_ids = {sample["id"]: sample["interactive_system"]["choice"] == sample["info"]["correct_answer_idx"] for sample in output_data}
    else:
        processed_ids = []

    expert_module = importlib.import_module(args.expert_module)
    expert_class = getattr(expert_module, args.expert_class)
    patient_module = importlib.import_module(args.patient_module)
    patient_class = getattr(patient_module, args.patient_class)
    
    patient_data_path = os.path.join(args.data_dir, args.dev_filename)
    patient_data = load_data(patient_data_path)

    history_logger = setup_logger('history_logger', args.history_log_filename)
    general_logger = setup_logger('general_logger', args.log_filename)

    num_processed = 0
    correct = []

    for pid, sample in patient_data.items():
        if pid in processed_ids:
            print(f"Skipping patient {pid} as it has already been processed.")
            correct.append(processed_ids[pid])
            continue

        history_logger.info(f"\n\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nPATIENT #{pid}")

        if type(sample["context"]) == str:
            sample["context"] = sample["context"].split(". ")

        choice, questions, answers, temp_choice_list, temp_additional_info = run_patient_interaction(expert_class, patient_class, sample)

        output_dict = {
            "id": pid,
            "info": {
                "context": ' '.join(sample["context"]),
                "correct_answer": sample["answer"],
                "correct_answer_idx": sample["answer_idx"],
                "question": sample["question"],
                "options": sample["options"],
                # "facts": sample["facts"],
            },
            "interactive_system": {
                "choice": choice,
                "questions": questions,
                "answers": answers,
                "num_questions": len(questions),
                "intermediate_choices": temp_choice_list,
                "correct": choice == sample["answer_idx"],
                "temp_additional_info": temp_additional_info
            },
            # "eval": {
            #     "confidence_scores": [],  # TODO: how confident is the expert_system in their choice
            #     "repeat_question_score": [],
            #     "repeat_answer_score": [],
            #     "relevancy_score": [],
            #     "delta_confidence_score": [],
            #     "specificity_score": []
            # }
        }

        correct.append(choice == sample["answer_idx"])
        num_processed += 1
        history_logger.info(f"||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nInteraction ended for patient #{pid}")


        with open(args.output_filename, 'a+') as f:
            f.write(json.dumps(output_dict) + '\n')

        general_logger.info(f'Processed {num_processed}/{len(patient_data)} patients | Accuracy: {sum(correct) / len(correct)}')
    print(f"Accuracy: {sum(correct)} / {len(correct)} = {sum(correct) / len(correct) if len(correct)>0 else None}")


def run_patient_interaction(expert_class, patient_class, sample):
    expert_system = expert_class(args, sample["question"], sample["options"])
    patient_system = patient_class(args, sample)  # Assuming the patient_system is initialized with the sample which includes necessary context
    temp_choice_list = []
    temp_additional_info = []  # To store optional data like confidence scores

    while len(patient_system.get_questions()) < args.max_questions:
        patient_state = patient_system.get_state()
        response_dict = expert_system.respond(patient_state)

        # The expert_system's respond method now returns a dictionary
        response_type = response_dict["type"]
        
        # Optional return values for analysis, e.g., confidence score, logprobs
        temp_additional_info.append({k: v for k, v in response_dict.items() if k not in ["type", "answered_idx", "question"]})

        if response_type == "question":
            temp_choice = response_dict["answered_idx"]  # still make the Expert generate a choice based on the current state for intermediate evaluation
            temp_choice_list.append(temp_choice)  # Log the question as an intermediate choice
            followup_question = response_dict["question"]
            patient_system.respond(followup_question)  # Patient generates an answer based on the last question asked, and add to memory

        elif response_type == "choice":
            expert_decision = response_dict["answered_idx"]
            temp_choice_list.append(expert_decision)
            return expert_decision, patient_system.get_questions(), patient_system.get_answers(), temp_choice_list, temp_additional_info
        
        else:
            raise ValueError("Invalid response type from expert_system.")
    
    # If max questions are reached and no final decision has been made
    patient_state = patient_system.get_state()
    response_dict = expert_system.respond(patient_state)
    stuck_response = response_dict["answered_idx"]
    # Optional return values for analysis, e.g., confidence score, logprobs
    temp_additional_info.append({k: v for k, v in response_dict.items() if k != "answered_idx"})
    
    return stuck_response, patient_system.get_questions(), patient_system.get_answers(), temp_choice_list + [stuck_response], temp_additional_info


if __name__ == "__main__":
    args = get_args()
    main()
