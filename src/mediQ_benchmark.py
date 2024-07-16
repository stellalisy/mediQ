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
    module = importlib.import_module(args.expert_module)
    expert_class = getattr(module, args.expert_class)
    

    patient_data_path = os.path.join(args.data_dir, args.dev_filename)
    patient_data = load_data(patient_data_path)

    history_logger = setup_logger('history_logger', args.history_log_filename)
    general_logger = setup_logger('general_logger', args.log_filename)

    num_processed = 0
    correct = []

    for pid, sample in patient_data.items():
        history_logger.info(f"\n\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nPATIENT #{pid}")

        if type(sample["context"]) == str:
            sample["context"] = sample["context"].split(". ")

        choice, questions, answers, temp_choice_list, temp_additional_info = run_patient_interaction(expert_class, sample)

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
    print(f"Accuracy: {sum(correct) / len(correct)}")


def run_patient_interaction(expert_class, sample):
    expert_system = expert_class(args, sample["question"], sample["options"])
    patient_system = Patient(args, sample)  # Assuming the patient_system is initialized with the sample which includes necessary context
    temp_choice_list = []
    temp_additional_info = []  # To store optional data like confidence scores

    while len(patient_system.get_questions()) < args.max_questions:
        patient_state = patient_system.get_state()
        response_dict = expert_system.respond(patient_state)

        # The expert_system's respond method now returns a dictionary
        response_type = response_dict["type"]
        response_content = response_dict["content"]
        # Optional return values for analysis, e.g., confidence score, logprobs
        temp_additional_info.append({k: v for k, v in response_dict.items() if k not in ["type", "content"]})

        if response_type == "question":
            temp_choice = expert_system.choice(patient_state)  # Expert makes a choice based on the current state
            temp_choice_list.append(temp_choice)  # Log the question as an intermediate choice
            patient_system.respond(response_content)  # Patient generates an answer based on the last question asked

        elif response_type == "choice":
            temp_choice_list.append(response_content)
            return response_content, patient_system.get_questions(), patient_system.get_answers(), temp_choice_list, temp_additional_info
        
        else:
            raise ValueError("Invalid response type from expert_system.")
        
    # If max questions are reached and no final decision has been made
    response_dict = expert_system.choice(patient_system.get_state())
    stuck_response = response_dict["chocie"]
    # Optional return values for analysis, e.g., confidence score, logprobs
    temp_additional_info.append({k: v for k, v in response_dict.items() if k != "chocie"})
    
    return stuck_response, patient_system.get_questions(), patient_system.get_answers(), temp_choice_list + [stuck_response], temp_additional_info


if __name__ == "__main__":
    args = get_args()
    main()
