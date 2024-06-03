import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# from mydifflib import get_close_matches

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
print(f"Device: {device}")

emb_model = SentenceTransformer('stsb-roberta-large', device=device)

def eval_sample(id, sample, choice, scores, questions, answers, answer_dne, temp_choice_list, threshold=0.85):
    questions_emb = emb_model.encode(questions)
    facts_emb = emb_model.encode(sample["facts"])
    facts_count = [0]*len(sample["facts"])
    answers_expanded, answers_count = [], []
    
    for answer in answers:
        answer = [a for a in answer.split('. ') if not a.isnumeric()]  # split the answer into atomic facts
        answers_expanded.extend(answer)
        answers_count.append(len(answer))
    answers_emb = emb_model.encode(answers_expanded)

    output_dict = {
        "id": id,
        "info": sample,
        "interactive_system": {
            "choice": choice,
            "confidence_scores": scores,
            "questions": questions,
            "answers": answers,
            "answer_dne": answer_dne,
            "num_questions": len(questions),
            "intermediate_choices": temp_choice_list,
        },
        "eval": {
            "repeat_question_score": [],
            "repeat_answer_score": [],
            "relevancy_score": [],
            "delta_confidence_score": [],
            "specificity_score": []
        }
    }
    
    # Example placeholder for evaluation metrics computation
    for i in range(len(questions)):
        output_dict["eval"]["repeat_question_score"].append(np.random.random())  # Placeholder
        output_dict["eval"]["repeat_answer_score"].append(np.random.random())  # Placeholder
        output_dict["eval"]["relevancy_score"].append(np.random.random())  # Placeholder
        output_dict["eval"]["delta_confidence_score"].append(np.random.random())  # Placeholder
        output_dict["eval"]["specificity_score"].append(np.random.random())  # Placeholder

    return output_dict

# Other functions should be similarly reviewed and implemented
