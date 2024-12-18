import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Run the benchmark with specified configurations.")
    parser.add_argument('--expert_module', type=str, default='expert', help='file name where the expert class is implemented.')
    parser.add_argument('--expert_class', type=str, required=True, help='Expert class name to use for the benchmark.')
    parser.add_argument('--expert_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Expert model name to use for the benchmark, can be a local model or a Huggingface model.')
    parser.add_argument('--patient_module', type=str, default='expert', help='file name where the patient class is implemented.')
    parser.add_argument('--patient_class', type=str, required=True, help='Patient class name to use for the benchmark.')
    parser.add_argument('--patient_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Patient model name to use for the benchmark, can be a local model or a Huggingface model.')

    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the development data files.')
    parser.add_argument('--dev_filename', type=str, required=True, help='Filename for development data.')

    parser.add_argument('--output_filename', type=str, default="results.jsonl")

    parser.add_argument("--max_questions", type=int, default=30)

    parser.add_argument('--log_filename', type=str, default='log.log', help='Filename for logging general benchmark info.')
    parser.add_argument('--history_log_filename', type=str, default='history_log.log', help='Filename for logging detailed history.')
    parser.add_argument('--message_log_filename', type=str, default='message_log.log', help='Filename for logging messages.')

    parser.add_argument('--rationale_generation', action='store_true', help='Generate rationales for the choices.')
    parser.add_argument('--self_consistency', type=int, default=1, help='Number of times to run the self-consistency check.')
    parser.add_argument('--abstain_threshold', type=float, default=0.8, help='Threshold for abstaining from making a choice.')
    parser.add_argument('--independent_modules', action='store_true', help='Cognitive modules within the Expert dont see previous convo.')

    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for sampling from the model.')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top p value for nucleus sampling.')
    parser.add_argument('--max_tokens', type=int, default=256, help='Maximum number of tokens to generate.')
    parser.add_argument('--top_logprobs', type=int, default=0, help='Number of top logprobs to return.')
    parser.add_argument('--api_account', type=str, default="mediQ", help='API keys are stored in keys.py, api_account is the name of the key.')
    args =  parser.parse_args()

    return args