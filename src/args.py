import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Run the benchmark with specified configurations.")
    parser.add_argument('--expert_module', type=str, default='expert', help='Module name where the expert class is implemented.')
    parser.add_argument('--expert_class', type=str, required=True, help='Expert class name to use for the benchmark.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the development data files.')
    parser.add_argument('--dev_filename', type=str, required=True, help='Filename for development data.')

    parser.add_argument('--output_filename', type=str, default="results.jsonl")

    parser.add_argument("--max_questions", type=int, default=30)

    parser.add_argument('--log_filename', type=str, default='log.log', help='Filename for logging general benchmark info.')
    parser.add_argument('--history_log_filename', type=str, default='history_log.log', help='Filename for logging detailed history.')
    parser.add_argument('--message_log_filename', type=str, default='message_log.log', help='Filename for logging messages.')

    parser.add_argument('--patient_variant', type=str, default='fact_select', help='Patient variant used to respond to expert questions')
    args =  parser.parse_args()

    return args