# Benchmark System for Patient-Expert Interaction

## Overview
This benchmark system simulates an interactive conversation between a patient and an expert. The system evaluates how well participants' expert modules can handle realistic patient queries by either asking relevant questions or making final decisions based on the conversation history.

## Requirements
- Python 3.8 or higher
- Necessary Python libraries: json, os, logging, importlib

## Installation
Clone this repository to your local machine using the following command:
```
git clone https://github.com/stellali7/MediQ.git
```

Navigate into the project directory:
```
cd MediQ
```

Install the required Python libraries (if not already installed):
```
pip install -r requirements.txt
```


## Project Structure
- `benchmark.py`: Main script to run the benchmark.
- `patient.py`: Defines the `Patient` class that simulates patient behavior.
- `expert.py`: Contains the `Expert` class which participants will extend to implement their response strategies.
- `args.py`: Handles command-line arguments for the benchmark system.

## Configuration
Before running the benchmark, configure the necessary parameters in `args.py`:
- `--expert_module`: The module name where the participant's expert class is implemented.
- `--expert_class`: The name of the expert class to use for the benchmark.
- `--data_dir`: Directory containing the development data files.
- `--dev_filename`: Filename for development data.
- `--log_filename`: Filename for logging general benchmark information.
- `--history_log_filename`: Filename for logging detailed interaction history.
- `--message_log_filename`: Filename for logging messages.
- `--output_filepath`: Path where the output JSONL files will be saved.

## Running the Benchmark
To run the benchmark, use the following command:
```
python benchmark.py --expert_module 'your_expert_module' --expert_class 'YourExpertClassName' \
                    --data_dir 'path_to_data_directory' --dev_filename 'dev_data.jsonl' \
                    --log_filename 'benchmark.log' --history_log_filename 'history.log' --message_log_filename 'messages.log' \
                    --output_filepath 'output.jsonl'
```

Ensure to replace the placeholder values with actual parameters relevant to your setup.

## How to Participate
Participants are expected to create their own `Expert` class within a module specified by `--expert_module`. The class should correctly implement the `respond` method to interact with the `Patient` instances based on their states. The response should either be a continuation question or a final decision. Your implementation will be tested against a variety of patient scenarios provided in the development dataset.

## Example
Here is an example of a simple `Expert` implementation:
