# MediQ: Question-Asking LLMs for Adaptive and Reliable Clinical Reasoning

## [[paper](https://arxiv.org/abs/2406.00922)] [[website](https://stellalisy.com/projects/mediQ/)] [[data](https://github.com/stellali7/mediQ/tree/main/data)]

## Overview
This benchmark system simulates an interactive conversation between a patient and an expert. The system evaluates how well participants' expert modules can handle realistic patient queries by either asking relevant questions or making final decisions based on the conversation history.

## Installation
Clone this repository to your local machine using the following command:
```
git clone https://github.com/stellali7/MediQ.git
```

Navigate into the project directory:
```
cd MediQ
```

Create a new conda environment with necessary packages (note: you need to be on a GPU node to install PyTorch with CUDA):
```
conda env create -f environment.yml
```


## Project Structure
- `benchmark.py`: Main script to run the benchmark.
- `patient.py`: Defines the `Patient` class that simulates patient behavior.
- `expert.py`: Contains the `Expert` class which participants will extend to implement their response strategies.
- `args.py`: Handles command-line arguments for the benchmark system.

## Configuration
Before running the benchmark, configure the necessary parameters in `args.py`:
- `--expert_module`: The file name (without `.py`) where the Expert class is implemented (e.g. expert if your Expert class definition is in `expert.py`)
- `--expert_class`: The name of the Expert class to be evaluated, this should be defined in the file `[expert_module].py` (e.g. RandomExpert)
- `--patient_module`: The file name (without `.py`) where the Patient class is implemented (e.g. patient if your Patient class definition is in `patient.py`)
- `--patient_class`: The name of the Patient class to use for the benchmark, this should be defined in the file `[patient_module].py` (e.g. RandomPatient)
- `--data_dir`: Directory containing the development data files.
- `--dev_filename`: Filename for development data.
- `--log_filename`: Filename for logging general benchmark information.
- `--history_log_filename`: Filename for logging detailed interaction history.
- `--message_log_filename`: Filename for logging messages.
- `--output_filepath`: Path where the output JSONL files will be saved.

## Running the Benchmark
NOTE: if you choose to use an OpenAI model to power the benchmark, you need to put the API key in `src/keys.py`.

To test run the benchmark, use the following command (note: the Patient system is provided as described in the paper, the Expert system is a skeleton code. For a fast test run, use `--patient_variant random` to not call use any actual model or API):
```
python mediQ_benchmark.py  --expert_module expert --expert_class FixedExpert \
                        --patient_module patient --patient_class RandomPatient \
                        --data_dir ../data --dev_filename all_dev_good.jsonl \
                        --output_filename out.jsonl --max_questions 10
```

Ensure to replace the placeholder values with actual parameters relevant to your setup.

## Try out your own Expert system
You can easily create their own `Expert` class within a module specified by `--expert_module`, or old a different model by specifying the model path in `--expert_model`. The class should correctly implement the `respond` method to interact with the `Patient` instances based on their states (the Patient can be customized as well). The response should either be a continuation question or a final decision. Your implementation will be tested against a variety of patient scenarios provided in the development dataset.

## How to Cite
```
@inproceedings{li2024mediq,
        title={MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning},
        author={Li, Shuyue Stella and Balachandran, Vidhisha and Feng, Shangbin and Ilgen, Jonathan S and Pierson, Emma and Koh, Pang Wei and Tsvetkov, Yulia},
        journal={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
        year={2024}
      }
```

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
