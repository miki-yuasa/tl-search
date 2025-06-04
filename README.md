# tl-search
Implementing TL Search algorithm for CtF from paper "[On Generating Explanations for Reinforcement Learning Policies: An Empirical Study](https://ieeexplore.ieee.org/abstract/document/10804622)".

## Installation
### Other packages
Run `poetry install`.
Refer to [poetry documentation](https://python-poetry.org/docs/) for more details.

### Notes
- When installing `pygame`, you may need to install `libsdl2-dev` and `libsdl2-image-dev` on your system.
Refer to [pygame installation guide](https://www.pygame.org/wiki/CompileUbuntu?parent=) for more details.

## Usage
### Organization
The directory structure is as follows:
```
- assets/ # contains CtF map files
- scripts/
    - archive/ # contains the archived scripts
    - plotting/ # contains the plotting scripts
    - utils/ # contains the utility scripts
    - main_*.py # main scripts for running the experiments
- tests/ # contains the test scripts
- tl-search/ # contains the main implementation of TL Search
    - envs/ # contains the environments
        - tl_*.py # contains the target policy and the TL reward function
    - search/ # contains the search algorithms
    - tl/ # contains the TL parsing and reward utilities
    - train/ # contains the training algorithms

```

### Scripts dir
- `main_tl_search_<env_name>.py`: main script for running TL Search on the specified environment.
- `main_tl_exhaustive_<env_name>.py`: main script for training all the policies for all the candidate explanations in the specified environment.
- `main_train_<env_name>.py`: main script for training the target policy with the normal reward function in the specified environment.
- `main_train_tl_<env_name>.py`: main script for training the target policy with the TL reward function in the specified environment.
- `main_tl_simulate_<env_name>.py`: main script for simulating the a policy in the specified environment.
- `main_eval_<env_name>.py`: main script for evaluating the target policy in the specified environment.

### Process
1. Train the target policy with the normal reward function / TL reward using `main_train_<env_name>.py`/`main_train_tl_<env_name>.py`.
2. Optionally, run `main_tl_exhaustive_<env_name>.py` to train all the policies for all the candidate explanations.
3. Run `main_tl_search_<env_name>.py` to run TL Search on the target policy.