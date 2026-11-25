# cpsc322-final-project
Connor and Felix's final project.

## Data cleaning

Raw data: a snapshot of [olbauday/FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) in `Inputs/data/`.

* `data_prep.py`: all cleaning steps, with comments explaining each decision. Run `python3 data_prep.py` to rebuild `clean_data/`.
* `data_cleaning.ipynb`: walks through each step, shows the results, and runs the checks behind our decisions.
* `mysklearn/`: `MyPyTable` and helper functions reused from our programming assignments.
* `clean_data/`: cleaned tables. `player_seasons.csv` is for the position classifier; `match_features.csv` is for the home-win classifier.
