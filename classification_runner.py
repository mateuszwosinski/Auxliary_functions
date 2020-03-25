from classification_models import *
from classification_functions import *

import itertools
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


RANDOM_STATE_PARAM = 0
MAIN_DIRECTORY = ''

parameters_combinations = [
    ["logistic_regression"],  # models
    ["smote"],  # type of  sampling method, to run without sampling method parameter must be ONLY None
    ["not majority"],  # type of sampling option, to run without sampling method parameter must be ONLY None
    [RANDOM_STATE_PARAM]  # random state param
]

if not os.path.exists(f"{MAIN_DIRECTORY}"):
    os.mkdir(f"{MAIN_DIRECTORY}")


x = df[features]
y = df[target]

scaler = MinMaxScaler()
scaler.fit(x.values)

x = pd.DataFrame(scaler.transform(x.values), index=x.index, columns=x.columns)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=RANDOM_STATE_PARAM)

for combination_tuple in list(itertools.product(*parameters_combinations)):
    if combination_tuple.count(None) != 1:

        model, technique, method, random_state_param = combination_tuple[0], combination_tuple[1], combination_tuple[2], combination_tuple[3]

        PATH = f"{model}_{technique}_{method}"

        if not os.path.exists(f"{MAIN_DIRECTORY}/{PATH}"):
            os.mkdir(f"{MAIN_DIRECTORY}/{PATH}")

        model_grid = select_model(option=model, random_state=random_state_param)
        print(f"\n{model} -- {technique} -- {method}\n")

        best_model = train_model(cv_grid=model_grid, X_train=x_train, y_train=y_train,
                             X_test=x_test, y_test=y_test, dir_path=f"{MAIN_DIRECTORY}/{PATH}",
                             model=str(model),random_state=random_state_param,
                             sample_technique=technique, sample_type=method)
        


