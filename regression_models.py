from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor


def linear_regression_model():
    lin_reg_params = {
        "LinearRegression__fit_intercept ": [True],
        "LinearRegression__normalize": [False]
    }

    pipe = Pipeline([
        ('LinearRegression', LinearRegression())
    ])

    # Initiating GridSearchCV
    log_reg_grid = GridSearchCV(estimator=pipe, param_grid=lin_reg_params, cv=StratifiedKFold(5), refit=True)
    return log_reg_grid


def decision_tree_regression_model(random_state_param):
    regression_tree_params = {
        "DecisionTreeRegressor__criterion": ["mse"],
        "DecisionTreeRegressor_splitter": ["best"],
        "DecisionTreeRegressor__max_depth": range(3, 15, 1),
        "DecisionTreeRegressor__min_samples_split": [2],
        "DecisionTreeRegressor__min_samples_leaf": [1],
        "DecisionTreeRegressor__max_features": ["auto", "sqrt"],
        "DecisionTreeRegressor__random_state": [random_state_param],
        "DecisionTreeRegressor__presort": [False]
    }

    pipe = Pipeline([
        ('DecisionTreeRegressor', DecisionTreeRegressor())
    ])

    dtr_grid = GridSearchCV(estimator=pipe, param_grid=regression_tree_params, cv=StratifiedKFold(5), refit=True)
    return dtr_grid


def random_forest_regression_model(random_state_param):
    # Random forest parameters
    rfc_params = {
        "RandomForestRegressor__n_estimators": range(80, 200, 10),
        "RandomForestRegressor__criterion": ["mse"],
        "RandomForestRegressor__max_depth": range(3, 9, 1),
        "RandomForestRegressor__min_samples_split": [2],
        "RandomForestRegressor__min_samples_leaf": [1],
        "RandomForestRegressor__max_features": ["auto", "sqrt"],
        "RandomForestRegressor__bootstrap": [True],
        "RandomForestRegressor__oob_score": [False],
        "RandomForestRegressor__random_state": [random_state_param],
        "RandomForestRegressor__warm_start": [False]
    }

    pipe = Pipeline([
        ('RandomForestRegressor', RandomForestRegressor())
    ])

    rfr_grid = RandomizedSearchCV(estimator=pipe, param_distributions=rfc_params, cv=StratifiedKFold(5), refit=True)
    return rfr_grid


def mlp_regression_model(random_state_param):
    mlp_regressor_params = {
        "MLPRegressor__hidden_layer_sizes": [(36, 16, 8)],  # [(100,), (18, 8, 4), (24, 10, 4)],
        "MLPRegressor__activation": ['relu'],
        "MLPRegressor__solver": ['adam'],  # ["sgd", "lbfgs"], 
        "MLPRegressor__alpha": [0.0001, 0.01],  # [0.0001, 0.05],
        "MLPRegressor__batch_size": [40, 50, 60],
        "MLPRegressor__learning_rate": ['constant'],
        "MLPRegressor__shuffle": [True],
        "MLPRegressor__max_iter": [200, 250, 300],  # [50, 100, 150],
        "MLPRegressor__random_state": [random_state_param],
        "MLPRegressor__verbose": [1],
        "MLPRegressor__early_stopping": [True]
    }

    pipe = Pipeline([
        ('MLPRegressor', MLPRegressor())
    ])

    mlpr_grid = RandomizedSearchCV(estimator=pipe, param_distributions=mlp_regressor_params, cv=StratifiedKFold(5), refit=True)
    return mlpr_grid


def svr_model():
    svr_params = {
        'SVR__kernel': ['sigmoid', 'rbf'],
        'SVR__degree': [2, 3],
        'SVR__gamma': [0.001, 0.01, 0.1],
        'SVR__C': [0.7],
        'SVR__verbose': [True],
        'SVR__max_iter': [-1],
    }

    pipe = Pipeline([
        ('SVR', SVR())
    ])

    svr_grid = GridSearchCV(estimator=pipe, param_grid=svr_params, cv=StratifiedKFold(5), refit=True)
    return svr_grid


def stochastic_gradient_descent_regression_model(random_state_param):
    sgd_params = {
        "SGDRegressor__loss": ["squared_loss", "huber"],
        "SGDRegressor__penalty": ["none", "l2", "l1", "elasticnet"],
        "SGDRegressor__alpha": [0.0001],
        "SGDRegressor__fit_intercept": [True],
        "SGDRegressor__l1_ratio": [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.25],
        "SGDRegressor__max_iter": [200, 300, 500, 1000],
        "SGDRegressor__shuffle": [True],
        "SGDRegressor__verbose": [1],
        "SGDRegressor__random_state": [random_state_param],
        "SGDRegressor__learning_rate": ["constant", "adaptive"],
        "SGDRegressor__eta0": [0.1],
        "SGDRegressor__early_stopping": [True],
    }

    pipe = Pipeline([
        ("SGDRegressor", SGDRegressor())
    ])

    sgd_grid = GridSearchCV(estimator=pipe, param_grid=sgd_params, cv=StratifiedKFold(5), refit=True)
    return sgd_grid


def ada_boost_regression_model(random_state_param):
    ada_params = {
        "AdaBoostRegressor__base_estimator": [
            DecisionTreeRegressor(random_state=random_state_param, max_depth=2),
            DecisionTreeRegressor(random_state=random_state_param, max_depth=3),
            RandomForestRegressor(random_state=random_state_param, max_depth=2),
            RandomForestRegressor(random_state=random_state_param, max_depth=3)],
        "AdaBoostRegressor__n_estimators": [25, 50, 75],
        "AdaBoostRegressor__learning_rate": [0.5, 0.7, 1.0],
        "AdaBoostRegressor__loss": ["linear"],  # ["linear", "square", "exponential"],
        "AdaBoostRegressor__random_state": [random_state_param],
    }

    pipe = Pipeline([
        ("AdaBoostRegressor", AdaBoostRegressor())
    ])

    ada_grid = GridSearchCV(estimator=pipe, param_grid=ada_params, cv=StratifiedKFold(5), refit=True)
    return ada_grid


def xgb_regression_model(random_state_param):
    xgb_params = {
        "XGBRegressor__max_depth": [2, 4, 6],
        "XGBRegressor__learning_rate": [0.05, 0.07],
        "XGBRegressor__seed": [random_state_param],
        "XGBRegressor__n_estimators": [100, 400, 800],
        "XGBRegressor__subsample": [0.5, 0.7, 1],
        "XGBRegressor__alpha": [0.7, 1],
        "XGBRegressor__eval_metric": ["rmse", "mae"]
    }

    pipe = Pipeline([
        ("XGBRegressor", XGBRegressor())
    ])

    xgb_grid = GridSearchCV(estimator=pipe, param_grid=xgb_params, cv=5, refit=True)
    return xgb_grid


def linear_svr_model(random_state_param):
    linear_svr_params = {
        "LinearSVR__loss ": ["epsilon_insensitive", "squared_epsilon_insensitive"],
        "LinearSVR__dual": [False],
        "LinearSVR__C": [0.1, 0.5, 1.0],
        "LinearSVR__fit_intercept ": [True],
        "LinearSVR__verbose": [1],
        "LinearSVR__random_state": [random_state_param]
    }

    pipe = Pipeline([
        ("LinearSVR", LinearSVR())
    ])

    linear_svr_grid = GridSearchCV(estimator=pipe, param_grid=linear_svr_params, cv=5, refit=True)
    return linear_svr_grid


def select_model(option, random_state):
    models_dictionary = {
        "linear_regression": linear_regression_model(),
        "regression_tree": decision_tree_regression_model(random_state_param=random_state),
        "random_forest_regressor": random_forest_regression_model(random_state_param=random_state),
        "mlp_regressor": mlp_regression_model(random_state_param=random_state),
        "svr": svr_model(),
        "sgd_regressor": stochastic_gradient_descent_regression_model(random_state_param=random_state),
        "ada_boost_regressor": ada_boost_regression_model(random_state_param=random_state),
        "xgb_regressor": xgb_regression_model(random_state_param=random_state),
        "linear_svr": linear_svr_model(random_state_param=random_state),
    }

    for model_name, model_grid in models_dictionary.items():
        try:
            if option == model_name:
                return model_grid
        except ValueError:
            print("Model nie istnieje.")
            quit()
