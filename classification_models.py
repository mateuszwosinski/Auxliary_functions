from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier


def logistic_regression_model(random_state_param):
    log_reg_params = {
        "LogisticRegression__penalty": ['l2'], #l2 norm
        "LogisticRegression__C": [0.01, 0.1, 1, 10], 
        # "LogisticRegression__class_weight": ["balanced"],
        "LogisticRegression__random_state": [random_state_param],
        "LogisticRegression__max_iter": range(40, 120, 10),
        "LogisticRegression__verbose": [1],
        "LogisticRegression__multi_class": ["ovr"],
        "LogisticRegression__solver": ["newton-cg"],
        "LogisticRegression__n_jobs": [-1]
    }

    pipe = Pipeline([
        ('LogisticRegression', LogisticRegression())
    ])

    # Initiate GridSearchCV
    log_reg_grid = GridSearchCV(estimator=pipe, param_grid=log_reg_params, cv=StratifiedKFold(5), refit=True)
    return log_reg_grid


def decision_tree_model(random_state_param):
    decision_tree_params = {
        "DecisionTreeClassifier__max_depth": range(4, 9),
        "DecisionTreeClassifier__splitter": ["best"],
        "DecisionTreeClassifier__min_samples_split": [0.05, 0.1, 0.15],
        "DecisionTreeClassifier__min_samples_leaf": [0.05, 0.1, 0.15],
        "DecisionTreeClassifier__max_features": ["auto", "log2", 0.75],
        "DecisionTreeClassifier__random_state": [random_state_param],
        # "DecisionTreeClassifier__class_weight": ["balanced"],
        "DecisionTreeClassifier__presort": [True]
    }

    pipe = Pipeline([
        ('DecisionTreeClassifier', DecisionTreeClassifier())
    ])

    # Initiate GridSearchCV
    dtc_grid = GridSearchCV(estimator=pipe, param_grid=decision_tree_params, cv=StratifiedKFold(5), refit=True)
    return dtc_grid


def random_forest_model(random_state_param):
    # Random forest parameters
    rfc_params = {
        'RandomForestClassifier__n_estimators': range(50, 150, 20),  # range(100, 260, 10),  # range(40, 180, 10),  # range(200, 800, 100),  # range(30, 110, 10),
        'RandomForestClassifier__max_depth': range(4, 9),  # range(2, 15, 2),
        "RandomForestClassifier__min_samples_split": [0.05, 0.1, 0.15],
        "RandomForestClassifier__min_samples_leaf": [0.05, 0.1, 0.15],
        'RandomForestClassifier__max_features': ['auto', None],
        'RandomForestClassifier__bootstrap': [True],
        'RandomForestClassifier__random_state': [random_state_param],
        # 'RandomForestClassifier__class_weight': ['balanced'],
        'RandomForestClassifier__verbose': [1]
    }

    pipe = Pipeline([
        ('RandomForestClassifier', RandomForestClassifier())
    ])

    rfc_grid = RandomizedSearchCV(estimator=pipe, param_distributions=rfc_params, cv=StratifiedKFold(5), refit=True)

    return rfc_grid


def mlp_classifier_model(random_state_param):
    mlp_classifier_params = {
        "MLPClassifier__hidden_layer_sizes": [(36, 16, 8)],  # [(100,), (18, 8, 4), (24, 10, 4)],
        "MLPClassifier__activation": ['relu'],
        "MLPClassifier__solver": ['adam'],
        "MLPClassifier__alpha": [0.001, 0.01],  # [0.0001, 0.05],
        "MLPClassifier__batch_size": [50, 60, 80],
        "MLPClassifier__learning_rate": ['constant'],
        "MLPClassifier__max_iter": [200, 250, 300],  # [50, 100, 150],
        "MLPClassifier__random_state": [random_state_param],
        "MLPClassifier__verbose": [1],
        "MLPClassifier__early_stopping": [True]
    }

    pipe = Pipeline([
        ('MLPClassifier', MLPClassifier())
    ])

    rfc_grid = RandomizedSearchCV(estimator=pipe, param_distributions=mlp_classifier_params, cv=StratifiedKFold(5), refit=True)
    return rfc_grid


def naive_bayesian_model():
    nb_params = {
        "MultinomialNB__alpha": [0.1, 0.25, 0.5, 0.75, 1.0]
    }

    pipe = Pipeline([
        ('MultinomialNB', MultinomialNB())
    ])

    nb_grid = GridSearchCV(estimator=pipe, param_grid=nb_params, cv=StratifiedKFold(5), refit=True)
    return nb_grid


def svc_model(random_state_param):
    svc_params = {
        'SVC__C': [0.7],
        'SVC__kernel': ['sigmoid', 'rbf'],
        'SVC__degree': [2],
        'SVC__gamma': [0.001, 0.01, 0.1],
        # 'SVC__class_weight': ["balanced"],
        'SVC__verbose': [True],
        'SVC__random_state': [random_state_param],
        "SVC__probability": [True]
    }

    pipe = Pipeline([
        ('SVC', SVC())
    ])

    svc_grid = GridSearchCV(estimator=pipe, param_grid=svc_params, cv=StratifiedKFold(5), refit=True)
    return svc_grid


def knn_model():
    knn_params = {
        "KNeighborsClassifier__n_neighbors": [10],
        "KNeighborsClassifier__weights": ["uniform", "distance"],
        "KNeighborsClassifier__algorithm": ["kd_tree", "brute"],
        "KNeighborsClassifier__leaf_size": [30, 40],
    }

    pipe = Pipeline([
        ('KNeighborsClassifier', KNeighborsClassifier())
    ])

    knn_grid = GridSearchCV(estimator=pipe, param_grid=knn_params, cv=StratifiedKFold(5), refit=True)
    return knn_grid


def stochastic_gradient_descent_model(random_state_param):
    sgd_params = {
        "SGDClassifier__loss": ["log", "squared_hinge"],
        "SGDClassifier__penalty": ["l2", "l1"],
        "SGDClassifier__l1_ratio": [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.25],
        "SGDClassifier__max_iter": [200, 300, 500, 1000],
        "SGDClassifier__shuffle": [True],
        "SGDClassifier__verbose": [1],
        "SGDClassifier__random_state": [random_state_param],
        "SGDClassifier__learning_rate": ["constant", "adaptive"],
        "SGDClassifier__eta0": [0.1],
        "SGDClassifier__early_stopping": [True],
        "SGDClassifier__class_weight": ["balanced"]
    }

    pipe = Pipeline([
        ("SGDClassifier", SGDClassifier())
    ])

    sgd_grid = GridSearchCV(estimator=pipe, param_grid=sgd_params, cv=StratifiedKFold(5), refit=True)
    return sgd_grid


def ada_boost_model(random_state_param):
    ada_params = {
        "AdaBoostClassifier__base_estimator": [
            DecisionTreeClassifier(random_state=random_state_param, max_depth=1, class_weight="balanced"),
            DecisionTreeClassifier(random_state=random_state_param, max_depth=2, class_weight="balanced"),
            DecisionTreeClassifier(random_state=random_state_param, max_depth=3, class_weight="balanced")],
        "AdaBoostClassifier__n_estimators": [25, 50, 75],
        "AdaBoostClassifier__learning_rate": [0.7, 1.0],
        "AdaBoostClassifier__algorithm": ["SAMME.R"],
        "AdaBoostClassifier__random_state": [random_state_param],
    }

    pipe = Pipeline([
        ("AdaBoostClassifier", AdaBoostClassifier())
    ])

    ada_grid = GridSearchCV(estimator=pipe, param_grid=ada_params, cv=StratifiedKFold(5), refit=True)
    return ada_grid


def xgb_classifier_model(random_state_param):
    xgb_params = {
        "XGBClassifier__random_state": [random_state_param],
        "XGBClassifier__eta": [0.05, 0.1, 0.15, 0.2],
        "XGBClassifier__max_depth": range(4, 9),
        "XGBClassifier__subsample": [0.7, 0.9],
        "XGBClassifier__colsample_bytree": [0.7, 0.9]
    }

    pipe = Pipeline([
        ("XGBClassifier", XGBClassifier())
    ])

    xgb_grid = GridSearchCV(estimator=pipe, param_grid=xgb_params, cv=5, refit=True)
    return xgb_grid


def linear_svc_model(random_state_param):
    linear_svc_params = {
        "LinearSVC__penalty": ["l1", "l2"],
        "LinearSVC__loss ": ["hinge"],
        "LinearSVC__dual": ["False"],
        "LinearSVC__C": [0.1, 0.5, 1.0],
        "LinearSVC__multi_class": ["ovr"],
        "LinearSVC__class_weight": ["balanced"],
        "LinearSVC__verbose": [1],
        "LinearSVC__random_state": [random_state_param]
    }

    pipe = Pipeline([
        ("LinearSVC", LinearSVC())
    ])

    linear_svc_grid = GridSearchCV(estimator=pipe, param_grid=linear_svc_params, cv=5, refit=True)
    return linear_svc_grid


def select_model(option, random_state):
    models_dictionary = {
        "logistic_regression": logistic_regression_model(random_state_param=random_state),
        "decision_tree": decision_tree_model(random_state_param=random_state),
        "random_forest": random_forest_model(random_state_param=random_state),
        "mlp_classifier": mlp_classifier_model(random_state_param=random_state),
        "naive_bayes": naive_bayesian_model(),
        "svc": svc_model(random_state_param=random_state),
        "knn": knn_model(),
        "sgd_classifier": stochastic_gradient_descent_model(random_state_param=random_state),
        "ada_boost": ada_boost_model(random_state_param=random_state),
        "xgb_classifier": xgb_classifier_model(random_state_param=random_state),
        "linear_svc": linear_svc_model(random_state_param=random_state),
    }

    for model_name, model_grid in models_dictionary.items():
        try:
            if option == model_name:
                return model_grid
        except ValueError:
            print("Model nie istnieje.")
            quit()

