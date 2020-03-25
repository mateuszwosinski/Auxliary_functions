import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

def plot_results(y_train, train_scores, y_test, test_scores, title, directory,
                 model_best_params, model_best_estimator, train_cvs, test_cvs,
                 scorings):
    
    df_train = pd.DataFrame(data=np.transpose([np.array(y_train),train_scores]),
                            columns=['actual','predict'],
                            index=y_train.index).sort_values('actual')
    
    df_test = pd.DataFrame(data=np.transpose([np.array(y_test),test_scores]),
                           columns=['actual','predict'],
                           index=y_test.index).sort_values('actual')
    
    train_index = list(range(0, y_train.shape[0]))
    test_index = list(range(0, y_test.shape[0]))
    
    plt.figure(figsize=(10, 8))
    plt.suptitle(title)
    
    # plots for actual and predicted values
    plt.subplot(121)
    plt.scatter(train_index, df_train['actual'], color = "red")
    plt.scatter(train_index, df_train['predict'], color = "green")
    plt.title("train")
    plt.xlabel("user id")
    plt.show()

    plt.subplot(122)
    plt.scatter(test_index, df_test['actual'], color = "red")
    plt.scatter(test_index, df_test['predict'], color = "green")
    plt.title("test")
    plt.xlabel("user id")
    plt.show()
    
    plt.tight_layout()
    plt.savefig(directory + '/' + title + '_scatter_plot.png')
    
    with open(directory + '/' + title + '_analysis_results.txt', 'a+') as file:
        file.write("\nBest parameters:\n" + str(model_best_params) + "\n")
        file.write("\nBest estimator:\n" + str(model_best_estimator) + "\n")
        for ind, train_cv in enumerate(train_cvs):
            file.write("\nTRAIN CROSS VAL SCORE\nScoring method: " + scorings[ind] + "\n" + str(train_cv) + "\n")
        for ind, test_cv in enumerate(test_cvs):
            file.write("\nTEST CROSS VAL SCORE\nScoring method: " + scorings[ind] + "\n" + str(test_cv) + "\n")

    # Save model
    pickle.dump(model_best_estimator, open(directory + '/' + title + '_best_estimator.sav', 'wb'))

def train_model(cv_grid, X_train, y_train, X_test, y_test, dir_path, model,
                random_state):
    
    # Fit model to train set
    model_grid = cv_grid.fit(X_train, y_train)

    model_best_params = model_grid.best_params_

    model_best_estimator = model_grid.best_estimator_

    # Train and test cross validation scores for each fold
    # 3 different scoring methods
    
    scorings=['neg_mean_absolute_error','neg_mean_squared_error','r2']
    train_cross_val_scores=[]
    test_cross_val_scores=[]
    
    for scoring in scorings:
        train_cross_val_scores.append(cross_val_score(estimator=model_best_estimator,
                                                      X=X_train, y=y_train, cv=5,
                                                      scoring=scoring)) 
        test_cross_val_scores.append(cross_val_score(estimator=model_best_estimator,
                                             X=X_test, y=y_test, cv=5,
                                             scoring=scoring))

    # Make prediction on train set
    model_train_proba = model_best_estimator.predict(X_train)

    # Make prediction on test set
    model_test_proba = model_best_estimator.predict(X_test)
    
    plot_results(y_train=y_train, train_scores=model_train_proba, y_test=y_test,
                 test_scores=model_test_proba, title=f"{model}", directory=dir_path,
                 model_best_params=model_best_params, model_best_estimator=model_best_estimator,
                 train_cvs=train_cross_val_scores, test_cvs=test_cross_val_scores,
                 scorings=scorings)
    
    

def encode_variable(data_frame, variable, fill_value):
    data_frame[variable] = data_frame[variable].fillna(fill_value)
    le = LabelEncoder()
    data_frame[variable] = le.fit_transform(data_frame[variable])
    return data_frame