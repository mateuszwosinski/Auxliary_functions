import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

import plot_functions


def plot_results(y_train, train_scores, y_test, test_scores, title, directory,
                 model_best_params, model_best_estimator, features, train_cvs, test_cvs):
    # Set plot size
    plt.figure(figsize=(10, 8))
    plt.suptitle(title)
    
    # Confusion matrices for train and test sets
    plt.subplot(121)
    train_conf_mat = pd.crosstab(y_train, train_scores.round(),
                                 rownames=['Actual'], colnames=['Predicted'],
                                 margins=True)
    sns.heatmap(data=train_conf_mat, annot=True, fmt='d', cmap='YlGnBu')
    plt.title("Train", fontsize=15)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.subplot(122)
    test_conf_mat = pd.crosstab(y_test, test_scores.round(),
                                rownames=['Actual'], colnames=['Predicted'],
                                margins=True)
    sns.heatmap(data=test_conf_mat, annot=True, fmt='d', cmap='YlGnBu')
    plt.title("Test", fontsize=15)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    
    plt.tight_layout()
    plt.savefig(directory + '/' + title + '_confusion_matrix.png')
    
    # target_names = np.array(sorted(y_train.unique())).astype(str)
    train_report = classification_report(y_train, train_scores.round())
    test_report = classification_report(y_test, test_scores.round())
      
    with open(directory + '/' + title + '_analysis_results.txt', 'a+') as file:
        file.write("\nClassification report train: \n" + str(train_report) + "\n")
        file.write("\nClassification report test: \n" + str(test_report) + "\n")
        file.write("\nBest parameters:\n" + str(model_best_params) + "\n")
        file.write("\nBest estimator:\n" + str(model_best_estimator) + "\n")
        file.write("\nTRAIN CROSS VAL SCORE\n" + str(train_cvs) + "\n")
        file.write("\nTEST CROSS VAL SCORE\n" + str(test_cvs) + "\n")

    # Save model
    pickle.dump(model_best_estimator, open(directory + '/' + title + '_best_estimator.sav', 'wb'))


def save_odd_ratios(directory, title, features, best_coef):
    
    best_odd_ratio = round(np.exp(best_coef),4)
    best_coef = round(best_coef,4)
    
    with open(directory + '/' + title + '_odd_ratios.txt', 'w') as file:
        file.write("\nBest estimator coefficients and odd ratios: \n")
        for ind, column in enumerate(features):
            file.write("\n" + column + "\n")
            file.write(str(best_coef[ind]) + "\n")
            file.write(str(best_odd_ratio[ind]) + "\n")


def save_sampling_info(directory, title, y_set, sample_technique):
    with open(directory + '/' + title + '_analysis_results.txt', 'w') as save_file:
        save_file.write(f"\n{sample_technique}\n")
        save_file.write("\nUP SAMPLED CLASS VALUE COUNTS:\n")
        unique_y, counts_y = np.unique(y_set, return_counts=True)
        save_file.write(f"{unique_y, counts_y}\n")


def train_model(cv_grid, X_train, y_train, X_test, y_test, dir_path, model,
                random_state, sample_technique, sample_type):
    
    features = X_train.columns
    
    if sample_technique == "smote":
        print("\nInitialize SMOTE\n")
        sample_method = SMOTE(random_state=random_state, sampling_strategy=sample_type)
        X_train, y_train = sample_method.fit_resample(X_train, y_train)
        save_sampling_info(directory=dir_path, title=model, y_set=y_train, sample_technique=sample_technique)

    elif sample_technique == "near_miss":
        print("\nInitialize NearMiss\n")
        sample_method = NearMiss(random_state=random_state, sampling_strategy=sample_type)
        X_train, y_train = sample_method.fit_resample(X_train, y_train)
        save_sampling_info(directory=dir_path, title=model, y_set=y_train, sample_technique=sample_technique)

    elif sample_technique == "ros":
        print("\nInitialize RandomOverSampler\n")
        sample_method = RandomOverSampler(random_state=random_state, sampling_strategy=sample_type)
        X_train, y_train = sample_method.fit_resample(X_train, y_train)
        save_sampling_info(directory=dir_path, title=model, y_set=y_train, sample_technique=sample_technique)

    elif sample_technique == "none" and sample_type == "none":
        print("\nInitialize no sample method.\n")
        X_train, y_train = X_train.values, y_train.values
    else:
        print("Podałeś złe parametry.")
        quit()

    # Convert test sets into numpy arrays
    X_test = X_test.values
    y_test = y_test.values

    # Fit model to train set
    model_grid = cv_grid.fit(X_train, y_train)

    model_best_params = model_grid.best_params_

    model_best_estimator = model_grid.best_estimator_
    
    # Train cross validation scores for each fold
    train_cross_val_score = cross_val_score(estimator=model_best_estimator, X=X_train, y=y_train, cv=5)

    # Test cross validation scores for each fold
    test_cross_val_score = cross_val_score(estimator=model_best_estimator, X=X_test, y=y_test, cv=5)

    # Make prediction on train set
    model_train_proba = model_best_estimator.predict(X_train)

    # Make prediction on test set
    model_test_proba = model_best_estimator.predict(X_test)

    # Plot results
    plot_results(y_train=y_train, train_scores=model_train_proba, y_test=y_test,
                 test_scores=model_test_proba, title=f"{model}", directory=dir_path,
                 model_best_params=model_best_params, model_best_estimator=model_best_estimator,
                 features=features, train_cvs=train_cross_val_score,
                 test_cvs=test_cross_val_score)

    return model_best_estimator    

def encode_variable(data_frame, variable, fill_value):
    data_frame[variable] = data_frame[variable].fillna(fill_value)
    le = LabelEncoder()
    data_frame[variable] = le.fit_transform(data_frame[variable])
    return data_frame

def rating_info(df, df_preds, col, title, dir_path):
    
    # Features to examine
    rating_features=['Procent_niedoplata', 'Niesplacona_faktura', 'Procent_terminowo',
                     'Sredni_termin', 'Srednia_naleznosc', 'Liczba_faktur']
    
    # Auxiliary column
    df['Rating'] = df_preds[col]
    
    # Write descriptions of each examined feature in every rating group
    with open(dir_path + '\\' + title + '_rating_info.txt', 'w') as save_file:
        for rating in sorted(df['Rating'].unique()):
            save_file.write("\n\nRating " + str(rating) + "\n")
            for feat in rating_features:
                save_file.write("\n" + feat + " ")
                save_file.write(str(round(df[df['Rating']==rating][feat].min(),2)) + " - " + str(round(df[df['Rating']==rating][feat].max(),2)) + " ")
                save_file.write("(srednio " + str(round(df[df['Rating']==rating][feat].mean(),2)) + ")") 
                    
    dir_path_count = dir_path + "\\" + title + "_countplot.png"   
    dir_path_bar = dir_path + "\\" + title + "_barplot.png"
    
    # Plot results
    plot_functions.plot_count(df, 'Rating', save=True, directory=dir_path_count)
    plot_functions.plot_bar(df, 'Rating', 'Wolumen', save=True, directory=dir_path_bar)
    
    # Drop auxiliary column
    df.drop('Rating',axis=1,inplace=True)
    
def rating_distribution(df, features, title, dir_path):
    
    if not os.path.exists(dir_path + "\\Ratingi"):
        os.mkdir(dir_path + "\\Ratingi")
    
    for rtg in sorted(df['Rating'].unique()):
        df_rtg=df[df['Rating']==rtg]
        clustering.plot_count(df_rtg, 'Klasa')
        plt.title('Rating ' + rtg + '\nLiczba obserwacji: ' + str(len(df_rtg)))
        directory = dir_path + "\\Ratingi\\" + title + "_rating" + rtg + ".png"  
        plt.savefig(directory)
        
    for feat in features:
        sns.catplot(x="Rating", y=feat, kind="box", data=df,
                    order=sorted(df['Rating'].unique()))
        directory = dir_path + "\\Ratingi\\" + title + "_" + feat + "_countplot.png"  
        plt.savefig(directory)
        
        
        
        
        
        