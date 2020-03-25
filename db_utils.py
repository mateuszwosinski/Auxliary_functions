from str2bool import str2bool
import argparse
import os
import pickle


def parse_cmd_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--update', '--update', type=bool, help='Do you want to update records from db? [True|False]')
    args = parser.parse_args()
    if isinstance(str2bool(vars(args)['update']), bool):
        print('Podałeś niewłaściwy parametr! Można wybrać tylko "True" albo "False". Spróbuj ponownie.')
        quit()
    return vars(args)


def save_to_pickle(df, df_name, mode='XL'):
    with open(os.getcwd() + f'\\dataframes\\' + mode +  '\\' + f'{df_name}.pkl', 'wb') as handle:
        pickle.dump(df, handle)


def load_from_pickle(df_name, mode='XL'):
    with open(os.getcwd() + '\\dataframes\\' + mode + '\\' + f'{df_name}', 'rb') as handle:
        return pickle.load(handle)
