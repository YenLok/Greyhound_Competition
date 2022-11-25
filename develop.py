
import pandas as pd
import numpy as np
import os, sys

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None) # to ensure console display all columns
pd.set_option('display.float_format', '{:0.3f}'.format)
pd.set_option('display.max_row', 50)
plt.style.use('ggplot')
from pathlib import Path
from copy import deepcopy


project = 'Greyhound_Competition'
projectPath = Path(r'E:/Projects') / f"{project}"
os.chdir(projectPath)
sys.path.append(str(projectPath))

dataPath = projectPath / 'data'
pickleDataPath = dataPath / 'pickle'
htmlDataPath = dataPath / 'html'
imageDataPath = dataPath / 'image'
dataInputPath = dataPath / 'input'
dataWorkingPath = dataPath / 'working'
dataOutputPath = dataPath / 'output'
modelPath = projectPath / 'models'

import pickle
def save_obj(obj, name):
    with open(pickleDataPath / f'{name}.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(pickleDataPath / f'{name}.pkl', 'rb') as f:
        return pickle.load(f)




##############################################################################
## Imports

import tensorflow as tf
print(tf.__version__)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src import project_functions as pf



##############################################################################
## Settings

col_feature = [str(x) for x in list(range(62))]
N = len(col_feature)
n_traps = 6
input_file = f"upwork_data2.csv"


##############################################################################
## Develop Model

df = pd.read_csv(dataInputPath / input_file)
df = df[df.trap <= 6]

def group_horse_and_result(element):
    if element[0] == 'finish':
        return 100 + element[1] # to make sure finish resuls are put near the end
    else:
        return element[1]

df = df.pivot(index = 'race_id', columns = 'trap', values = df.columns[2:])
rearranged_columns = sorted(list(df.columns.values), key = group_horse_and_result)
df = df[rearranged_columns]
df.dropna(inplace = True)
X = df[df.columns[:-6]]
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X),columns = X.columns)
y_won = df[df.columns[-6:]].applymap(lambda x: 1 if x == 1 else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y_won, train_size=0.8, test_size=0.2, random_state=1)
train_id_list = list(y_train.index)
test_id_list = list(y_test.index)
train_df = df.loc[train_id_list]
test_df = df.loc[test_id_list]

bayes_trials_results = pf.get_bayes_opt_params(N, n_traps, col_feature, y_won, train_df, train_id_list, test_df, test_id_list)
best_bayes_output = bayes_trials_results[0]
optimal_params = deepcopy(best_bayes_output['params'])
save_obj(bayes_trials_results, f"bayes_trials_results")
save_obj(optimal_params, f"optimal_params")
save_obj(scaler, f"scaler")
print(optimal_params)
print()



for i in range(10):
    print(bayes_trials_results[i])

# {'loss': 0.5823221772909164, 'params': {'IDR1': 0.2, 'IDR2': 0.2, 'IDR3': 0, 'IN1': 24, 'IN2': 32, 'IN3': 16, 'LR': 0.005, 'MBN1': False, 'MBN2': True, 'MBN3': False, 'MDR1': 0, 'MDR2': 0, 'MDR3': 0, 'MN1': 16, 'MN2': 24, 'MN3': 0, 'act': 'elu', 'apply_reduce_lr_on_plateau': False}, 'status': 'ok', 'accuracy': 0.23535564541816711, 'precision': 1.0}
# {'loss': 0.5834989517927169, 'params': {'IDR1': 0.2, 'IDR2': 0.2, 'IDR3': 0.2, 'IN1': 24, 'IN2': 8, 'IN3': 16, 'LR': 0.001, 'MBN1': True, 'MBN2': True, 'MBN3': True, 'MDR1': 0, 'MDR2': 0.2, 'MDR3': 0.2, 'MN1': 32, 'MN2': 0, 'MN3': 24, 'act': 'elu', 'apply_reduce_lr_on_plateau': False}, 'status': 'ok', 'accuracy': 0.23300209641456604, 'precision': 1.0}
# {'loss': 0.5845449760556221, 'params': {'IDR1': 0.2, 'IDR2': 0.2, 'IDR3': 0, 'IN1': 32, 'IN2': 32, 'IN3': 16, 'LR': 0.01, 'MBN1': False, 'MBN2': True, 'MBN3': True, 'MDR1': 0, 'MDR2': 0, 'MDR3': 0, 'MN1': 32, 'MN2': 32, 'MN3': 0, 'act': 'relu', 'apply_reduce_lr_on_plateau': False}, 'status': 'ok', 'accuracy': 0.2309100478887558, 'precision': 1.0}
# {'loss': 0.6061715483665466, 'params': {'IDR1': 0, 'IDR2': 0.2, 'IDR3': 0.2, 'IN1': 32, 'IN2': 32, 'IN3': 16, 'LR': 0.005, 'MBN1': True, 'MBN2': True, 'MBN3': True, 'MDR1': 0, 'MDR2': 0.2, 'MDR3': 0.2, 'MN1': 16, 'MN2': 0, 'MN3': 32, 'act': 'elu', 'apply_reduce_lr_on_plateau': False}, 'status': 'ok', 'accuracy': 0.22515690326690674, 'precision': 0.75}
# {'loss': 0.6486009433865547, 'params': {'IDR1': 0.2, 'IDR2': 0.2, 'IDR3': 0.2, 'IN1': 24, 'IN2': 24, 'IN3': 16, 'LR': 0.005, 'MBN1': False, 'MBN2': True, 'MBN3': True, 'MDR1': 0, 'MDR2': 0, 'MDR3': 0, 'MN1': 32, 'MN2': 0, 'MN3': 8, 'act': 'elu', 'apply_reduce_lr_on_plateau': False}, 'status': 'ok', 'accuracy': 0.23404811322689056, 'precision': 0.625}































