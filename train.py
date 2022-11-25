

import numpy as np
import pandas as pd
import feather
import os, sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None) # to ensure console display all columns
pd.set_option('display.float_format', '{:0.3f}'.format)
pd.set_option('display.max_row', 50)
plt.style.use('ggplot')
from pathlib import Path


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

from src import project_functions as pf

import tensorflow as tf
print(tf.__version__)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping



##############################################################################
## Settings

col_feature = [str(x) for x in list(range(62))]
N = len(col_feature)
n_traps = 6
input_file = f"upwork_data2.csv"


##############################################################################
## Train Model


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


## The generally good architecture based on the observations from the top architectures with the best results obtained from develop.py
pf.set_seed(1)
act = 'relu'

def get_MLP_input_layer(N, act):
    input = Input(shape=(N,))

    x = Dense(32, activation=act)(input)
    x = Dropout(0.2)(x)

    x = Dense(32, activation=act)(x)
    x = Dropout(0.2)(x)

    x = Dense(16, activation=act)(x)
    #x = Dropout(0.2)(x)

    x = Model(inputs=input, outputs=x)

    return x

concat_list = []
for i in range(n_traps):
    concat_list += [get_MLP_input_layer(N, act)]

combined = concatenate([x.output for x in concat_list])

m = Dense(32, activation=act)(combined)
#m = BatchNormalization()(m)

m = Dense(32, activation=act)(m)
m = BatchNormalization()(m)

m = Dense(n_traps, activation="softmax")(m)
model = Model(inputs=[x.input for x in concat_list], outputs=m)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.005),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['acc', tf.keras.metrics.Precision(name='precision')],
)

early_stop = EarlyStopping(monitor='val_acc', min_delta=0,
                           patience=50, verbose=0, mode='auto',
                           baseline=0, restore_best_weights=True)

history = model.fit([train_df.xs(i, level="trap", axis=1)[col_feature].values for i in range(1, n_traps+1)],
                    y_won.loc[train_id_list].values,
                    epochs=1000,
                    batch_size=1028,
                    validation_data=([test_df.xs(i, level="trap", axis=1)[col_feature].values for i in range(1, n_traps+1)],
                                     y_won.loc[test_id_list].values),
                    shuffle=True, verbose=1,
                    callbacks=[early_stop])

test_loss, test_acc, test_precision = model.evaluate([test_df.xs(i, level="trap", axis=1)[col_feature].values for i in range(1, n_traps+1)],
                                                     y_won.loc[test_id_list].values,
                                                     batch_size=1028,
                                                     verbose=0)

print(f"Accuracy: {test_acc:,.3f}, Precision: {test_precision:,.3f}")
# model.save(modelPath / f'model')

























