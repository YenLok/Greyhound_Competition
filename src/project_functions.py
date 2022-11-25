
import os, random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from hyperopt import hp, tpe, Trials, STATUS_OK, fmin
from functools import partial


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_mlp_nn(N, n_traps,
               IN1, IDR1, IBN1,
               IN2, IDR2, IBN2,
               IN3, IDR3, IBN3,
               MN1, MDR1, MBN1,
               MN2, MDR2, MBN2,
               MN3, MDR3, MBN3,
               LR, act):

    set_seed(1)

    def get_MLP_input_layer(N, IN1, IDR1, IBN1,
                            IN2, IDR2, IBN2,
                            IN3, IDR3, IBN3, act):
        input = Input(shape=(N,))

        x = Dense(IN1, activation=act)(input)
        if IDR1 > 0:
            x = Dropout(IDR1)(x)
        if IBN1:
            x = BatchNormalization()(x)

        if IN2 > 0:
            x = Dense(IN2, activation=act)(x)
            if IDR2 > 0:
                x = Dropout(IDR2)(x)
            if IBN2:
                x = BatchNormalization()(x)

        if IN3 > 0:
            x = Dense(IN3, activation=act)(x)
            if IDR3 > 0:
                x = Dropout(IDR3)(x)
            if IBN3:
                x = BatchNormalization()(x)

        x = Model(inputs=input, outputs=x)
        return x


    concat_list = []
    for i in range(n_traps):
        concat_list += [get_MLP_input_layer(N, IN1, IDR1, IBN1,
                                            IN2, IDR2, IBN2,
                                            IN3, IDR3, IBN3, act)]

    combined = concatenate([x.output for x in concat_list])

    m = Dense(MN1, activation=act)(combined)

    if MDR1 > 0:
        m = Dropout(MDR1)(m)
    if MBN1:
        m = BatchNormalization()(m)

    if MN2 > 0:
        m = Dense(MN2, activation=act)(m)
        if MDR2 > 0:
            m = Dropout(MDR2)(m)
        if MBN2:
            m = BatchNormalization()(m)

    if MN3 > 0:
        m = Dense(MN3, activation=act)(m)
        if MDR3 > 0:
            m = Dropout(MDR3)(m)
        if MBN3:
            m = BatchNormalization()(m)

    m = Dense(n_traps, activation="softmax")(m)
    model = Model(inputs=[x.input for x in concat_list], outputs=m)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=LR),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['acc', tf.keras.metrics.Precision(name='precision')],
    )

    return model


def get_model_from_params(params, N, n_traps):

    model = get_mlp_nn(N, n_traps,
                       params['IN1'], params['IDR1'], params['IBN1'],
                       params['IN2'], params['IDR2'], params['IBN2'],
                       params['IN3'], params['IDR3'], params['IBN3'],
                       params['MN1'], params['MDR1'], params['MBN1'],
                       params['MN2'], params['MDR2'], params['MBN2'],
                       params['MN3'], params['MDR3'], params['MBN3'],
                       params['LR'], params['act'])

    return model


def get_bayes_opt_params(N, n_traps, col_feature, y_won, train_df, train_id_list, test_df, test_id_list):

    def bayes_objective(params):
        set_seed(1)
        model = None
        model = get_model_from_params(params, N, n_traps)

        lr_decay = ReduceLROnPlateau(monitor='loss',
                                     patience=1, verbose=0,
                                     factor=0.5, min_lr=1e-8)

        early_stop = EarlyStopping(monitor='val_acc', min_delta=0,
                                   patience=50, verbose=0, mode='auto',
                                   baseline=0, restore_best_weights=True)

        callbacks = []
        if params['apply_reduce_lr_on_plateau']:
            callbacks += [lr_decay]
        callbacks += [early_stop]

        history = model.fit([train_df.xs(i, level="trap", axis=1)[col_feature].values for i in range(1, n_traps+1)],
                            y_won.loc[train_id_list].values,
                            epochs=1000,
                            batch_size=1028,
                            validation_data=([test_df.xs(i, level="trap", axis=1)[col_feature].values for i in range(1, n_traps+1)],
                                             y_won.loc[test_id_list].values),
                            shuffle=True, verbose=0,
                            callbacks=callbacks)

        test_loss, test_acc, test_precision = model.evaluate([test_df.xs(i, level="trap", axis=1)[col_feature].values for i in range(1, n_traps+1)],
                                                             y_won.loc[test_id_list].values,
                                                             batch_size=1028,
                                                             verbose=0)

        score = (min(test_precision, 0.8) * 0.75 + test_acc) / 2
        loss = 1 - score
        print(f"Accuracy: {test_acc:,.3f}, Precision: {test_precision:,.3f}")

        return {'loss': loss, 'params': params, 'status': STATUS_OK, 'accuracy': test_acc, 'precision': test_precision}

    space = {
            'IN1': hp.choice('IN1', [16, 24, 32]),
            'IDR1': hp.choice('IDR1', [0, 0.2]),
            'IBN1': hp.choice('IBN1', [True, False]),

            'IN2': hp.choice('IN2', [0, 8, 16, 24, 32]),
            'IDR2': hp.choice('IDR2', [0, 0.2]),
            'IBN2': hp.choice('IBN2', [True, False]),

            'IN3': hp.choice('IN3', [0, 8, 16, 24, 32]),
            'IDR3': hp.choice('IDR3', [0, 0.2]),
            'IBN3': hp.choice('IBN3', [True, False]),

            'MN1': hp.choice('MN1', [16, 24, 32]),
            'MDR1': hp.choice('MDR1', [0, 0.2]),
            'MBN1': hp.choice('MBN1', [True, False]),

            'MN2': hp.choice('MN2', [0, 8, 16, 24, 32]),
            'MDR2': hp.choice('MDR2', [0, 0.2]),
            'MBN2': hp.choice('MBN2', [True, False]),

            'MN3': hp.choice('MN3', [0, 8, 16, 24, 32]),
            'MDR3': hp.choice('MDR3', [0, 0.2]),
            'MBN3': hp.choice('MBN3', [True, False]),

            'act': hp.choice('act', ['elu', 'relu']),
            'LR': hp.choice('LR', [1e-2, 1e-3, 0.005]),

            'apply_reduce_lr_on_plateau': hp.choice('apply_reduce_lr_on_plateau', [True, False]),
            }

    bayes_trials = Trials()
    #rstate = np.random.default_rng(100)
    rstate = np.random.RandomState(100)
    bayes_output = fmin(fn = bayes_objective,
                        space = space,
                        algo = partial(tpe.suggest, n_startup_jobs = 10, n_EI_candidates = 24),
                        max_evals = 250,
                        trials = bayes_trials,
                        rstate=rstate)

    bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])

    return bayes_trials_results





