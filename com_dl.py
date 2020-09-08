from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from numpy import *
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from numpy.random import seed
import argparse as ap
import sys
from sklearn.preprocessing import StandardScaler
from keras.layers import merge,Multiply


from keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout,Bidirectional,Multiply
from keras.models import Model
from attention_utils import get_activations
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import  pandas as pd
import  numpy as np


SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def attention_model(INPUT_DIMS,TIME_STEPS,lstm_units):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = Conv1D(filters = 64, kernel_size = 1, activation = 'relu')(inputs)  #, padding = 'same'
    x = Dropout(0.3)(x)

    #lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    #对于GPU可以使用CuDNNLSTM
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    fc1=Dense(256, activation='relu')(attention_mul )
    output = Dense(2, activation='softmax')(fc1)
    model = Model(inputs=[inputs], outputs=output)
    return model

def load_datasets(dataset):
    OtuX = np.load(file=dataset + "_Otu_Matrix.npy")
    PhyX = np.load(file=dataset + "_Phy_Matrix.npy")
    newX = np.load(file=dataset + "_Combine_Matrix.npy")
    label = np.load(file=dataset + "_Label.npy")

    # print('Otu (sizes * features * time points): ',OtuX.shape)
    # print('Phy (sizes * features * time points): ',PhyX.shape)
    print('Combine features (sizes * features * time points): ', newX.shape)
    print('Label size: ', len(label))

    return OtuX, PhyX, newX, label


def train_gru(n_step, n_input, X_train, y_train, X_test, es, seed, ar):
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras import regularizers
    from keras.layers import GRU, Bidirectional, Dropout, Dense, Activation, BatchNormalization
    from keras.models import Sequential
    from keras import backend as K
    import keras
    from tensorflow import set_random_seed
    import warnings
    warnings.filterwarnings('ignore')
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


    # set_random_seed(seed)

    earlystop1 = EarlyStopping(monitor='loss',
                               min_delta=0.0001,
                               patience=10,
                               verbose=0, restore_best_weights=True)

    earlystop2 = EarlyStopping(monitor='val_loss',
                               min_delta=0.0001,
                               patience=10,
                               verbose=0, restore_best_weights=True)

    reduce_lr1 = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)

    reduce_lr2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)

    K.clear_session()

    if ar=='gru1':
        n_hidden=512
        bs= X_train.shape[0]
        model = Sequential()
        model.add(GRU(n_hidden,
                      batch_input_shape=(None, n_step, n_input)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(2, activation='softmax'))
    elif ar=='gru2':
        n_hidden = 512
        bs = X_train.shape[0]
        model = Sequential()
        model.add(Bidirectional(GRU(n_hidden,
                      batch_input_shape=(None, n_step, n_input))))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(2, activation='softmax'))
    elif ar == 'gru3':
        n_hidden = 512
        bs = X_train.shape[0]
        model = Sequential()
        model.add(Bidirectional(GRU(n_hidden,return_sequences=True,
                                    batch_input_shape=(None, n_step, n_input))))
        model.add(Bidirectional(GRU(256)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(2, activation='softmax'))

    elif ar == 'gru4':
        n_hidden = 512
        bs = 4
        model = Sequential()
        model.add(Bidirectional(GRU(n_hidden,return_sequences=True,
                                    batch_input_shape=(None, n_step, n_input))))
        model.add(Dropout(0.2))
        model.add(Bidirectional(GRU(256)))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

    elif ar == 'gru7':
        n_hidden = 512
        bs =  8   #8
        model = Sequential()
        model.add(Bidirectional(GRU(n_hidden,return_sequences=True,batch_input_shape=(None, n_step, n_input),
         kernel_regularizer=regularizers.l2(0.0002), kernel_initializer='he_uniform', recurrent_dropout=0.2
         )))

        model.add(Dropout(0.2))
        model.add(Bidirectional(GRU(256, kernel_regularizer=regularizers.l2(0.0002), kernel_initializer='he_uniform'
        , recurrent_dropout=0.2)))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0002), kernel_initializer='he_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.0002)))

    elif ar == 'gru8':

        bs=1
   
        INPUT_DIMS = X_train.shape[2]
        TIME_STEPS = X_train.shape[1]
        lstm_units = 512

        model = attention_model(INPUT_DIMS,TIME_STEPS,lstm_units)

    elif ar == 'gru5':
        n_hidden = 512
        bs =  4
        model = Sequential()
        model.add(Bidirectional(GRU(n_hidden,return_sequences=True,batch_input_shape=(None, n_step, n_input), kernel_regularizer=regularizers.l2(0.0001))))

        model.add(Dropout(0.2))
        model.add(Bidirectional(GRU(256, kernel_regularizer=regularizers.l2(0.0001))))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.0001)))

    elif ar == 'gru6':
        n_hidden = 512
        bs = X_train.shape[0]
        model = Sequential()
        model.add(Bidirectional(GRU(n_hidden,return_sequences=True,
                                    batch_input_shape=(None, n_step, n_input),
                                    use_bias=False)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Bidirectional(GRU(256, use_bias=False)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(128,  use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(2, activation='softmax'))



    model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-4), loss='binary_crossentropy',
                  metrics=['accuracy'])




    if es == 'tr':

        model.fit(X_train, y_train,
                  batch_size=bs,
                  epochs=2000, verbose=1,
                  callbacks=[earlystop1, reduce_lr1])
    else:
        model.fit(X_train, y_train,
                  batch_size=bs,
                  epochs=2000, verbose=1,
                  callbacks=[earlystop2, reduce_lr2],
                  validation_split=0.1)


    pred_pro = model.predict(X_test)[:, 1]


    # pred_class = model.predict_classes(X_test)

    predict = model.predict(X_test)
    pred_class=np.argmax(predict,axis=1)

    K.clear_session()
    return pred_pro, pred_class


def evaluate_dl(newX, label, rt, es, cl):
    X = newX
    y = label
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    kf.get_n_splits(X, y)

    rs_auc = []
    rs_f1 = []
    rs_acc = []
    rs_pre = []
    rs_rec = []

    repeat_times = int(rt)

    allrs_probability = []

    for i in range(repeat_times):

        all_auc = []
        all_f1 = []
        all_acc = []
        all_pre = []
        all_rec = []

        print('==================== ' + str(i + 1) + ' repeat ====================')
        # seed(i)
        fold = 0

        allcv_probability = []
        true_label = []

        for train_index, test_index in kf.split(X, y):

            fold += 1
            print('********** ' + str(fold) + ' fold **********')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            train_size = X_train.shape[0]
            test_size = X_test.shape[0]
            otu_len = X_train.shape[1]

            print('Input dimension: ', X_train.shape, y_train.shape)

            if cl == 'mlp':
                from sklearn.neural_network import MLPClassifier
                # reshape to 1D
                X_train = X_train.reshape(train_size, -1)
                X_test = X_test.reshape(test_size, -1)

                print('Dimension to mlp: ', X_train.shape, X_test.shape)

                #           Normalizaiton
                X_scaler = StandardScaler()
                X_scaler.fit(X_train)
                X_train = X_scaler.transform(X_train)
                X_test = X_scaler.transform(X_test)

                model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=20000).fit(X_train, y_train)
                pred = model.predict_proba(X_test)[:, 1]
                predl = model.predict(X_test)

            else:
                se = i

                from keras.utils import to_categorical

                X_train = np.array([rot90(i, -1) for i in X_train])
                X_test = np.array([rot90(i, -1) for i in X_test])


                # Normalizaiton
                X_train = np.vstack(X_train)
                X_test = np.vstack(X_test)

                X_scaler = StandardScaler()
                # X_scaler = MinMaxScaler(feature_range=(-1, 1))
                X_scaler.fit(X_train)
                X_train = X_scaler.transform(X_train)
                X_test = X_scaler.transform(X_test)

                X_train = X_train.reshape(train_size, -1, otu_len)
                X_test = X_test.reshape(test_size, -1, otu_len)
                y_train = to_categorical(y_train, num_classes=2)
                print('Dimensions to GRU: ', X_train.shape, y_train.shape)

                n_step = X_train.shape[1]
                n_input = X_train.shape[2]




                pred, predl = train_gru(n_step, n_input,
                                X_train, y_train, X_test, es, se, cl)

            allcv_probability.extend(pred)
            true_label.extend(y_test)

            auc = roc_auc_score(y_test, pred)
            f1s = f1_score(y_test, predl)
            acc = accuracy_score(y_test, predl)
            pre = precision_score(y_test, predl)
            rec = recall_score(y_test, predl)

            print('Fold ' + str(fold) + ' AUC :  %.4f' % auc)
            print('Fold ' + str(fold) + ' F1 :  %.4f' % f1s)
            print('Fold ' + str(fold) + ' ACC :  %.4f' % acc)
            print('Fold ' + str(fold) + ' Precision :  %.4f' % pre)
            print('Fold ' + str(fold) + ' Recall :  %.4f' % rec)

            all_auc.append(auc)
            all_f1.append(f1s)
            all_acc.append(acc)
            all_pre.append(pre)
            all_rec.append(rec)

        print('\n')
        print(str(i + 1) + ' rounds average AUC :  %.4f' % mean(all_auc))
        print(str(i + 1) + ' rounds average F1 :  %.4f' % mean(all_f1))
        print(str(i + 1) + ' rounds average ACC :  %.4f' % mean(all_acc))
        print(str(i + 1) + ' rounds average Precision :  %.4f' % mean(all_pre))
        print(str(i + 1) + ' rounds average Recall :  %.4f' % mean(all_rec))

        allrs_probability.append(allcv_probability)
        rs_auc.append(mean(all_auc))
        rs_f1.append(mean(all_f1))
        rs_acc.append(mean(all_acc))
        rs_pre.append(mean(all_pre))
        rs_rec.append(mean(all_rec))
    print('\n')
    print('#################### Over! ####################')

    print(str(rt) + ' rounds average 5-fold  AUC :  %.4f' % mean(rs_auc))
    print(str(rt) + ' rounds average 5-fold  F1 :  %.4f' % mean(rs_f1))
    print(str(rt) + ' rounds average 5-fold  ACC :  %.4f' % mean(rs_acc))
    print(str(rt) + ' rounds average 5-fold  Precision :  %.4f' % mean(rs_pre))
    print(str(rt) + ' rounds average 5-fold  Recall :  %.4f' % mean(rs_rec))
    print('\n')
    best_auc = max(rs_auc)
    best_seed = rs_auc.index(best_auc)
    print('Best seed for ' + str(rt) + ' rounds: ', best_seed)

    print('Best AUC for ' + str(rt) + ' rounds: %.4f' % rs_auc[best_seed])
    print('Best F1 for ' + str(rt) + ' rounds: %.4f' % rs_f1[best_seed])


    best_pro = allrs_probability[best_seed]



    # return the probability with the best auc and f1
    print('True_label: '+'\t')
    print(true_label)
    print('Best_pro: '+'\t')
    print(best_pro)

    return true_label, best_pro


def read_params(args):
    parser = ap.ArgumentParser(description='Experiment')
    arg = parser.add_argument
    arg('-fn', '--fn', type=str, help='datasets')
    arg('-clf', '--clf', type=str, help='classifier')
    arg('-rs', '--rs', type=str, help='repeat time')
    arg('-es', '--es', type=str, help='earlystopping')
    return vars(parser.parse_args())



if __name__ == "__main__":
    par = read_params(sys.argv)
    file_name = str(par['fn'])
    cl = str(par['clf'])
    rs = str(par['rs'])
    es = str(par['es'])

    OtuX,PhyX,newX,label=load_datasets(file_name)
    actual_label, pre_probability= evaluate_dl(newX, label, rs, es, cl)


# python3 -W ignore com_tc.py -fn david -clf lr -rs 1
#  python3 -W ignore com_dl.py -fn david -clf gru6 -rs 1 -es tr
# python3 -W ignore com_dl.py -fn david -clf mlp -rs

#
#
# if __name__ == "__main__":
#     par = read_params(sys.argv)
#     file_name = ['david', 'bdiet', 'bdeliv', 'knat', 'kegg', 'kige']
#     clf = ['mlp', 'gru1', 'gru2', 'gru3', 'gru4','gru5','gru6']
#     rs = 10
#     es='tr'
#
#     for i in file_name:
#         print('\n')
#         print('\033[31mFuck ' + str(i) + ' dataset!\033[0m')
#         OtuX, PhyX, newX, label = load_datasets(i)
#
#         for j in clf:
#             print('\n')
#             print('\033[31mFuck ' + str(i) + ' dataset!\033[0m')
#             print('\033[32mFuck ' + str(j) + ' classifier!\033[0m')
#             actual_label, pre_probability = evaluate_dl(newX, label, rs, es, j)



# print('++++++++++++++++++++++++++Early stopping on test data++++++++++++++++++++++++++++')
#
#
# if __name__ == "__main__":
#     par = read_params(sys.argv)
#     file_name = ['david', 'bdiet', 'bdeliv', 'knat', 'kegg', 'kige']
#     clf = ['mlp', 'gru1', 'gru2', 'gru3', 'gru4','gru5','gru6']
#     rs = 10
#     es='ts'
#
#     for i in file_name:
#         print('\n')
#         print('\033[31mFuck ' + str(i) + ' dataset!\033[0m')
#         OtuX, PhyX, newX, label = load_datasets(i)
#
#         for j in clf:
#             print('\n')
#             print('\033[31mFuck ' + str(i) + ' dataset!\033[0m')
#             print('\033[32mFuck ' + str(j) + ' classifier!\033[0m')
#             actual_label, pre_probability = evaluate_dl(newX, label, rs, es, j)