from numpy import *
import numpy as np
import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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


def zero_inflat(X, max_length):
    #     16 308 8

    transformX = []
    for i in X:
        if i.shape[1] > max_length:
            transformX.append(i[:, 0:max_length])
        else:

            temp = np.pad(i, ((0, 0), (0, max_length - i.shape[1])), 'constant', constant_values=(0, 0))
            transformX.append(temp)
    transformX = np.array(transformX)
    print(transformX.shape)

    return transformX

def train_gru(fold, n_step, n_input, X_train, y_train, X_test, es, rt,cl):
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras import regularizers
    from keras.layers import GRU, Bidirectional, Dropout, Dense, Activation, BatchNormalization,Input
    from keras.models import Sequential
    from keras import backend as K
    from tensorflow import set_random_seed
    import warnings
    warnings.filterwarnings('ignore')
    import os
    from keras.models import Model
    from keras import models
    from keras.utils import to_categorical
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    
    # y_train = to_categorical(y_train, num_classes=2)

    set_random_seed(rt)

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


    n_hidden=512
    bs= X_train.shape[0]
    base_model = keras.models.load_model('1 fold knat.h5') 
    
    if cl=='tcrf' :
        input_shape=(None, n_step, n_input)
        model = Sequential()
        model.add(Bidirectional(GRU(n_hidden,return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(GRU(256)))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-4), loss='binary_crossentropy',
                    metrics=['accuracy'])
        model.build(input_shape) 
        model.summary()

        for i in range(0,len(base_model.layers)-1):
            model.layers[i].set_weights(base_model.layers[i].get_weights())
        
    
        layer_name = 'dropout_3'
        # layer_name = 'dense_1'
        mid_model = Model(input=model.input,
                        output=model.get_layer(layer_name).output)
    
        X_train =  mid_model.predict(X_train)
        X_test = mid_model.predict(X_test)


        # paramgrid = {'C': logspace(-4, 4, 20)}
        # svmcv = GridSearchCV(
        #     SVC(probability=True),
        #     paramgrid,
        #     cv=5,
        #     n_jobs=-1, verbose=0)

        paramgrid = {"n_estimators": [10, 50, 100, 200, 500, 1000]}
        svmcv = GridSearchCV(
            RandomForestClassifier(), 
            paramgrid,                                   
            cv=5,                                   
            n_jobs=-1, verbose=0)

        svmcv.fit(X_train, y_train)
        clf = svmcv.best_estimator_

        pred_pro = clf.predict_proba(X_test)[:, 1]
        pred_class = clf.predict(X_test)

    
    elif cl=='tcsvm' :
        input_shape=(None, n_step, n_input)
        model = Sequential()
        model.add(Bidirectional(GRU(n_hidden,return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(GRU(256)))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-4), loss='binary_crossentropy',
                    metrics=['accuracy'])
        model.build(input_shape) 
        model.summary()

        for i in range(0,len(base_model.layers)-1):
            model.layers[i].set_weights(base_model.layers[i].get_weights())
        
    
        layer_name = 'dropout_3'
        # layer_name = 'dense_1'
        mid_model = Model(input=model.input,
                        output=model.get_layer(layer_name).output)
    
        X_train =  mid_model.predict(X_train)
        X_test = mid_model.predict(X_test)


        paramgrid = {'C': logspace(-4, 4, 20)}
        svmcv = GridSearchCV(
            SVC(probability=True),
            paramgrid,
            cv=5,
            n_jobs=-1, verbose=0)

        svmcv.fit(X_train, y_train)
        clf = svmcv.best_estimator_

        pred_pro = clf.predict_proba(X_test)[:, 1]
        pred_class = clf.predict(X_test)


    elif cl=='ft1':
        y_train = to_categorical(y_train, num_classes=2)

        input_shape=(None, n_step, n_input)
        model = Sequential()
        model.add(Bidirectional(GRU(n_hidden,return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(GRU(256)))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-4), loss='binary_crossentropy',
                    metrics=['accuracy'])
        model.build(input_shape) 
        # model.summary()

        for i in range(0,len(base_model.layers)):
            model.layers[i].set_weights(base_model.layers[i].get_weights())
        

        for layer in model.layers:
            layer.trainable = False

        model.layers[-1].trainable = True
        model.layers[-2].trainable = True
        model.layers[-3].trainable = True

    elif cl=='ft2':
        y_train = to_categorical(y_train, num_classes=2)
        model = models.Sequential()
        model.add(base_model)
        model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-4), loss='binary_crossentropy',
                    metrics=['accuracy'])
        for layer in model.layers:
            layer.trainable = True

    elif cl=='ft3':

        y_train = to_categorical(y_train, num_classes=2)

        input_shape=(None, n_step, n_input)
        model = Sequential()
        model.add(Bidirectional(GRU(n_hidden,return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(GRU(256)))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-4), loss='binary_crossentropy',
                    metrics=['accuracy'])
        model.build(input_shape) 
        # model.summary()

        for i in range(0,len(base_model.layers)-1):
            model.layers[i].set_weights(base_model.layers[i].get_weights())

    # model.fit(X_train, y_train,
    #         batch_size=bs,
    #         epochs=2000, verbose=0,
    #         callbacks=[earlystop2, reduce_lr2],
    #         validation_split=0.1)


 

    model.fit(X_train, y_train,
                  batch_size=bs,
                  epochs=2000, verbose=0,
                  callbacks=[earlystop1, reduce_lr1])
 


    # model.save(str(fold) + " fold knat.h5")


    pred_pro = model.predict(X_test)[:, 1]
    pred_class = model.predict_classes(X_test)
    K.clear_session()
    return pred_pro, pred_class


def evaluate_dl(newX, label, rt, es, cl):
    X = newX
    y = label
    kf = StratifiedKFold(n_splits=5)
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
        seed(i)
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
            # y_train = to_categorical(y_train, num_classes=2)
            print('Dimensions to GRU: ', X_train.shape, y_train.shape)

            X_train = zero_inflat(X_train, 417)
            X_test = zero_inflat(X_test, 417)
            print('Dimensions after zero_inflat: ', X_train.shape, X_test.shape)

            n_step = X_train.shape[1]
            n_input = X_train.shape[2]

            pred, predl = train_gru(fold, n_step, n_input,
                            X_train, y_train, X_test, es, se,cl)

            allcv_probability.extend(pred)
            true_label.extend(y_test)

            auc = roc_auc_score(y_test, pred)
            f1s = f1_score(y_test, predl)
            acc = accuracy_score(y_test, predl)
            pre = precision_score(y_test, predl)
            rec = recall_score(y_test, predl)

            print('Fold ' + str(fold) + ' AUC :  %.4f' % auc)
            print('Fold ' + str(fold) + ' F1 :  %.4f' % f1s)
            # print('Fold ' + str(fold) + ' ACC :  %.4f' % acc)
            # print('Fold ' + str(fold) + ' Precision :  %.4f' % pre)
            # print('Fold ' + str(fold) + ' Recall :  %.4f' % rec)

            all_auc.append(auc)
            all_f1.append(f1s)
            all_acc.append(acc)
            all_pre.append(pre)
            all_rec.append(rec)

        print('\n')
        print(str(i + 1) + ' rounds average AUC :  %.4f' % mean(all_auc))
        print(str(i + 1) + ' rounds average F1 :  %.4f' % mean(all_f1))
        # print(str(i + 1) + ' rounds average ACC :  %.4f' % mean(all_acc))
        # print(str(i + 1) + ' rounds average Precision :  %.4f' % mean(all_pre))
        # print(str(i + 1) + ' rounds average Recall :  %.4f' % mean(all_rec))

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
    # print(str(rt) + ' rounds average 5-fold  ACC :  %.4f' % mean(rs_acc))
    # print(str(rt) + ' rounds average 5-fold  Precision :  %.4f' % mean(rs_pre))
    # print(str(rt) + ' rounds average 5-fold  Recall :  %.4f' % mean(rs_rec))
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
    rs = 1
    es = 'tr'

    OtuX,PhyX,newX,label=load_datasets(file_name)
    actual_label, pre_probability= evaluate_dl(newX, label, rs, es, cl)


# python3 -W ignore com_tc.py -fn david -clf lr -rs 1
#  python3 -W ignore com_dl.py -fn david -clf gru6 -rs 1 -es tr
# python3 -W ignore com_dl.py -fn david -clf mlp -rs
# python3 -W ignore fine_tuning.py


# if __name__ == "__main__":
#     par = read_params(sys.argv)
#     file_name = ['david', 'bdiet', 'bdeliv', 'kegg', 'kige']
#     cl = ['ft1', 'ft2', 'ft3']
#     rs = 10
#     es='tr'

#     for i in file_name:
#         print('\n')
#         print('\033[31mFuck ' + str(i) + ' dataset!\033[0m')
#         OtuX, PhyX, newX, label = load_datasets(i)
        
#         for j in cl:
#             print('\n')
#             print('\033[31mFuck ' + str(i) + ' dataset!\033[0m')
#             print('\033[32mFuck ' + str(j) + ' classifier!\033[0m')
#             actual_label, pre_probability = evaluate_dl(newX, label, rs, es, j)

      
# if __name__ == "__main__":
#     par = read_params(sys.argv)
#     file_name = ['david', 'bdiet', 'bdeliv', 'kegg', 'kige']
#     cl = ['ft1', 'ft2', 'ft3']
#     rs = 10
#     es='te'

#     for i in file_name:
#         print('\n')
#         print('\033[31mFuck ' + str(i) + ' dataset!\033[0m')
#         OtuX, PhyX, newX, label = load_datasets(i)
        
#         for j in cl:
#             print('\n')
#             print('\033[31mFuck ' + str(i) + ' dataset!\033[0m')
#             print('\033[32mFuck ' + str(j) + ' classifier!\033[0m')
#             actual_label, pre_probability = evaluate_dl(newX, label, rs, es, j)



# print('++++++++++++++++++++++++++Early stopping on test data++++++++++++++++++++++++++++')


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


