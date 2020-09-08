
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from numpy import *
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from numpy.random import seed
import argparse as ap
import sys
from keras.utils import to_categorical
from pandas.core.common import flatten
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

def train_ae(X_train,X_test,train_size,test_size,seed):

    from keras.layers import Input,Dense
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras import regularizers
    from keras.layers import GRU, Bidirectional, Dropout, Dense, Activation, BatchNormalization
    from keras.models import Sequential
    from keras import backend as K
    import keras
    from keras.models import Model
    from tensorflow import set_random_seed
    import warnings
    from keras import regularizers
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings('ignore')
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    set_random_seed(seed)
    output_dim1= 512
    
    output_dim2= int(int(X_train.shape[1])*(1/2))

    input_img = Input(shape=(X_train.shape[1],))

    encoded1 = Dense(output_dim1, activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_img)
    encoded2 = Dense(output_dim2, activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded1)

    decoded1 = Dense(output_dim1, activation='sigmoid',activity_regularizer=regularizers.l1(10e-5))(encoded2)
    decoded2 = Dense(X_train.shape[1], activation='sigmoid',activity_regularizer=regularizers.l1(10e-5))(decoded1)


    autoencoder = Model(input_img, decoded2)

    autoencoder.compile(optimizer='adam', loss='MSE')
    
    earlystop1 = EarlyStopping(monitor = 'val_loss',
                          min_delta = 1e-4,
                          patience = 10,
                          verbose = 0)

    rX_train, vX_train = train_test_split(X_train, test_size=0.1, shuffle=True)

    autoencoder.fit(rX_train, rX_train,
                    epochs=1000,
                    batch_size=rX_train.shape[0],
                    callbacks=[earlystop1],verbose=0,
                    validation_data=(vX_train, vX_train))
    encoder = Model(input_img, encoded2)
    # encoded_input = Input(shape=(X_train.shape[1],))

    dtrain_X = encoder.predict(X_train)

    dtest_X = encoder.predict(X_test)

    
    K.clear_session()

    return dtrain_X, dtest_X
   
def train_gru(n_step, n_input, X_train, y_train, X_test, es, seed):
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


    set_random_seed(seed)
    y_train = to_categorical(y_train, num_classes=2)
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
   
    model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-4), loss='binary_crossentropy',
                  metrics=['accuracy'])




    if es == 'tr':

        model.fit(X_train, y_train,
                  batch_size=bs,
                  epochs=2000, verbose=0,
                  callbacks=[earlystop1, reduce_lr1])
    else:
        model.fit(X_train, y_train,
                  batch_size=bs,
                  epochs=2000, verbose=0,
                  callbacks=[earlystop2, reduce_lr2],
                  validation_split=0.1)


    pred_pro = model.predict(X_test)[:, 1]
    pred_class = model.predict_classes(X_test)
    K.clear_session()
    return pred_pro, pred_class


def evaluate_dl(newX, label, rt, es, sfm):
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



            train_size=X_train.shape[0]
            test_size=X_test.shape[0]
            time_dim=X_train.shape[2]
            
            print('Input dimension: ',X_train.shape,y_train.shape)

            # 16,308,8

              # give the label to each sample with the specific time points
            ys_train=[[i]*time_dim for i in y_train]
            ys_train=list(flatten(ys_train)) 
            
            ys_test=[[i]*time_dim for i in y_test]
            ys_test=list(flatten(ys_test)) 

            X_train=np.array([rot90(i,-1) for i in X_train])
            X_test = np.array([rot90(i,-1) for i in X_test])

            Xs_train=np.vstack(X_train)
            Xs_test=np.vstack(X_test)
        
            print('Dimensions to FS: ',Xs_train.shape,len(ys_train),Xs_test.shape,len(ys_test))
        
# #           Normalizaiton
#             X_scaler = StandardScaler()
#             X_scaler.fit(Xs_train)
#             Xs_train = X_scaler.transform(Xs_train)
#             Xs_test = X_scaler.transform(Xs_test)

            cvs=StratifiedKFold(n_splits=5, shuffle=True)

            if sfm !='ae':
            
            
                if sfm=='vt':  #pca ae
                    
                    from sklearn.feature_selection import VarianceThreshold
                    
                    model = VarianceThreshold(threshold=(.99 * (1 - .99)))
                    embeded_lr_feature=model.fit_transform(Xs_train)


                elif sfm=='pca':
                    from sklearn.decomposition import PCA
                    
                    model = PCA(n_components=0.99)
                    embeded_lr_feature=model.fit_transform(Xs_train)

                    
                elif sfm=='rfe':

                    X_scaler = StandardScaler()
                    X_scaler.fit(Xs_train)
                    Xs_train = X_scaler.transform(Xs_train)
                    Xs_test = X_scaler.transform(Xs_test)
                    # l 2
                    from sklearn.feature_selection import RFE
                    from sklearn.linear_model import LogisticRegression
                    paramgrid = {'C': logspace(-4,4,20)}
                    svmcv = GridSearchCV(
                    LogisticRegression(max_iter=2000000), 
                    paramgrid,                                   
                    cv=cvs,                                   
                    n_jobs=-1, verbose=0)
                    svmcv.fit(Xs_train, ys_train)
                    lsvc=svmcv.best_estimator_
                    model = RFE(lsvc, step=1)
                    embeded_lr_feature=model.fit_transform(Xs_train, ys_train)

                elif sfm=='l1':
                    X_scaler = StandardScaler()
                    X_scaler.fit(Xs_train)
                    Xs_train = X_scaler.transform(Xs_train)
                    Xs_test = X_scaler.transform(Xs_test)
                    from sklearn.feature_selection import SelectFromModel
                    from sklearn.svm import LinearSVC
                    paramgrid = {'C': logspace(-4,4,20)}
                    svmcv = GridSearchCV(
                    LinearSVC( penalty="l1", dual=False,max_iter=2000000), 
                    paramgrid,                                   
                    cv=cvs,                                   
                    n_jobs=-1, verbose=0)
                    svmcv.fit(Xs_train, ys_train)
                    lsvc=svmcv.best_estimator_

                    model = SelectFromModel(lsvc, prefit=True)
                
                    embeded_lr_feature=model.transform(Xs_train)

                elif sfm=='rf':
                    from sklearn.feature_selection import SelectFromModel
                    paramgrid = {"n_estimators": [10, 50, 100, 200, 500, 1000]}
                    svmcv = GridSearchCV(
                    RandomForestClassifier(), 
                    paramgrid,                                   
                    cv=cvs,                                   
                    n_jobs=-1, verbose=0)
                    svmcv.fit(Xs_train, ys_train)
                    lsvc=svmcv.best_estimator_

                    model = SelectFromModel(lsvc, prefit=True)
                
                    embeded_lr_feature=model.transform(Xs_train)
                
                
                print(str(embeded_lr_feature.shape[1]), 'selected features')
                
                if embeded_lr_feature.shape[1]==0:
                    
                    X_train=Xs_train
                    X_test=Xs_test
                    
                else:
                    
                    X_train = model.transform(Xs_train)
                    X_test=model.transform(Xs_test)
                
                
                
                X_scaler = StandardScaler()
                X_scaler.fit(X_train)
                X_train = X_scaler.transform(X_train)
                X_test = X_scaler.transform(X_test)

            else:

                X_train, X_test = train_ae(Xs_train,Xs_test,train_size,test_size,i)

                print('Out of AE: ',X_train.shape, X_test.shape)
            
           # reshape to the matirix
            X_train = X_train.reshape(train_size,time_dim,-1)
            X_test = X_test.reshape(test_size,time_dim,-1)
            
            print('Dimensions to GRU: ',X_train.shape,X_test.shape)
            se = i
            n_step = X_train.shape[1]
            n_input = X_train.shape[2]

            pred, predl = train_gru(n_step, n_input,
                            X_train, y_train, X_test, es, se)



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
    rs = str(par['rs'])
    es = str(par['es'])

    OtuX,PhyX,newX,label=load_datasets(file_name)
    actual_label, pre_probability= evaluate_dl(newX, label, rs, es, cl)



# if __name__ == "__main__":
#     par = read_params(sys.argv)
#     file_name = ['david','bdiet','bdeliv','knat','kegg','kige']
#     cl = ['ae', 'vt', 'pca', 'rfe', 'l1', 'rf']
#     rs = 10

#     es='tr'

#     for i in file_name:
#         print('\n')
#         print('\033[31mFuck '+str(i)+' dataset!\033[0m')
#         OtuX, PhyX, newX, label = load_datasets(i)

#         for j in cl:
#             print('\n')
#             print('\033[31mFuck ' + str(i) + ' dataset!\033[0m')
#             print('\033[32mFuck '+str(j)+' classifier!\033[0m')
#             actual_label, pre_probability= evaluate_dl(newX,label,rs,es,j)

# if __name__ == "__main__":
#     par = read_params(sys.argv)
#     file_name = ['david','bdiet','bdeliv','knat','kegg','kige']
#     cl = ['ae', 'vt', 'pca', 'rfe', 'l1', 'rf']
#     rs = 10

#     es='te'

#     for i in file_name:
#         print('\n')
#         print('\033[31mFuck '+str(i)+' dataset!\033[0m')
#         OtuX, PhyX, newX, label = load_datasets(i)

#         for j in cl:
#             print('\n')
#             print('\033[31mFuck ' + str(i) + ' dataset!\033[0m')
#             print('\033[32mFuck '+str(j)+' classifier!\033[0m')
#             actual_label, pre_probability= evaluate_dl(newX,label,rs,es,j)