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

def load_datasets(dataset):
 
    OtuX= np.load(file=dataset+"_Otu_Matrix.npy")
    PhyX= np.load(file=dataset+"_Phy_Matrix.npy")
    newX= np.load(file=dataset+"_Combine_Matrix.npy")
    label= np.load(file=dataset+"_Label.npy")
    
    # print('Otu (sizes * features * time points): ',OtuX.shape)
    # print('Phy (sizes * features * time points): ',PhyX.shape)
    print('Combine features (sizes * features * time points): ',newX.shape)
    print('Label size: ', len(label))
 
    return OtuX,PhyX,newX,label


def evaluate_ml(newX,label,rt,cl):
    X = newX
    y=label
    kf = StratifiedKFold(n_splits=5)
    kf.get_n_splits(X,y)

    rs_auc = []
    rs_f1 = []
    rs_acc = []
    rs_pre = []
    rs_rec = []

    repeat_times = int(rt)


    allrs_probability=[]

    for i in range(repeat_times):

        all_auc = []
        all_f1 = []
        all_acc = []
        all_pre = []
        all_rec = []

        print('==================== '+str(i+1)+' repeat ====================')
        # seed(i)
        fold = 0

        allcv_probability=[]
        true_label=[]

        for train_index, test_index in kf.split(X,y):

            fold += 1
            print('********** ' + str(fold) + ' fold **********')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test  = y[train_index], y[test_index]

            train_size=X_train.shape[0]
            test_size=X_test.shape[0]

            X_train = np.array([rot90(i, -1) for i in X_train])
            X_test = np.array([rot90(i, -1) for i in X_test])

            # Normalizaiton
            X_train = np.vstack(X_train)
            X_test = np.vstack(X_test)
            from sklearn.preprocessing import StandardScaler
            X_scaler = StandardScaler()
            # X_scaler = MinMaxScaler(feature_range=(-1, 1))
            X_scaler.fit(X_train)
            X_train = X_scaler.transform(X_train)
            X_test = X_scaler.transform(X_test)

       
            
            print('Input dimension: ',X_train.shape,y_train.shape)

           # reshape to 1D
            X_train=X_train.reshape(train_size,-1)
            X_test=X_test.reshape(test_size,-1)
            
            print('Dimension to clf: ',X_train.shape,X_test.shape)

# #           Normalizaiton
#             from sklearn.preprocessing import StandardScaler
#             X_scaler = StandardScaler()
#             X_scaler.fit(X_train)
#             X_train = X_scaler.transform(X_train)
#             X_test = X_scaler.transform(X_test)


            if cl=='rf':
                paramgrid = {"n_estimators": [1,5,10,20,50]}
                svmcv = GridSearchCV(
                  RandomForestClassifier(), 
                  paramgrid,                                   
                  cv=5,                                   
                  n_jobs=-1, verbose=0)

            elif cl=='lr':
                paramgrid = {'C': logspace(-4,4,20)}
                svmcv = GridSearchCV(
                  LogisticRegression(max_iter=20000), 
                  paramgrid,                                   
                  cv=5,                                   
                  n_jobs=-1, verbose=0)

            elif cl=='svm':
                paramgrid = {'C': logspace(-4,4,20)}
                svmcv = GridSearchCV(
                  SVC(probability=True), 
                  paramgrid,                                   
                  cv=5,                                   
                  n_jobs=-1, verbose=0)

            elif cl=='knn':
                diff = min(sum(y_train),len(y_train)-sum(y_train))

                # the largest k should be no more than the number of samples

                paramgrid = {'n_neighbors': range(2,diff+1)}
                svmcv = GridSearchCV(
                  KNeighborsClassifier(), 
                  paramgrid,                                   
                  cv=5,                                   
                  n_jobs=-1, verbose=0)

            elif cl=='xgb':
                paramgrid = {'learning_rate': array([1, 0.1, 0.01, 0.0001, 0.00001]), 
                'n_estimators': array([10, 50, 100, 200, 500])}
                svmcv = GridSearchCV(
                  XGBClassifier(), 
                  paramgrid,                                   
                  cv=5,                                   
                  n_jobs=1, verbose=0)


            svmcv.fit(X_train, y_train)
            model=svmcv.best_estimator_
            
            pred=model.predict_proba(X_test)[:,1]
            predl = model.predict(X_test)

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
        print(str(i+1) + ' rounds average AUC :  %.4f' % mean(all_auc))
        print(str(i+1) + ' rounds average F1 :  %.4f' % mean(all_f1))
        print(str(i+1) + ' rounds average ACC :  %.4f' % mean(all_acc))
        print(str(i+1) + ' rounds average Precision :  %.4f' % mean(all_pre))
        print(str(i+1) + ' rounds average Recall :  %.4f' % mean(all_rec))

        allrs_probability.append(allcv_probability)
        rs_auc.append(mean(all_auc))
        rs_f1.append(mean(all_f1))
        rs_acc.append(mean(all_acc))
        rs_pre.append(mean(all_pre))
        rs_rec.append(mean(all_rec))
    print('\n')
    print('#################### Over! ####################')
    best_auc = max(rs_auc)
    best_seed = rs_auc.index(best_auc)
    print('Best seed for '+ str(rt) + ' rounds: ', best_seed)

    best_pro = allrs_probability[best_seed]


    print(str(rt) + ' rounds average 5-fold  AUC :  %.4f' % mean(rs_auc))
    print(str(rt) + ' rounds average 5-fold  F1 :  %.4f' % mean(rs_f1))
    print(str(rt) + ' rounds average 5-fold  ACC :  %.4f' % mean(rs_acc))
    print(str(rt) + ' rounds average 5-fold  Precision :  %.4f' % mean(rs_pre))
    print(str(rt) + ' rounds average 5-fold  Recall :  %.4f' % mean(rs_rec))

    # return the probability with the best auc and f1

    return true_label, best_pro





def read_params(args):
    parser = ap.ArgumentParser(description='Experiment')
    arg = parser.add_argument
    arg('-fn', '--fn', type=str, help='datasets')
    arg('-clf', '--clf', type=str, help='classifier')
    arg('-rs', '--rs', type=str, help='repeat time')
    return vars(parser.parse_args())



if __name__ == "__main__":
    par = read_params(sys.argv)
    file_name = str(par['fn'])
    clf = str(par['clf'])
    rs = str(par['rs'])

    OtuX,PhyX,newX,label=load_datasets(file_name)
    actual_label, pre_probability= evaluate_ml(newX,label,rs,clf)


 # python3 -W ignore com_tc.py -fn david -clf lr -rs 1
