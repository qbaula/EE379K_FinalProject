from __future__ import division
import pandas as pd
import numpy as np
import XGBoostClassifier as xg
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import StratifiedKFold

SEED = 80085

# Open .csv file and performs feature processing
def read_data(filename):
    data = pd.read_csv(filename, sep=',', header=0)
    data.drop('ROLE_CODE', axis=1, inplace=True) # redundant column

    # Count feature to show significance of category
    for col in data.columns:
        if col == 'ACTION' or col == 'id':
            continue
        else:
            print ('Adding counts for ' + col)
            count = data[col].value_counts()
            data['count_'+col] = data[col].replace(count)

    # OneHotEncoding adds too many features (100s, possibly 1000s) and takes too long to process
    # encode_columns = ['ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_FAMILY']
    # for column in encode_columns:
    #     print ("Performing OneHotEncoding for " + column)
    #     dummies = pd.get_dummies(data[column], prefix=column)
    #     data = pd.concat([data, dummies], axis=1)
    #     data = data.drop(column, axis=1)

    return data

def make_submission(csv_name, idx, preds):
    submission = pd.DataFrame({ 'id': idx,
                                'ACTION': preds })
    submission.to_csv(csv_name + ".csv", index=False, columns = ['id', 'ACTION'])

def bagged_set(X_t,y_c,model, seed, estimators, xt, update_seed=True):

   # create array object to hold predictions
   baggedpred=[ 0.0  for d in range(0, (xt.shape[0]))]
   #loop for as many times as we want bags
   for n in range (0, estimators):
        #shuff;e first, aids in increasing variance and forces different results
        #X_t,y_c=shuffle(Xs,ys, random_state=seed+n)

        if update_seed: # update seed if requested, to give a slightly different model
            model.set_params(random_state=seed + n)
        model.fit(X_t,y_c) # fit model0.0917411475506
        preds=model.predict_proba(xt)[:,1] # predict probabilities
        # update bag's array
        for j in range (0, (xt.shape[0])):
                baggedpred[j]+=preds[j]
   # divide with number of bags to create an average estimate
   for j in range (0, len(baggedpred)):
                baggedpred[j]/=float(estimators)
   # return probabilities
   return np.array(baggedpred)

def main():
    X_train = read_data('train.csv')
    y_train = X_train['ACTION']
    X_train.drop('ACTION', axis=1, inplace=True)

    X_test = read_data('test.csv')
    id_test = X_test['id']
    X_test.drop('id', axis=1, inplace=True)

    print (X_train.shape)
    print (X_train.head())

    model=xg.XGBoostClassifier(num_round=1000 ,nthread=25,  eta=0.02, gamma=1,max_depth=20, min_child_weight=0.1, subsample=0.9,
                                   colsample_bytree=0.5,objective='binary:logistic',seed=1)

    # === training & metrics === #
    train_stacker = [ 0.0  for k in range (0,(X_train.shape[0])) ]
    mean_auc = 0.0
    bagging = 20 # number of models trained with different seeds
    n = 5  # number of folds in strattified cv
    kfolder = StratifiedKFold(y_train, n_folds=n, shuffle=True, random_state=SEED)
    i = 0
    for train_index, test_index in kfolder: # for each train and test pair of indices in the kfolder object
        # creaning and validation sets
        X_train_part, X_cv = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_part, y_cv = np.array(y_train)[train_index], np.array(y_train)[test_index]
        print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))

        # train model and make predictions
        preds = bagged_set(X_train_part, y_train_part, model, SEED , bagging, X_cv, update_seed=True)

        # compute AUC metric for this CV fold
        roc_auc = roc_auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        mean_auc += roc_auc

        no=0
        for real_index in test_index:
                 train_stacker[real_index]=(preds[no])
                 no+=1
        i+=1

    mean_auc/=n
    print (" Average AUC: %f" % (mean_auc) )

    preds = bagged_set(X_train, y_train, model, SEED, bagging, X_test, update_seed=True)
    make_submission('submit4', id_test, preds)

main()
