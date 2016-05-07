from __future__ import division
import pandas as pd
import numpy as np
import XGBoostClassifier as xg
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import StratifiedKFold
import argparse

SEED = 42

# Open .csv file and performs feature processing
def read_data(filename, verbose):
    data = pd.read_csv(filename, sep=',', header=0)

    # Drop redundant column
    data.drop('ROLE_CODE', axis=1, inplace=True)

    # 2nd Order Feature Combination
    data_comb = data
    headers=[f for f in data_comb.columns]
    for i, col in enumerate(headers):
            for i1, col2 in enumerate(headers):
                if i1 <= i or col == 'ACTION' or col2 == 'ACTION' or col == 'id' or col2 == 'id':
                    continue
                else:
                    if (verbose):
                        print 'Combined column name: ' + col + '_' + col2
                    data_comb[col+"_"+col2] = data_comb[[col, col2]].apply(lambda x: int(x[0])  + int(x[1]), axis=1)

    # Add features count
    headers=[f for f in data_comb.columns]
    for i in range(data_comb.shape[1]):
        if headers[i] == 'ACTION' or headers[i] == 'id':
            continue
        else:
            if (verbose):
                print 'Adding counts for column ' + headers[i] + ' with ' + str(len(np.unique(data_comb[headers[i]]))) + 'unique values'
            cnt = data_comb[headers[i]].value_counts().to_dict()
            #cnt = dict((k, -1) if v < 3  else (k,v) for k, v in cnt.items()  ) # if u want to encode rare values as "special"
            data_comb[headers[i]].replace(cnt, inplace=True)

    # OneHotEncoding adds too many features (100s, possibly 1000s) and takes too long to process
    # encode_columns = ['ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_FAMILY']
    # for column in encode_columns:
    #     if (verbose):
    #        print ("Performing OneHotEncoding for " + column)
    #     dummies = pd.get_dummies(data[column], prefix=column)
    #     data = pd.concat([data, dummies], axis=1)
    #     data = data.drop(column, axis=1)

    return data

def make_submission(csv_name, idx, preds):
    submission = pd.DataFrame({ 'id': idx,
                                'ACTION': preds })
    submission.to_csv(csv_name + ".csv", index=False, columns = ['id', 'ACTION'])

# Perform bagging and return the predictions for cross-validation
def estimator_bagging(model, X_train, y_train, X_cv, estimators, seed):
    """ estimator_bagging performs bagging on the model
    using X_train and y_train with (estimators) number of models.
    Then, it returns predictions using X_cv as the input.
    It also slightly modifies the seed for each estimator created.
    """

    predictions = [0.0  for d in range(0, (X_cv.shape[0]))]

    for n in range (0, estimators):
         model.set_params(random_state = seed + n)
         model.fit(X_train, y_train)
         preds = model.predict_proba(X_cv)[:,1]
         for j in range (0, (X_cv.shape[0])):
                 predictions[j] += preds[j]

    for j in range (0, len(predictions)): # divide with number of bags to create an average estimate
                 predictions[j] /= float(estimators)

    return np.array(predictions)

def read_args():
    parser = argparse.ArgumentParser(description='XGBoost model for the Amazon Access Challenge')
    parser.add_argument('threads', help='Number of threads to use', default=50)

    parser.add_argument('-v', '--verbose', help='Increase output verbosity',
                        action='store_true')
    parser.add_argument('-b', '--bagging', help='set bagging parameter',
                        type=int, metavar='num_trees')
    parser.add_argument('-c', '--cross_validation', help='enable n-fold cross validation',
                        type=int, metavar='num_folds')

    args = parser.parse_args()
    return args

def main():
    args = read_args()

    print ('Reading data...')
    X_train = read_data('train.csv', args.verbose)
    y_train = X_train['ACTION']
    X_train.drop('ACTION', axis=1, inplace=True)

    X_test = read_data('test.csv', args.verbose)
    id_test = X_test['id']
    X_test.drop('id', axis=1, inplace=True)


    if (args.verbose):
        print ('')
        print ('Training data: ')
        print ('Shape: ' + str(X_train.shape))
        print (X_train.head())
        print ('')
        print ('Testing data: ')
        print ('Shape: ' + str(X_test.shape))
        print (X_test.head())
        print ('')
    else:
        print ('Data read successfully')

    # Create model
    model = xg.XGBoostClassifier(nthread=args.threads, num_round=1000,
                                 eta=0.02, gamma=1, max_depth=20, subsample=0.9,
                                 min_child_weight=0.1, colsample_bytree=0.5,
                                 objective='binary:logistic', seed=SEED)

    # Initialize training
    num_folds = 5       # number of folds in cv
    num_trees = 10      # number of models trained with different seeds

    if (args.bagging is not None):
        num_trees = args.bagging

    if (args.cross_validation is not None):
        mean_auc = 0.0
        i = 0               # keep track of which CV fold we're working on
        num_folds = args.cross_validation
        folds = StratifiedKFold(y_train, n_folds=num_folds, shuffle=True, random_state=SEED)

        print ('Performing %d-fold cross validation' % (num_folds))
        for train_index, test_index in folds:
            # extract training and validation sets
            X_train_part, X_cv = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_part, y_cv = np.array(y_train)[train_index], np.array(y_train)[test_index]
            if (args.verbose) and (i == 0):
                print ('train size: %d. test size: %d, cols: %d' % ((X_train_part.shape[0]), (X_cv.shape[0]) ,(X_train.shape[1]) ))

            # train model and make predictions
            preds = estimator_bagging(model, X_train_part, y_train_part, X_cv, num_trees, SEED)

            # compute AUC metric for this CV fold
            roc_auc = roc_auc_score(y_cv, preds)
            mean_auc += roc_auc

            if (args.verbose):
                print ('Fold %d/%d AUC : %f' % (i+1, num_folds, roc_auc))
            else:
                print ('Fold %d/%d complete'  % (i+1, num_folds))
            i += 1

        mean_auc /= num_folds
        print ('Average AUC for all CV folds: %f' % (mean_auc) )

    print ('Training...')
    print ('Bagging parameters:')
    print ('    Number of trees: %d' % (num_trees))
    preds = estimator_bagging(model, X_train, y_train, X_test, num_trees, SEED)
    make_submission('finalSubmission', id_test, preds)

    print ('')
    print ('Program complete!')
    print ('')
main()
