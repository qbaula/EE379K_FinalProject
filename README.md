## EE379K Final Project 

### Background
Our solution to the [Amazon Employee Access Challenge] uses the amazing [XGBoost] library to achieve desirable results.
From the XGBoost docs,
>XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. 

### Usage
Place the datasets train.csv and test.csv in the same directory as amazon\_xgboost.py and XGBoostClassifier.py . Run amazon\_xgboost.py from the command line and specify the number of threads to use for boosting.
```sh
python ./amazon_xgboost.py 25
```
To specify additional arguments, refer to the help message of the program.
```sh
positional arguments:
  threads               Number of threads to use

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity
  -b num_trees, --bagging num_trees
                        set bagging parameter
  -c num_folds, --cross_validation num_folds
                        enable n-fold cross validation
```
- Verbosity will display program status as it processes
- The bagging (**B**ootstrap **agg**regat**ing**) parameter specifies how many trees to create in order to reduce variance and avoid overfitting. A higher bagging parameter will run slower.
- If enabled, cross-validation will train the model with n-folds on the testing data 

### Our Process
To improve our private score on Kaggle, we performed multiple data preprocessing techniques. This includes:
- Feature removal
- 2nd Order Feature Combination
- Adding feature counts as new features

Finally, as a last push of increasing our private score, we performed the following:
- Tweaking hyperparameters through RandomizedSearchCV & GridSearchCV
- Bagging

For more details of our process, please look into the report PDF.

### Warnings
Program takes quite a long time to run (>2 hours!). 
Also, XGBoostClassifier.py does not work across platforms and accross different versions of XGBoost. 

[Amazon Employee Access Challenge]: https://www.kaggle.com/c/amazon-employee-access-challenge
[XGBoost]: https://github.com/dmlc/xgboost