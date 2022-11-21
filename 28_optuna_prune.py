#-----------------------------------------------------------------------------------------#
import pandas as pd
import optuna
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
#------------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------------------#
# pip install optuna
#------------------------------------------------------------------------------------------#

'''
step 1: import file
'''

filepath = '../datasets/save_tabular/data_all_2020.csv'
df = pd.read_csv(filepath)
# NOTE preprocessing data
le = LabelEncoder()
y = le.fit_transform(df['crop_types'])
X = df.drop('crop_types', axis=1)

'''
step 2: define function for optuna
'''

def objective(trial, data=X, target=y):
    train_x, test_x, train_y, test_y = train_test_split(data, target,
                                                        test_size=0.33,
                                                        shuffle=y,
                                                        stratify=y)
    param = {
        'tree_method':'gpu_hist',  
        'booster': 'gbtree',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 1e-1, 1.),
        'subsample': trial.suggest_float('subsample', 1e-1, 1.),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1.),
        'eta': trial.suggest_float('eta', 1e-5, 1.),
        'n_estimators': trial.suggest_categorical('n_estimators', [1, 10, 100, 10000]),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'max_leaves': trial.suggest_int('max_leaves', 3, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)
    }
    model = xgb.XGBClassifier(**param,
                              num_class=4,
                              eval_metric='mlogloss', # cost function
                              gpu_id=1)  
    model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,verbose=False)
    preds = model.predict(test_x)
    # rmse = mean_squared_error(test_y, preds, squared=False) # evaluation matric
    weigthed_f1_score = f1_score(test_y, preds, average='weighted') # evaluation matric
    return weigthed_f1_score

'''
step 3: begin tuning parameters
'''

n_train_iter = 500

sampler = optuna.samplers.TPESampler(seed=42) 
# pruner  = optuna.pruners.SuccessiveHalvingPruner()
pruner  = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=n_train_iter, reduction_factor=3)
study   = optuna.create_study(sampler=sampler, pruner=pruner, direction='minimize')
study.optimize(objective, n_trials=n_train_iter, gc_after_trial=True)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


'''
step 4: plot results 
'''

# NOTE plot_optimization_histor: shows the scores from all trials as well as the best score so far at each point.
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
# NOTE plot_parallel_coordinate: interactively visualizes the hyperparameters and scores
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.show()
# NOTE plot_slice: shows the evolution of the search. You can see where in the hyperparameter space your search went and which parts of the space were explored more.
fig = optuna.visualization.plot_slice(study)
fig.show()
# NOTE plot_contour: plots parameter interactions on an interactive chart. You can choose which hyperparameters you would like to explore.
fig = optuna.visualization.plot_contour(study, params=['alpha',
                                                       'n_estimators', 
                                                       'min_child_weight', 
                                                       'subsample', 
                                                       'learning_rate', 
                                                       'subsample'
                                                      ])
fig.show()
# NOTE plot parameter imprtances
fig = optuna.visualization.plot_param_importances(study)
fig.show()