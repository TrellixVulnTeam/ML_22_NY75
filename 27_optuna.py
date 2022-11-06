import pandas as pd
import optuna
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
        # 'n_estimators': 10000,
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)
    }
    model = xgb.XGBClassifier(**param, gpu_id=1)  
    model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100, verbose=False)
    preds = model.predict(test_x)
    rmse = mean_squared_error(test_y, preds, squared=False)
    return rmse

'''
step 3: begin tuning parameters
'''

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)
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
                            'min_child_weight',
                            'subsample',
                            'learning_rate',
                            'subsample'])
fig.show()
# NOTE plot parameter imprtances
fig = optuna.visualization.plot_param_importances(study)
fig.show()