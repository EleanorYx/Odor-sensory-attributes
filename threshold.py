import sys
import warnings
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

RAN_SEED = 100


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    maccs_cols = [col for col in df.columns if 'MACCS' in col]

    x = df[maccs_cols]
    y = df['-log#odt']

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=RAN_SEED)
    print("X_train data shape: ", train_x.shape)
    print("X_test data shape: ", test_x.shape)
    print("y_train data shape: ", train_y.shape)
    print("y_test data shape: ", test_y.shape)

    return train_x, test_x, train_y, test_y


def metric_model(model, train_x, test_x, train_y, test_y):
    test_r = model.score(test_x, test_y)
    train_r = model.score(train_x, train_y)

    pred_train_y = model.predict(train_x)
    pred_test_y = model.predict(test_x)
    train_rmse = math.sqrt(mean_squared_error(train_y, pred_train_y))
    test_rmse = math.sqrt(mean_squared_error(test_y, pred_test_y))
    print('R2  Train: {:.4f}'.format(train_r), ' Test: {:.4f}'.format(test_r))
    print('RMSE Train: {:.4f}'.format(train_rmse), ' Test: {:.4f}'.format(test_rmse))


def fit_with_random_forest(train_x, test_x, train_y, test_y):
    print('>> Random Forest')
    rfr = RandomForestRegressor(random_state=RAN_SEED, n_estimators=10, max_depth=11, min_samples_leaf=2,
                                min_samples_split=6, max_features=105)
    rfr.fit(train_x, train_y)
    metric_model(rfr, train_x, test_x, train_y, test_y)


def fit_with_gbdt(train_x, test_x, train_y, test_y):
    print('>> GBDT')
    gbr = GBR(random_state=RAN_SEED, loss="squared_error", n_estimators=50, max_depth=5)
    gbr.fit(train_x, train_y)
    metric_model(gbr, train_x, test_x, train_y, test_y)


def fit_with_mlp(train_x, test_x, train_y, test_y):
    print('>> MLP')
    mlp = MLPRegressor(hidden_layer_sizes=(20,),
                       max_iter=50,
                       activation='relu',
                       alpha=0.4,
                       learning_rate_init=0.03,
                       early_stopping=True,
                       solver='sgd',
                       validation_fraction=0.2,
                       beta_1=0.9,
                       beta_2=0.999,
                       verbose=0,
                       random_state=RAN_SEED)

    mlp.fit(train_x, train_y)
    metric_model(mlp, train_x, test_x, train_y, test_y)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print("load data from file:", file_path)
    else:
        print("No valid dataset file path provided.")
        sys.exit(1)
        
    train_x, test_x, train_y, test_y = load_dataset(file_path)
    fit_with_random_forest(train_x, test_x, train_y, test_y)
    fit_with_gbdt(train_x, test_x, train_y, test_y)
    fit_with_mlp(train_x, test_x, train_y, test_y)