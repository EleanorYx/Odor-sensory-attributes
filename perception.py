import sys
import warnings
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBC
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')


types = ['sulfurous', 'fatty', 'citrus', 'camphor',
         'medicinal', 'ammonical',
         'musty']


def show_confusion_matrix(model, train_x, train_y, test_x, test_y):
    pred_train_y = model.predict(train_x)
    pred_test_y = model.predict(test_x)

    cm_train = confusion_matrix(train_y, pred_train_y)
    cm_test = confusion_matrix(test_y, pred_test_y)

    print('Train:')
    print(cm_train)
    print('Test:')
    print(cm_test)


def load_dataset(file_path):
    df = pd.read_csv(file_path)

    maccs_cols = [col for col in df.columns if 'MACCS' in col]

    X = df[maccs_cols]
    y = df['Smell Percepts']

    # init LabelEncoder
    label_encoder = LabelEncoder()

    # transform label
    y = label_encoder.fit_transform(y)

    # class names
    class_names = label_encoder.classes_

    X_train, X_test, y_train, y_test = train_test_split(df[maccs_cols], y, test_size=0.2, random_state=100)
    print("X_train data shape: ", X_train.shape)
    print("X_test data shape: ", X_test.shape)
    print("y_train data shape: ", y_train.shape)
    print("y_test data shape: ", y_test.shape)

    return df, X_train, X_test, y_train, y_test


def fit_with_random_forest(X_train, X_test, y_train, y_test):
    # Random Forest
    rf = RandomForestClassifier(n_jobs=-1, random_state=100, n_estimators=300, max_depth=8, min_samples_split=5,
                                min_samples_leaf=1, class_weight='balanced')

    rf.fit(X_train, y_train)
    print('>> Random Forest')
    show_confusion_matrix(rf, X_train, y_train, X_test, y_test)


def fit_with_gbdt(df, X_train, X_test, y_train, y_test):
    # GBDT
    class_counts = df['Smell Percepts'].value_counts()
    total_samples = len(df)
    sample_weights = df['Smell Percepts'].apply(lambda x: total_samples / class_counts[x])

    gbc = GBC(random_state=100, n_estimators=40, learning_rate=0.06, max_depth=3, subsample=0.7, min_samples_leaf=1,
              min_samples_split=2, max_features=103, min_impurity_decrease=0.1)
    gbc.fit(X_train, y_train, sample_weight=sample_weights[X_train.index])

    print('>> GBDT')
    show_confusion_matrix(gbc, X_train, y_train, X_test, y_test)


def fit_with_mlp(X_train, X_test, y_train, y_test):
    smote = SMOTE(random_state=100)
    X_train_b, y_train_b = smote.fit_resample(X_train, y_train)
    print('>> MLP')
    mlp = MLPClassifier(hidden_layer_sizes=(100,),
                        max_iter=50,
                        alpha=1,
                        learning_rate_init=0.01,
                        early_stopping=True,
                        solver='adam',
                        verbose=0,
                        random_state=100)

    # fitting
    mlp.fit(X_train_b, y_train_b)
    show_confusion_matrix(mlp, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print("load data from file:", file_path)
    else:
        print("No valid dataset file path provided.")
        sys.exit(1)

    df, X_train, X_test, y_train, y_test = load_dataset(file_path)
    print(f"types: {types}")
    fit_with_random_forest(X_train, X_test, y_train, y_test)
    fit_with_gbdt(df, X_train, X_test, y_train, y_test)
    fit_with_mlp(X_train, X_test, y_train, y_test)
