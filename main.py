import pandas as pd
import numpy as np # data pre-processing

import seaborn as sns
import matplotlib.pyplot as plt # data visualization

from sklearn.model_selection import train_test_split
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
# RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def load_dataset(file_name):
    return pd.read_csv(file_name)
def dataset_info(df):
    print("Data set file coloumns Info")
    print(df.info())
    print("Data set shape")
    print(df.shape)
    
def dataset_missing_data(df):
    df.isnull().sum() # checks for missing data

def clean_fix_values(df):
    df_1= df.dropna()
    df_1.reset_index(drop=True, inplace=True)
    df_2=df.fillna(df.mean()) # cleaning the data & fixing the missing values
def plot_coloumns_data(df):
    df.hist(figsize=(20,15))
    plt.show()

def normalize_data(df):
    #     
    df['ph'] = df['ph'].fillna(df.groupby('Potability')['ph'].transform('mean'))
    df['Sulfate'] = df['Sulfate'].fillna(df.groupby('Potability')['Sulfate'].transform('mean'))
    df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df.groupby('Potability')['Trihalomethanes'].transform('mean'))
    df.isna().sum()
    # 

    df['ph']= df['ph']/max(df['ph'])
    df['Hardness']= df['Hardness']/max(df['Hardness'])
    df['Solids']= df['Solids']/max(df['Solids'])
    df['Chloramines']= df['Chloramines']/max(df['Chloramines'])
    df['Sulfate']= df['Sulfate']/max(df['Sulfate'])
    df['Conductivity']= df['Conductivity']/max(df['Conductivity'])
    df['Organic_carbon']= df['Organic_carbon']/max(df['Organic_carbon'])
    df['Trihalomethanes']= df['Trihalomethanes']/max(df['Trihalomethanes'])
    df['Turbidity']= df['Turbidity']/max(df['Turbidity'])
    return df

def create_test_train_data(df):
    X_train, X_test, y_train, y_test = train_test_split( df.drop(['Potability'],axis=1), 
                                                        df.Potability, test_size=0.2, random_state=0,
                                                        stratify = df.Potability)
    return X_train, X_test, y_train, y_test

def logistic_regression_learn(X_train, X_test, y_train, y_test):
    clf= LogisticRegression()
    params= {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
    clf_grid= GridSearchCV(estimator= clf, param_grid= params, cv=5)
    clf_grid.fit(X_train,y_train)
    
    y_pred= clf_grid.predict(X_test)
    acc= accuracy_score(y_test, y_pred)
    print('Accuracy for Logistic Regression: ', acc)
    
def random_forest_learn(X_train, X_test, y_train, y_test):
    # 
    
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred2= classifier.predict(X_test)
    acc2= accuracy_score(y_test, y_pred2)
    print('Accuracy for RandomForest Classifier is: ', acc2)
def K_neighbor_learn(X_train, X_test, y_train, y_test):
    classifier=KNeighborsClassifier()

    params1 = {
        'n_neighbors': (1,10, 1),
        'leaf_size': (20,40,1),
        'p': (1,2),
        'weights': ('uniform', 'distance'),
        'metric': ('minkowski', 'chebyshev'),}
    clf2_grid= GridSearchCV(estimator= classifier, param_grid= params1, cv=5)
    clf2_grid.fit(X_train,y_train)
    y_pred3= clf2_grid.predict(X_test)
    acc3= accuracy_score(y_test, y_pred3)
    print('Accuracy for K neighbor classifier is: ', acc3)

if __name__ == "__main__":
    df = load_dataset('water_potability.csv')
    dataset_info(df)
    dataset_missing_data(df)
    clean_fix_values(df)
    plot_coloumns_data(df)
    #machine learning
    df = normalize_data(df)
    X_train, X_test, y_train, y_test = create_test_train_data(df)
    logistic_regression_learn(X_train, X_test, y_train, y_test)
    K_neighbor_learn(X_train, X_test, y_train, y_test)
    random_forest_learn(X_train, X_test, y_train, y_test)

