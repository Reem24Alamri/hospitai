import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.metrics import mean_absolute_error,confusion_matrix, ConfusionMatrixDisplay,classification_report, precision_recall_curve
from sklearn.model_selection import cross_validate, cross_val_score,train_test_split, learning_curve, cross_val_predict



def sk_learn_proc(df: pd.DataFrame):

    X = df.drop(columns='DURATION OF STAY')
    y = df['DURATION OF STAY']

    num_transformer = RobustScaler()
    cat_transformer = OneHotEncoder(drop='if_binary', sparse_output=False)

    preprocessor = ColumnTransformer([
        ('num_transformer', num_transformer, ['AGE', 'HB', 'TLC', 'PLATELETS', 'GLUCOSE', 'UREA', 'CREATININE', 'EF']),
        ('cat_transformer', cat_transformer, ['GENDER', 'RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD'])],
        remainder='passthrough')

    X_proc = pd.DataFrame(preprocessor.fit_transform(X), columns=list(X.columns))

    return (X_proc,y)
