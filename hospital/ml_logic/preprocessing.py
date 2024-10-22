import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.metrics import mean_absolute_error,confusion_matrix, ConfusionMatrixDisplay,classification_report, precision_recall_curve
from sklearn.model_selection import cross_validate, cross_val_score,train_test_split, learning_curve, cross_val_predict



def sk_learn_proc(X: pd.DataFrame):

    # creation of sin and cos features
    # month_dict = {
    #     'Jan':1,
    #     'Feb':2,
    #     'Mar':3,
    #     'Apr':4,
    #     'May':5,
    #     'Jun':6,
    #     'Jul':7,
    #     'Aug':8,
    #     'Sep':9,
    #     'Oct':10,
    #     'Nov':11,
    #     'Dec':12
    # }
    # X['month_nb'] = X['month year'].apply(lambda x: month_dict[x[:3]])
    # months_in_a_year = 12
    # X['sin_admission'] = np.sin(2 * np.pi * (X['month_nb'] - 1) / months_in_a_year)
    # X['cos_admission'] = np.cos(2 * np.pi * (X['month_nb'] - 1) / months_in_a_year)
    # X.drop(columns=['month year', 'month_nb'], inplace=True)

    num_transformer = RobustScaler()
    cat_transformer = OneHotEncoder(drop='if_binary', sparse_output=False)

    preprocessor = ColumnTransformer([
        ('num_transformer', num_transformer, ['AGE', 'HB', 'TLC', 'PLATELETS', 'GLUCOSE', 'UREA', 'CREATININE', 'EF']),
        ('cat_transformer', cat_transformer, ['GENDER', 'RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD'])],
        remainder='passthrough')

    X_proc = preprocessor.fit_transform(X)
    cleaned_column_names = [name.split('__')[-1] for name in preprocessor.get_feature_names_out()]

    X_proc = pd.DataFrame(X_proc, columns=cleaned_column_names)
    X_proc.rename(
        columns={
            'GENDER_M':'GENDER',
            'RURAL_U':'RURAL',
            'TYPE OF ADMISSION-EMERGENCY/OPD_O':'TYPE OF ADMISSION-EMERGENCY/OPD'
        },
        inplace=True
    )

    return X_proc
