import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.metrics import mean_absolute_error,confusion_matrix, ConfusionMatrixDisplay,classification_report, precision_recall_curve
from sklearn.model_selection import cross_validate, cross_val_score,train_test_split, learning_curve, cross_val_predict



def sk_learn_proc(df):

    X = df.drop(columns='DURATION OF STAY')
    y = df['DURATION OF STAY']

    num_transformer = RobustScaler()
    cat_transformer = OneHotEncoder(drop='if_binary', sparse_output=False)

    preprocessor = ColumnTransformer([
        ('num_transformer', num_transformer, ['AGE', 'HB', 'TLC', 'PLATELETS', 'GLUCOSE', 'UREA', 'CREATININE', 'EF']),
        ('cat_transformer', cat_transformer, ['GENDER', 'RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD'])],
        remainder='passthrough')

    # df_numerical = df[numericals]
    # df_numerical = rob.fit_transform(df_numerical)
    # df_numerical = pd.DataFrame(data=df_numerical, columns=['AGE', 'HB', 'TLC', 'PLATELETS', 'GLUCOSE', 'UREA', 'CREATININE', 'EF'])

    # df[numericals] = df_numerical.reset_index(drop=True)

    # df_non_numerical = df.select_dtypes(exclude=['number'])

    # enc_binary = OneHotEncoder(drop='if_binary', sparse_output=False)
    # enc_binary.fit(df_non_numerical)

    # df_non_numerical[enc_binary.get_feature_names_out()] = enc_binary.transform(df_non_numerical)

    # df_non_numerical_cols = df_non_numerical.iloc[:, :3].columns

    # df_non_numerical = df_non_numerical.iloc[:, 3:]

    # df_non_numerical.columns = df_non_numerical_cols



    # df_numerical_reset = df_numerical.reset_index(drop=True)
    # df_non_numerical_reset = df_non_numerical.reset_index(drop=True)
    # df_sin_cos = df[['sin_admission', 'cos_admission']].reset_index(drop=True)

    # treated_cols = list(df_numerical_reset.columns) + list(df_non_numerical_reset.columns) + list(df_sin_cos.columns)

    # df_other_cols = df.drop(columns=treated_cols)
    # df_other_cols_reset = df_other_cols.reset_index(drop=True)

    # df_processed = pd.concat([df_numerical_reset, df_non_numerical_reset, df_sin_cos, df_other_cols_reset], axis=1)
    X_proc = pd.DataFrame(preprocessor.fit_transform(X), columns=list(X.columns))







    return (X_proc,y)
