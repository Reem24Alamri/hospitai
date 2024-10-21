import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def model_initalize():

    return KNeighborsRegressor(metric= 'manhattan', n_neighbors= 11, weights= 'distance')

def train_model(model, X, y):

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X, y)

    return model

def evaluate_model(model, X, y):

    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)

    return print(f'Mean Absolute Error (MAE): {mae}')

def save_model(model, filename='trained_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f'Model has been saved as {filename}')
