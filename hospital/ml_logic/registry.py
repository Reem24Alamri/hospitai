import pickle

def load_model():
    with open('knn_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    return loaded_model
