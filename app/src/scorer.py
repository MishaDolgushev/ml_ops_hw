import pandas as pd
import joblib

# Make prediction
def make_pred(dt, path_to_file):

    print('Importing pretrained model...')
    # Import model
    model = joblib.load('./models/model.pkl')
    # model.load_model('./models/my_catboost_model.cbm')

    # Define optimal threshold
    model_th = 0.3147

    # Make submission dataframe
    submission = pd.DataFrame({
        'client_id':  pd.read_csv(path_to_file)['client_id'],
        'preds': (model.predict(dt).data[:, 0] > model_th) * 1
    })
    print('Prediction complete!')

    # Return proba for positive class
    return submission