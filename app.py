import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (Model, IntegerField, FloatField,TextField, IntegrityError, BooleanField)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
#for pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from category_encoders import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV
from utils import *


########################################
# Begin database

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    predicted_outcome = BooleanField()
    outcome = BooleanField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

with open('pipeline.pickle', 'rb') as fh:
    pipeline = joblib.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver

app = Flask(__name__)


@app.route('/should_search/', methods=['POST'])
def should_search():
    obs_dict = request.get_json()
    ##start validations
    request_ok, error = check_id(obs_dict)
    if not request_ok:
        response = {'observation_id': None, 'error': error}
        print(response)
        return jsonify(response)
    #Defining id
    _id = obs_dict['observation_id']

    request_ok, error = check_columns(obs_dict)
    if not request_ok:
        response = {'observation_id': _id, 'error': error}
        print(response)
        return jsonify(response)

    request_ok, error = check_categorical_data(obs_dict)
    if not request_ok:
        response = {'observation_id': _id, 'error': error}
        print(response)
        return jsonify(response)

    request_ok, error = check_type(obs_dict)
    if not request_ok:
        response = {'observation_id': _id, 'error': error}
        print(response)
        return jsonify(response)

    request_ok, error = check_gender(obs_dict)
    if not request_ok:
        response = {'observation_id': _id, 'error': error}
        print(response)
        return jsonify(response)

    request_ok, error = check_age(obs_dict)
    if not request_ok:
        response = {'observation_id': _id, 'error': error}
        print(response)
        return jsonify(response)

    request_ok, error = check_ethnicity(obs_dict)
    if not request_ok:
        response = {'observation_id': _id, 'error': error}
        print(response)
        return jsonify(response)

    ##end of validations

    ##start transformations
    obs_dict = transform_numerical_data(obs_dict)
    obs_dict = transform_operation(obs_dict)
    obs_dict = transform_legislation(obs_dict)
 

    obs = pd.DataFrame([obs_dict])
    obs = new_features(obs)
    obs = obs[columns].astype(dtypes)
    ##end transformations

    #predicted = pipeline.predict(obs)[0]
    proba = pipeline.predict_proba(obs)[0, 1]
    threshold = 0.398
    if proba >= threshold:
        predicted = True
    else:
        predicted = False

    response = {'outcome': bool(predicted)}

    p = Prediction(
        observation_id = _id,
        predicted_outcome = bool(predicted),
        observation = obs_dict)
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: ID: '{}' already exists".format(_id)
        response = {'error': error_msg}
        DB.rollback()
    
    return jsonify(response)

@app.route('/search_result/', methods=['POST'])
def search_result():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.outcome = bool(obs['outcome'])
        p.save()
        response = {}
        for key in ['observation_id', 'outcome', 'predicted_outcome']:
            response[key] = model_to_dict(p)[key]
        return jsonify(response)
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg})


# End webserver
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)