import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim

##Validation functions 

def check_id(observation):
    #Checks if observation_id is present
    keys = observation.keys()
    if "observation_id" not in keys:
        error = "observation_id is missing"
        return False, error
    if type(observation["observation_id"]) != str:
        error = "Field `observation_id` must be of type str"
        return False, error
    return True, ""
    
def check_columns(observation):
    #Validates that our observation has all the required columns
    valid_columns = ["observation_id", "Type", "Date", "Part of a policing operation", "Latitude", "Longitude",
    "Gender","Age range", "Officer-defined ethnicity","Legislation","Object of search", "station"]
    
    keys = observation.keys()

    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "The following column(s) are missing: {}".format(missing)
        return False, error
    return True, ""

# Categorical
def check_categorical_data(observation):

    cat_columns = ["Type", "Date",  "Gender", "Age range", "Officer-defined ethnicity",
    "Legislation", "Object of search", "station"]
    invalid_columns = []
    for col in cat_columns:
        if type(observation[col]) != str:
            invalid_columns.append(col)
    if len(invalid_columns) > 0:
        error = "Invalid categorical value in: {}".format(invalid_columns)
        return False, error
    return True, ""
# #Numerical
# def check_numerical_data(observation):

#     num_columns = ["Latitude", "Longitude"]

#     invalid_columns = []
#     for col in invalid_columns:
#         if type(observation[col]) != float:
#             invalid_columns.append(col)
#     if len(invalid_columns) > 0:
#         error = "Invalid numerical value in: {}".format(invalid_columns)
#         return False, error
#     return True, ""
# #Boolean
# def check_boolean_data(observation):

#     column = "Part of a policing operation"

#     if type(observation[column]) != bool:
#         error = "Invalid boolean value in: {}".format(column)
#         return False, error       
#     return  True, ""

def check_type(observation):

    column = "Type"

    valid_type = ['Person search', 'Person and Vehicle search', 'Vehicle search']

    if observation[column] not in valid_type:
        error = "Invalid type of search value"
        return False, error
    return True, ""

def check_gender(observation):
    column = "Gender"

    valid_type = ['Male', 'Female', 'Other']
    if observation[column] not in valid_type:
        error = "Invalid Gender value"
        return False, error
    return True, ""

def check_age(observation):
    column = "Age range"

    valid_type = ['25-34', 'over 34', '10-17', '18-24', 'under 10']
    if observation[column] not in valid_type:
        error = "Invalid Age range value"
        return False, error
    return True, ""

def check_ethnicity(observation):
    column = "Officer-defined ethnicity"

    valid_type = ['White', 'Other', 'Asian', 'Black', 'Mixed']

    if observation[column] not in valid_type:
        error = "Invalid Officer-defined ethnicity value"
        return False, error
    return True, ""

## Functions for data processing and new features

def get_suburb_city(lat, lon):
    geolocator = Nominatim(user_agent="geoapi_app")
    if lat != 0 and lon != 0:
        location = geolocator.reverse(f"{lat},{lon}")
        address = location.raw['address']
        try:
            suburb = address['suburb']
        except KeyError:
            suburb = 'Unknown suburb'
        try:
            city = address['city']
        except KeyError:
            city = 'Unknown city'
    else:
        return ['Unknown suburb', 'Unknown city']
    
    return [suburb, city]

def new_features(df):
    _df = df.copy()

    num_columns = ["Latitude", "Longitude"]
    cat_columns = ["Type", "Gender", "Age range", 
    "Officer-defined ethnicity", "Legislation", "Object of search", "station"]
    bool_columns = ["Part of a policing operation"]

        
    ##Create new features
    #Date column
    _df['Date'] = pd.to_datetime(_df['Date'])
    #new column for the day of the week
    _df['DayOfWeek'] = _df['Date'].dt.day_name()
    #new column for the month
    _df['Month'] = _df['Date'].dt.month_name()
    #Dropping the Date column 
    _df.drop(columns=['Date'], inplace=True)

    #Transform Latitude and Longitude to float type

    #Fill the none values with zero
    _df['Latitude'] = _df['Latitude'].fillna(0)
    _df['Longitude'] = _df['Longitude'].fillna(0)
    #Create new variables city and 
    _df[['suburb', 'city']] = _df.apply(lambda row: pd.Series(get_suburb_city(row['Latitude'], row['Longitude'])), axis=1)
    _df.drop(columns=['Latitude', 'Longitude'], inplace=True)
        
    #Categorical features

    for cat in cat_columns:
        _df[cat] = _df[cat].str.lower()

    _df['Legislation'] = _df['Legislation'].replace({None: "Unknown Legislation"})
    _df['Legislation'] = _df['Legislation'].str.split('(', expand=True)[0].str.strip()

    _df['Part of a policing operation'] = _df['Part of a policing operation'].fillna(None)
    _df['Part of a policing operation'] = _df['Part of a policing operation'].replace({None: False})

        
    return _df

