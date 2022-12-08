#Input is a json containing the property URL, city, and state
#Read in the dataset for that city and find the entry corresponding to that url (id)
    #If it has any na's, fill them with 0s or mean
#Read in the pickled pre-trained model for that city (using file structure for state and city)
#Calculate predicted price for that property using that model
#Return a json containing the property's predicted price, actual price, and whether it's overpriced, underpriced, or appropriately priced
#https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

import pickle
import pandas as pd
import json

from sklearn.linear_model import LinearRegression

def generate_prediction(input_json):
    input_dict = json.loads(input_json)

    property_url = input_dict['property_url']
    property_state = input_dict['property_state']
    property_city = input_dict['property_city']

    loaded_model = pickle.load(open(property_state+'/'+property_city+'/'+property_city+'-linear-reg-model.sav','rb'))

    dataset = pd.read_csv(property_state+'/'+property_city+'/'+property_city+'-NO-NA-with-locations.csv')
    prop_id = property_url.split('/')[-1]
    prop_ = dataset[dataset['id'] == int(prop_id)]


    actual_price = prop_['price'][0]

    prop_ = prop_.drop(['neighbourhood', 'neighbourhood_cleansed', 'id','price'], axis="columns").astype('float32')

    predicted_price = loaded_model.predict(prop_)[0]
    price_status = get_price_status(predicted_price,actual_price)

    return json.dumps({'predicted_price':str(predicted_price),'actual_price':str(actual_price),'price_status':str(price_status)})

def get_price_status(predicted_price,actual_price):
    if predicted_price < actual_price - 25:
        return 'overpriced'
    elif predicted_price > actual_price + 25:
        return 'underpriced'
    else:
        return 'good'



if __name__ == '__main__':
    j = json.dumps({'property_url':'https://www.airbnb.com/rooms/45417','property_state':'ca','property_city':'los-angeles'})
    print(generate_prediction(j))
