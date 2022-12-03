from django.shortcuts import render
import pickle
import pandas as pd
import os


# Create your views here.

def index(request):
    return render(request, "input.html")


def predict(request):
    property_url = request.POST['link']
    property_state = request.POST['state']
    property_city = request.POST['city']
    
    dir = os.path.dirname(__file__)
    model_filename = os.path.join(dir, property_state,property_city,property_city+'-linear-reg-model.sav')
    
    loaded_model = pickle.load(open(model_filename,'rb'))

    data_filename = os.path.join(dir,property_state,property_city,property_city+'-NO-NA-with-locations.csv')
    dataset = pd.read_csv(data_filename)
    prop_id = property_url.split('/')[-1]
    prop_ = dataset[dataset['id'] == int(prop_id)]


    actual_price = prop_['price'][0]

    prop_ = prop_.drop(['neighbourhood', 'neighbourhood_cleansed', 'id','price'], axis="columns").astype('float32')

    predicted_price = loaded_model.predict(prop_)[0]
    price_status = get_price_status(predicted_price,actual_price)

    return render(request, "result.html", {'predicted_price':str(round(predicted_price)),'actual_price':str(round(actual_price)),'price_status':str(price_status)})


def get_price_status(predicted_price,actual_price):
    if predicted_price < actual_price - 25:
        return 'Overpriced'
    elif predicted_price > actual_price + 25:
        return 'Underpriced'
    else:
        return 'Good Price'