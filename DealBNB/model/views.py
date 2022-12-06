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
    
    cities = ["los-angeles", "oakland", "san-francisco"]
    
    if property_state == "ca" and property_city not in cities:
        return render(request, "error.html", {'error_message': f"{property_city} is not in ca. Please select either: los-angeles, oakland, or san-francisco."})
    elif property_state == "dc" and property_city != "washington-dc":
        return render(request, "error.html", {'error_message': f"{property_city} is not in dc. Please select washington-dc."})
    elif property_state not in ['ca', 'dc']:
        return render(request, "error.html", {'error_message': f"{property_state} is not an available state. Try searching in ca or dc."})


    dir = os.path.dirname(__file__)
    model_filename = os.path.join(dir, property_state,property_city,property_city+'-linear-reg-model.sav')
    
    loaded_model = pickle.load(open(model_filename,'rb'))

    data_filename = os.path.join(dir,property_state,property_city,property_city+'-NO-NA-with-locations.csv')
    dataset = pd.read_csv(data_filename)
    prop_id = property_url.split('/')[-1]
    prop_id = prop_id.split('?')[0]
    
    prop_ = dataset[dataset['id'] == int(prop_id)]
    if (len(prop_) == 0):
        return render(request, "error.html", {'error_message': f"Property not found in the dataset, please try again with another property."})

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