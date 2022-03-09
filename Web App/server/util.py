import pickle
import numpy as np
import pandas as pd
import requests
import json

__model = None
__columns = None

def load_artifacts():
    global __model
    global __columns
    global __name

    with open('Web App/server/artifacts/columns.json','r') as f:
        __columns = json.load(f)['data_columns']
        __name=__columns[:22]


def predict_price(name,transmission,fuel,owner,year,km_driven,engine,max_power):
    global __model
    global __columns
    x = []
    x[:26] = np.zeros(32,dtype='int32')
    x[27] = owner
    x[28] = year
    x[29] = km_driven
    x[30] = engine
    x[31] = max_power
    name_index = __columns.index(name.lower())
    transmission_index = __columns.index(transmission.lower())
    fuel_index = __columns.index(fuel.lower())

    
    if name_index>=0:
        x[name_index] = 1
    if transmission_index>=0:
        x[transmission_index] = 1
    if fuel_index>=2:
        x[fuel_index] = 1

#Integrate MLflow part(model serve in the web page)   
    price = pd.DataFrame(x).transpose()
    host = 'localhost'
    port = '5001'
    url = f'http://{host}:{port}/invocations'
    headers = {'Content-Type': 'application/json',}
    # test contains our data from the original train/valid/test split
    http_data = price.to_json(orient='split')
    r = requests.post(url=url, headers=headers, data=http_data)
    price1=r.text.replace('[','')
    price1=price1.replace(']','')
    price1=float(price1)
    return format(round(price1,2))  
    
def get_car_name():
    return __name
if __name__ == '__main__':
    
    load_artifacts()
    print(get_car_name())
    print(predict_price('Tata','Manual','Diesel',1,2013,20000,2050,150))