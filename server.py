from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import json

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def home():
    return render_template('app.html')

@app.route('/predict_home_price', methods=["GET", "POST"])
def predict_home_price():

    if request.method == "POST":

        filename = 'bengaluru_home_prices_model.pickle'
        regressor = pickle.load(open(filename, 'rb'))

        filename1 = 'columns.json'
        data_columns = json.load(open(filename1, 'r'))
        data_columns = data_columns['data_columns']


        total_sqft = int(request.form["total_sqft"])
        location = request.form["location"]
        house_type_clean = int(request.form["house_type_clean"])
        bath_clean = int(request.form["bath_clean"])
        balcony_clean = int(request.form["balcony_clean"])
        area_type = request.form["area_type"]

        data_columns = [x.lower().strip() for x in data_columns]

        for i in range(len(data_columns)):
            if  data_columns[i] == location:
                loc_index = i

        dict_area_type = {'Built-up  Area': 1, 'Carpet  Area':0 , 'Plot  Area': 3, 'Super built-up  Area': 2}
        area_type_clean = dict_area_type.get(area_type)

        x = np.zeros(len(data_columns))
        x[0] = total_sqft
        x[1] = house_type_clean
        x[2] = bath_clean
        x[3] = balcony_clean
        x[4] = area_type_clean

        if loc_index >= 0:
            x[loc_index] = 1

        p = np.float128(regressor.predict([x])[0])
        return render_template('app.html', pred_price_disp = ('The house costs: {:.2f} lakhs'.format(p)))

    return render_template('app.html')

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    #print(int(np.where(data_columns == '1st phase jp nagar')[0][0]))
    #util.load_saved_artifacts()
    #print(data_columns)
    app.run(debug = True)
