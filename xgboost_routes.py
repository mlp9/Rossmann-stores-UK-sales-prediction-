import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
import datetime
from datetime import timedelta

from utils import onehotCategorical

app = Flask(__name__,template_folder='templates')

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    model = joblib.load('model.pkl')
    if request.method=='POST':
        
        entered_li = []
        
        month = request.form['Month']
        #day = int(request.form['Day'])
        promo = int(request.form['Promo'])
        #promo2 = int(request.form['Promo2'])
        stateH = int(request.form['StateH'])
        schoolH = int(request.form['SchoolH'])
        assortment = int(request.form['Assortment'])
        storeType = int(request.form['StoreType'])
        store = int(request.form['store'])
        
        date_entry = month
        year, month, day = map(int, date_entry.split('-'))

        # one-hot encode categorical variables
        stateH_encode = onehotCategorical(stateH, 4)
        assortment_encode = onehotCategorical(assortment, 3)
        storeType_encode = onehotCategorical(storeType, 4)
        store_encode = onehotCategorical(store, 1115, store=1)

        comp_dist = 5458.1
        entered_li.extend(store_encode)
        entered_li.extend(storeType_encode)
        entered_li.extend(assortment_encode)
        entered_li.extend(stateH_encode)
        entered_li.extend([comp_dist])
        #entered_li.extend([promo2])
        entered_li.extend([promo])
        entered_li.extend([day])
        entered_li.extend([month])
        entered_li.extend([schoolH])
        
        data = [[store,1270,promo,schoolH,storeType,assortment,stateH,6,day,month,year,50,132,0,0]]
        df = pd.DataFrame(data,columns=['Store','CompetitionDistance','Promo','SchoolHoliday','StoreType','Assortment',
                                        'StateHoliday','DayOfWeek','Month','Day','Year','WeekOfYear','CompetitionOpen','PromoOpen','IsPromoMonth'])
        #data = [[1,1270,1,0,3,1,0,6,9,15,2019,37,132,0,0]]
        #df = pd.DataFrame(data,columns=['Store','CompetitionDistance','Promo','SchoolHoliday','StoreType','Assortment','StateHoliday','DayOfWeek','Month','Day','Year','WeekOfYear','CompetitionOpen','PromoOpen','IsPromoMonth'])     
        prediction = model.predict(xgb.DMatrix(df))
        prediction=np.expm1(prediction)
        #prediction = model.predict(entered_li.values.reshape(1, -1))
        label = "$"+str(np.squeeze(prediction.round(2)))
        return render_template('index.html', label=label)

if __name__ == '__main__':
    # start API
    app.run()
