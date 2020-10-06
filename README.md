# Rossmann-stores-UK-sales-prediction-

Main aim of this project is to predict sales for each 1115 stores daily and the data related to sales is available on Kaggle. The given features are mostly store related and there is only one column which is related to customer which shows the total customer visiting the store. The motivation of this project is the following: by reliably predicting daily sales, store managers may decrease operational costs, and create effective staff schedules (ex. by assigning more staffs during busy days). Also, we would like to identify which type of techniques are both efficient and effective in a real-world sales prediction task. Before building any regression or machine learning models for the prediction, we attempted to identify which features in the data might be significant to determine daily sales of stores, and investigate any interesting relationships between variables. Dependable sales forecasts empower store managers to make viable staff plans that expansion profitability and inspiration.
Machine learning models were used to analyse data and predict future sales or behaviour. The final prediction of sales is the amount. So, regression methods require a linear relationship between input feature and output feature, and may not mode sales prediction behave accurately. It is possible to capture these non-linear relationships with Random Forest and Gradient Boosting algorithms such as XGBoost

## Steps: 
## Data Pre-processing: 
1) We first merged “train.csv” and “store.csv” by “Store” because we can predict daily sales better with more data related to the sales. We also merged “test.csv” and “store.csv” by “Store”. Then, We removed “Customers” field from the training data because it does not exist in the test data so we cannot use “Customers” to predict “Sales”.
2) Open column: Removing all those entries where store is closed as sales is zero. For training the model we have only open store.
3) Sales: Removed sales with value 0, used only Sales bigger than zero.
4) CompetitionDistance: Replacing NaN values with 0.
5) Date: There are lots of dates that need to be handled in the data. Moreover, for our model we have split the date into day, month, year which is passed as an input to our model.
6) Competition Since[X] & Promo2Since[X]: Both columns have a high percentage of missing values and they won't be accurate as indicators, so we have removed features.
7) StateHoliday: Converting all 0(integer) to ”0”(String) because data in the StateHoliday is a mix of numerical and string values. Hence, for the sake of consistency,all values were converted to string.
8) Label Encoding: we performed label encoding on the categorical features present in our dataset, including StateHoliday in train.csv, test.csv and also Store Type, Assortment in store.csv.

## Model Selection: 
We saw that the sales pattern veer off fundamentally from normal conduct. A single model would be hard to deal with the majority of the extraordinary cases these stores show. Thusly, we felt that techniques that outfit a substantial arrangement of models would be best arranged to make expectations in our condition. One methodology that applies is bagging, where subsets of the data are used to shape distinctive models that are then pooled together[3]. Another methodology is to develop various models dependent on differing subsets of the indicators, so all models are not centered on the most compelling indicators, yet can rather catch the subtleties of lesser predictors. Boosting is another technique where each model isn't entrusted with predicting a definitive answer, yet rather on adjusting the residual error of the past model[1]. These strategies structure the reason for a few of the model sorts we used, for example, Random Forests and Boosted Trees. 
We consolidated the improved modelling power from feature selection with the error decrease of tree outfits. We arranged an XGBoost and Random Forest model with our target of limiting RMSPE (Root Mean Squared Percent Error).Models we have prepared:
1. XGBoost
2. Random Forest
