print("machine")

import pandas as pd

melb = pd.read_csv("melb_data.csv")  

melb.head()
print(melb.head())

melb.describe()

melb.columns

melb = melb.dropna(axis=0)

y = melb.Price
print(y)

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']

x = melb[melbourne_features]

x.describe()
x.head()

from sklearn.tree import DecisionTreeRegressor

melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(x, y)

print("Predcting the values of the following houses: ")
print(x.head())
print("The predictions are: ")
print(melbourne_model.predict(x.head()))

from sklearn.metrics import mean_absolute_error

predictedprices = melbourne_model.predict(x)
mean_absolute_error(y, predictedprices)

from sklearn.model_selection import train_test_split


#split
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)

#train on training data
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_x, train_y)

#validate on validate data, predict what the oucomes using val x data and then see if it lines up with real val y data

predicted_with_valxdata = melbourne_model.predict(val_x)

#how far are we off from val y with our predictedwithvalxdata
print(mean_absolute_error(val_y, predicted_with_valxdata))


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor



def MAEDecision(max_leaf_nodes, train_x, val_x, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_x, train_y)
    modelpredict = model.predict(val_x)
    mae = mean_absolute_error(val_y, modelpredict)
    return(mae)


Numbers_of_leaves = [5, 10, 100, 500, 5000]

for max_leaf_nodes in Numbers_of_leaves:
    my_mae = MAEDecision(max_leaf_nodes, train_x, val_x, train_y, val_y)
    print(f'Max leaf node = {max_leaf_nodes} \t Mean absolute error = {my_mae}')



bestmodel = DecisionTreeRegressor(max_leaf_nodes=500, random_state=0)
bestmodel.fit(train_x, train_y)
bestmodelprediction = bestmodel.predict(val_x)

print(mean_absolute_error(val_y, bestmodelprediction))

