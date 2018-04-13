import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


iowa_file_path = 'train.csv'
iowa_data = pd.read_csv(iowa_file_path)

print(iowa_data.describe())

print(iowa_data.columns)

iowa_price_data = iowa_data.SalePrice

print(iowa_price_data.head())

columns_of_interest = ['LotArea', 'GarageCars']

two_columns_of_data = iowa_data[columns_of_interest]
print(two_columns_of_data.describe())

y = iowa_data.SalePrice

iowa_predictors = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

X = iowa_data[iowa_predictors]

iowa_model = DecisionTreeRegressor()

iowa_model.fit(X, y)


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(iowa_model.predict(X.head()))

predicted_home_prices = iowa_model.predict(X)

mean_absolute_error(y, predicted_home_prices)


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

iowa_model = DecisionTreeRegressor()

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))


def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return (mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))


forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
iowa_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, iowa_preds))


test = pd.read_csv('test.csv')
test_x = test[iowa_predictors]
predicted_prices = forest_model.predict(test_x)
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test.Id, 'SalesPrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)