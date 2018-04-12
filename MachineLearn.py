import pandas as pd
from sklearn.tree import DecisionTreeRegressor


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

iowa_predictors = ['LotArea', 'YearBuilt', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'PoolArea']

X = iowa_data[iowa_predictors]

iowa_model = DecisionTreeRegressor()

iowa_model.fit(X, y)


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")

print(iowa_model.predict(X.head()))
