from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
data = pd.read_excel(r"C:\Users\JQ\Desktop\ending\data.xlsx")
data = data.drop(["Unnamed: 0"], axis = 1)
imputer = KNNImputer(n_neighbors=5)
data_c  = data[data.isna().T.any() == False]
data_na = data[data.isna().T.any() == True]
data_test = data_c.sample(frac=322*0.3/data_c.shape[0], random_state=2324)
data1 = data_c[~data_c.index.isin(data_test.index)]
data_train = data_na.append(data1)
data_train.to_excel(r"C:\Users\JQ\Desktop\3-1\data_train.xlsx")
data_train[data_train.columns[:-1]] = imputer.fit_transform(data_train[data_train.columns[:-1]])

data_train.to_excel(r"C:\Users\JQ\Desktop\3-1\data_traink.xlsx")
data_test.to_excel(r"C:\Users\JQ\Desktop\3-1\data_test.xlsx")

