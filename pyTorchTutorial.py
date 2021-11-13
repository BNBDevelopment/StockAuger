import os
import pandas as pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def pyTorchCleanData():
    aggFilelocation = os.curdir + "\data\\aggregate_CSV.csv"

    allData = pandas.read_csv(aggFilelocation)

    #We need to scale the data so that its normalized with respect to the range -1,1
    scaler = MinMaxScaler(feature_range=(-1,1))
    normalizedPriceData = allData[['Close']]
    #Throws a warning but not error - NEEDS FIX
    normalizedPriceData['Close'] = scaler.fit_transform(normalizedPriceData['Close'].values.reshape(-1,1))


    #lookback number must be greater than number of data rows
    splitDataTraningTest(normalizedPriceData, 1)

#in this case, howFarLookback is how many days in the past we look to predict the next (called the sliding window method)
def splitDataTraningTest(stock, howFarLookback):
    data = []

    #convert out input stock to an array (numpy array)
    data_raw = stock.to_numpy()
    print("------------")
    print(data_raw)
    #Create all possible windows of analysis given the howFarLookback param)
    for index in range(len(data_raw) - howFarLookback):
        data.append(data_raw[index: index + howFarLookback])
    print("DATA")
    print(data)





