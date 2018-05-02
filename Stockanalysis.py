import quandl
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing,svm,model_selection
import matplotlib.pyplot as plt
from matplotlib import style

api_key = "Give your API key"

def grab_data():
    main_df=pd.DataFrame
    fifty_states=pd.read_html("https://simple.wikipedia.org/wiki/List_of_U.S._states")

    for abbv in fifty_states[0][1][1:]:
        df = quandl.get("FMAC/HPI_"+str(abbv), authtoken=api_key)
        df.columns=[str(abbv)]
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100.0

        if main_df.empty:
            main_df=df
        else:
            main_df=main_df.join(df)

    return main_df

def HPI_Benchmark():
    df = quandl.get("FMAC/HPI_USA", authtoken=api_key)
    df.columns = ["United States"]
    df["United States"] = (df["United States"] - df["United States"][0]) / df["United States"][0] * 100.0
    df.rename(columns={'United States': 'US_HPI'}, inplace=True)
    return df


def mortgage_30y():
    df = quandl.get("FMAC/MORTG", trim_start="1975-01-01", authtoken=api_key)
    df["Value"] = (df["Value"] - df["Value"][0]) / df["Value"][0] * 100.0
    df = df.resample('1D').mean()
    df = df.resample('M').mean()
    return df


def sp500_data():
    df = pd.read_csv("sp500.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df["Adj Close"] = (df["Adj Close"] - df["Adj Close"][0]) / df["Adj Close"][0] * 100.0
    df.set_index('Date', inplace=True)
    df = df.resample('M').mean()
    df.rename(columns={'Adj Close': 'sp500'}, inplace=True)
    df = df['sp500']
    return df


def gdp_data():
    df = quandl.get("BCB/4385", trim_start="1975-01-01", authtoken=api_key)
    df["Value"] = (df["Value"] - df["Value"][0]) / df["Value"][0] * 100.0
    df = df.resample('M').mean()
    df.rename(columns={'Value': 'GDP'}, inplace=True)
    df = df['GDP']
    return df


def us_unemployment():
    df = quandl.get("ECPI/JOB_G", trim_start="1975-01-01", authtoken=api_key)
    df["Unemployment Rate"] = (df["Unemployment Rate"] - df["Unemployment Rate"][0]) / df["Unemployment Rate"][
        0] * 100.0
    df = df.resample('1D').mean()
    df = df.resample('M').mean()
    return df


def create_labels(cur_hpi,fut_hpi):
    if fut_hpi>cur_hpi:
        return 1
    else:
        return 0
        
House_data=grab_data()
m30 = mortgage_30y()
sp500 = sp500_data()
gdp = gdp_data()
HPI_Bench = HPI_Benchmark()
#unemployment = us_unemployment()
m30.columns = ['M30']
housing_data = House_data.join([HPI_Bench,m30, sp500, gdp])
housing_data.dropna(inplace=True)
housing_data=housing_data.pct_change()
housing_data.replace([np.inf,-np.inf],np.nan,inplace=True)
housing_data.dropna(inplace=True)
housing_data['fut_hpi']=housing_data['US_HPI'].shift(-1)
housing_data['label']=list(map(create_labels,housing_data['US_HPI'],housing_data['fut_hpi']))

X=np.array(housing_data.drop(['label','fut_hpi'],1))
X=preprocessing.scale(X)
y=np.array(housing_data['label'])

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y)

clf=svm.SVC(kernel='rbf')
clf.fit(X_train,y_train)

print(clf.score(X_test,y_test))
