import marimo

__generated_with = "0.4.10"
app = marimo.App(width="full")


@app.cell
def __():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error,r2_score
    return (
        LinearRegression,
        mean_squared_error,
        np,
        pd,
        r2_score,
        train_test_split,
    )


@app.cell
def __(pd):
    data_train=pd.read_csv('/home/anuragverma/Desktop/Files_Desktop/FSDI/Kaggle Training/House_Prediction/house-prices-advanced-regression-techniques/train.csv')
    data_test=pd.read_csv('/home/anuragverma/Desktop/Files_Desktop/FSDI/Kaggle Training/House_Prediction/house-prices-advanced-regression-techniques/test.csv')
    return data_test, data_train


@app.cell
def __(data_test, data_train):
    print('No. of Rows in original train data is :',data_train.shape[0] )
    print('No. of Columns in original train data is:',data_train.shape[1])
    print('No. of Rows in original test data is :',data_test.shape[0])
    print('No. of Columns in original test data is :',data_test.shape[1])

    return


@app.cell
def __(data_test, data_train):
    #Preparing X_train and Y data for training
    #Removed SalePrice and Id columns
    #Removed Non-Int64 and Non-Float64 columns in X_Train
    #Removed Null and replaced with 0 
    X_columns=[]
    for i in data_train.columns:
        X_columns.append(i)
    X_columns.remove('SalePrice')
    X_columns.remove('Id')
    X=data_train[X_columns]
    X_Train=X.select_dtypes(include=['int64','float64'])
    X_Train=X_Train.fillna(0)
    Y=data_train['SalePrice']

    #Preprocessing Test data
    #Removed Id Column
    #Removed Non-Int64 and Non-Float64 columns in X_Train
    #Removed Null and replaced with 0 

    X_Test_columns=[]
    for i in data_test.columns:
        X_Test_columns.append(i)

    X_Test_columns.remove('Id')
    X_Test=data_test[X_Test_columns]
    X_Test=X_Test.select_dtypes(include=['int64','float64'])
    X_Test=X_Test.fillna(0)

    print('No. of Rows in  X is :',X.shape[0] )
    print('No. of Columns X is:',X.shape[1])
    print('No. of Rows in  X_Train is :',X_Train.shape[0] )
    print('No. of Columns X_Train is:',X_Train.shape[1])
    print('No. of Rows and column in training data Y is :',Y.shape)

    print('No. of Rows in  X_Test is :',X_Test.shape[0] )
    print('No. of Columns X_Test is:',X_Test.shape[1])
    return X, X_Test, X_Test_columns, X_Train, X_columns, Y, i


@app.cell
def __(LinearRegression, X_Train, Y):
    #Fitting model on X_Train and Y
    model=LinearRegression()
    model.fit(X_Train,Y)
    return model,


@app.cell
def __(X_Test, model):
    #Predicting on X_Test
    Y_Pred=model.predict(X_Test)
    return Y_Pred,


@app.cell
def __(Y_Pred, data_test, pd):
    #Creating dataframe to export in Csv
    Y_Pred_Df=pd.DataFrame(Y_Pred,columns=['SalePrice'])
    Y_Pred_Df.insert(0,'Id',data_test['Id'])
    return Y_Pred_Df,


@app.cell
def __(Y_Pred_Df):
    print(Y_Pred_Df)
    return


@app.cell
def __(Y_Pred_Df):
    Y_Pred_Df.to_csv("Benchmark1.csv",index=False)
    return


@app.cell
def __():
    #Score in Kaggle- 0.21,765
    return


if __name__ == "__main__":
    app.run()
