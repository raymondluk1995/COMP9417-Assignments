import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

# prediction for instance x
def h(x,theta_0,theta_1):
    return (theta_0+theta_1*x)

# function of Stochastic Gradient Descent
def sgd(train_x,train_y,theta_0=-1,theta_1=-0.5,alpha=0.01):
    J_values = []
    for j in range(50):
        J_value = 0
        for i in range(train_x.shape[0]):
            theta_0 = theta_0 + alpha*(train_y[i]-h(train_x[i],theta_0,theta_1))
            theta_1 = theta_1 + alpha*\
            (train_y[i]-h(train_x[i],theta_0,theta_1))*train_x[i]
            J_value += (train_y[i]-h(train_x[i],theta_0,theta_1))**2
        J_values.append(J_value/train_x.shape[0])
    return theta_0,theta_1,J_values

# Normalize feature data and creat training sets and test sets
def pre_process(filename):
    # Read data file
    df = pd.read_csv(filename)
    min_list =[]
    max_list=[]
    for i in range(1,len(df.columns)-1):
        min_list.append(min(df[df.columns[i]]))
        max_list.append(max(df[df.columns[i]]))
        df[df.columns[i]] = (df[df.columns[i]]-min_list[i-1])/(max_list[i-1]-min_list[i-1])
    # Data Split
    data_x = df.iloc[:,:-1]
    data_y = df.iloc[:,-1]
    # 300 rows for training and 100 rows for testing
    train_x,test_x,train_y,test_y = train_test_split(data_x,data_y,\
                                                     test_size=0.25,shuffle=False)
    return (df,train_x,test_x,train_y,test_y)

# Build a linear regression model for given feature name
def feature_regression(train_x,test_x,train_y,test_y,feature_name):
    train_x = train_x[feature_name]
    test_x = test_x[feature_name]
    theta_0,theta_1,J_values = sgd(train_x,train_y)
    RMSE_train =0
    RMSE_test = 0
    for i in range(train_y.shape[0]):
        RMSE_train += (train_y[i]-h(train_x[i],theta_0,theta_1))**2
    for i in range(train_y.shape[0],train_y.shape[0]+test_y.shape[0]):
        RMSE_test += (test_y[i]-h(test_x[i],theta_0,theta_1))**2
    RMSE_train = np.sqrt(RMSE_train/train_y.shape[0])
    RMSE_test = np.sqrt(RMSE_test/test_y.shape[0])
    print('In the regression model for {0}, the theta 0 is {1},\
and the theta_1 is {2}'.format(feature_name,theta_0,theta_1))
    print('RMSE for {0} training set is {1:3},\
and RMSE for house age test set is {2:3}'.\
          format(feature_name,RMSE_train,RMSE_test))
    return (J_values)

# Plot all regression model's cost functions
def plot_all(data):
    if(len(data)==0):
          print('Input data for plotting is empty, exit now.')
          sys.exit()
    x_axis = np.arange(len(data[0][0]))
    plt.figure()
    plt.style.use('seaborn')
    for i in range(len(data)):
          plt.plot(x_axis,data[i][0],label=data[i][1])
    plt.legend()
    plt.xlabel('iteration time')
    plt.ylabel('cost function')
    plt.title('Cost Function for x features')
    plt.savefig('plot.png')
    plt.show()
          
def main():
    df,train_x,test_x,train_y,test_y = pre_process("house_prices.csv")
    plot_data = []
    for feature_name in df.columns[1:-1]:
          J_values = feature_regression(train_x,test_x,train_y,test_y,feature_name)
          plot_data.append([J_values,feature_name])
    plot_all(plot_data)

if(__name__=="__main__"):
    main()
    

