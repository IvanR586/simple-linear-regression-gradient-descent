import pandas as pd


#transfers data from table to dataframe object
df = pd.read_csv("singleFeature.csv")

#returns the predicted value given the parameter vector as a list and the ith feature value. Is based on a 2d linear equation.
def hypothesis_value(theta_0, theta_1, x_i):
    return theta_0 + (theta_1 * x_i)

#Returns the value of some theta after one gradient descent, not for b intercept
def theta_update_not_for_intercept(theta_j, learning_rate):
     theta_j = theta_j
     m = len(df)
     sum = 0
     learning_rate = learning_rate
     for i in range(m):
        y_i = df.iloc[i, 1]
        predicted_y_i = hypothesis_value(theta_j[0], theta_j[1], df.iloc[i, 0])
        sum += (y_i - predicted_y_i) * df.iloc[i, 0]
     return theta_j[1] + (sum * (learning_rate / m))

#Returns the value of some theta after one gradient descent iteration, for b intercept
def theta_update_for_intercept(theta_j, learning_rate):
     theta_j = theta_j
     m = len(df)
     sum = 0
     learning_rate = learning_rate
     for i in range(m):
        y_i = df.iloc[i, 1]
        predicted_y_i = hypothesis_value(theta_j[0], theta_j[1], df.iloc[i, 0])
        sum += (y_i - predicted_y_i) 
     return theta_j[0] + (sum * (learning_rate / m))

#Performs Gradient Descent on parameter vector, and tracks the value of the loss function at each iteration
#Returns loss values for each iteration, each iteration value, and parameter vector
def gradient_descent_update_and_loss_function_value_tracker(theta_j, iterations, learning_rate):
    theta_j = theta_j
    learning_rate = learning_rate
    iterations = iterations
    loss = []
    iteration = []
    for i in range(iterations):
        loss.append(loss_function(theta_j[0], theta_j[1]))
        iteration.append(i+1)
        theta_j[0] = theta_update_for_intercept(theta_j, learning_rate)
        theta_j[1] = theta_update_not_for_intercept(theta_j, learning_rate)
    return loss, iteration, theta_j[0], theta_j[1]

#Returns the loss function value, given a parameter vector as a list.
def loss_function(theta_0, theta_1):
    m = len(df)
    loss = 0
    for i in range(m):
        y_i = df.iloc[i, 1]
        predicted_y_i = hypothesis_value(theta_0, theta_1, df.iloc[i, 0])
        loss += (y_i - predicted_y_i) ** 2
    loss = loss * (1/2) * (1/m)
    return loss
