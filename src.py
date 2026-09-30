import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import gradient_descent_update_and_loss_function_value_tracker, hypothesis_value, loss_function, df

#Assigns reference variables to column 1 and 2
x = df['area (thousands of square feet)']
y = df['price (in units of 10k)']

#Creates scatter plot of data from above table
plt.figure()
plt.scatter(x,y, marker='x')
plt.xlabel("Area (thousands of square feet)")
plt.ylabel("Price (in units of 10k)")
plt.title("House Price vs Square Feet")

'''Prints the loss function value after the input of 
following ordered pairs (respectively): (0,0), (0,1).'''
print(f"J(0,0): {loss_function(0,0)}")
print(f"J(-1,20): {loss_function(-1,20)}")
print()

'''
 #1.Plots loss function value versus current iteration of gradient descent using 
#some learning rate and starting parameter vector
#2. Prints parameter vector after n iterations of gradient descent using some learning rate
#3. Prints prediction values for x = 3.5 and x = 7, given previous parameter vector 
'''
learning_rates = [0.01, 0.1, 0.2, 0.4]

for i in range(len(learning_rates)):
    x, y, optimal_theta_0, optimal_theta_1 = gradient_descent_update_and_loss_function_value_tracker([0,0], 15, learning_rates[i])
    print(f'Final parameter vector for learning rate:{learning_rates[i]} = {float(optimal_theta_0), float(optimal_theta_1)}')
    print(f'Predicited value for x = 3.5 is {hypothesis_value(optimal_theta_0, optimal_theta_1, 3.5)} \nPredicted value for x = 7 is {hypothesis_value(optimal_theta_0, optimal_theta_1, 7)}')
    print()
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Loss Value")
    plt.ylabel("Iteration")
    plt.title(f"Learning Rate: {learning_rates[i]}")


plt.show()


