# Linear Regression with Gradient Descent

## Overview

This project implements simple linear regression and gradient descent directly in Python using numerical and data-processing tools rather than a prebuilt machine-learning estimator.

The dataset contains 100 observations with two variables:

- **House living area (`x`)** — measured in thousands of square feet
- **House price (`y`)** — measured in units of $10,000

The regression model, loss function, and gradient descent parameter updates were implemented manually. The goal was to develop a deeper understanding of how linear regression works internally and how different learning rates affect the optimization process.

## Model

The project uses a simple linear regression model:

ŷ = θ₀ + θ₁x

where:

- `θ₀` is the intercept
- `θ₁` is the coefficient for house size
- `x` is the house living area
- `ŷ` is the predicted house price

The model parameters are learned using gradient descent.

## Loss Function

Model performance during training is measured using the squared-error loss function:

J(θ) = (1 / 2m) Σ(yᵢ - hθ(xᵢ))²

where `m` is the number of observations.

To verify the loss-function implementation, the function was evaluated using known parameter values:

- `θ = (0, 0)` → approximately `1806.551`
- `θ = (-1, 20)` → approximately `330.099`

Matching these expected values provided a check that the loss calculation was working correctly before running gradient descent.

## Gradient Descent

Gradient descent was implemented manually to update both model parameters over multiple iterations.

For each iteration, the program:

1. Generates predictions using the current parameters.
2. Calculates the prediction error.
3. Computes the parameter updates for `θ₀` and `θ₁`.
4. Updates both parameters simultaneously.
5. Calculates and records the new loss.

Each trial begins with:

θ = (0, 0)

and runs for **15 iterations**.

## Learning Rate Experiments

To examine how the learning rate affects gradient descent, the model was trained independently using four learning rates:

- `η = 0.01`
- `η = 0.1`
- `η = 0.2`
- `η = 0.4`

All four trials are executed in a single run by looping over the learning rates.

A separate loss curve is generated for each trial, showing how the loss changes over the 15 training iterations.

This makes it possible to compare the behavior of gradient descent under different learning-rate settings.

## Predictions

After each gradient descent trial, the program reports:

- The final learned parameter vector
- The final loss value
- The predicted value of `y` for `x = 3.5`
- The predicted value of `y` for `x = 7.0`

Since `x` is measured in thousands of square feet, these inputs correspond to:

- A **3,500-square-foot house**
- A **7,000-square-foot house**

The predictions demonstrate how the learned regression equation can be applied to new input values.

## Visualizations

### House Size vs. Price

The original dataset is displayed in a two-dimensional scatter plot.

- Horizontal axis: house living area
- Vertical axis: house price
- Each observation is displayed using a cross (`x`) marker

This provides a visual representation of the relationship between the input and output variables.

### Gradient Descent Loss Curves

The project also generates four separate loss plots, one for each learning rate.

Each graph shows:

- Iteration number on the horizontal axis
- Loss value on the vertical axis
- The learning rate used in the plot title

These plots provide a visual comparison of how the learning rate influences the optimization process.

## Skills Demonstrated

This project demonstrates experience with:

- Python
- Numerical computation
- Tabular data handling
- Data visualization
- Simple linear regression
- Gradient descent
- Loss functions
- Parameter optimization
- Predictive modeling
- Hyperparameter experimentation
- Machine-learning fundamentals

## Key Takeaways

This project provided hands-on experience with the mathematical and computational steps behind linear regression.

Rather than relying on a prebuilt regression estimator to handle model training, the prediction function, loss calculation, and gradient descent updates were implemented directly.

This made it possible to observe how the model parameters change during training and how the choice of learning rate affects the behavior of the optimization process.

The project also reinforced the importance of validating individual components, such as the loss function, before using them as part of a larger machine-learning workflow.

## Project Context

This project was completed as part of **CSC 3730 – Machine Learning and Data Analytics** at Louisiana State University.

## AI Use Disclosure

The project code was written by me. I used AI occasionally as a reference for questions about programming syntax and for help understanding or troubleshooting errors.

This README was written with the assistance of AI based on my description of the project, its implementation, and its requirements.