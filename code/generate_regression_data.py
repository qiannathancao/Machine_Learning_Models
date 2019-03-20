import numpy as np 

def generate_regression_data(degree, N, noise=1.0):
    """
    Generate two numpy arrays that have a roughly polynomial relationship 
    between them of degree 'degree'. The two arrays should have size
    'N'. The values in x should range between -1 and 1. The coefficients of the
    polynomial are chosen at random.

    There is a noise parameter that is added to the regression. Reminder, you can add
    random noise to the output of a function by simply doing:
        f'(x) = f(x) + N(0, noise)
    where N(mu, std) is a draw from a normal distribution with mean 0 and standard deviation 1.

    In code, that looks like:
        y += np.random.normal(loc=0.0, scale=noise, size=y.shape)

    As noise increases, the regression will become harder and harder to fit.

    Args:
        degree (int): degree of polynomial that relates the output x and y
        N (int): number of points to generate
        noise (float): standard devation of random noise to add to the relationship 
            between x and y.
    Returns:
        x (np.ndarray): explanatory variable of size N, ranges between -1 and 1.
        y (np.ndarray): response variable of size N.
            responds to x as a polynomial of degree. 

    """
    raise NotImplementedError()