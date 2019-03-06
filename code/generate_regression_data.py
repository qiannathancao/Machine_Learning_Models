import numpy as np 

def generate_regression_data(degree, N):
    """
    Generate two numpy arrays that have a polynomial relationship 
    between them of degree 'degree'. The two arrays should have size
    'N'. The values in x should range between -1 and 1.

    Args:
        degree: degree of polynomial that relates the output x and y
        N: number of points to generate
    Returns:
        x (np.ndarray): explanatory variable of size N, ranges between -1 and 1.
        y (np.ndarray): response variable of size N.
            responds to x as a polynomial of degree. 

    """
    raise NotImplementedError()