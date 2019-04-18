import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement polynomial regression from scratch.
        
        This class takes as input "degree", which is the degree of the polynomial 
        used to fit the data. For example, degree = 2 would fit a polynomial of the 
        form:

            ax^2 + bx + c
        
        Your code will be tested by comparing it with implementations inside sklearn.
        DO NOT USE THESE IMPLEMENTATIONS DIRECTLY IN YOUR CODE. You may find the 
        following documentation useful:

        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        Here are helpful slides:

        http://interactiveaudiolab.github.io/teaching/eecs349stuff/eecs349_linear_regression.pdf
    
        The internal representation of this class is up to you. Read each function
        documentation carefully to make sure the input and output matches so you can
        pass the test cases. However, do not use the functions numpy.polyfit or numpy.polval. 
        You should implement the closed form solution of least squares as detailed in slide 10
        of the lecture slides linked above.

        in test:
            p = PolynomialRegression(degree)
            p.fit(x, y)
            y_hat = p.predict(x)


        Usage:
            import numpy as np
            
            x = np.random.random(100)
            y = np.random.random(100)
            learner = PolynomialRegression(degree = 1)
            learner.fit(x, y) # this should be pretty much a flat line
            predicted = learner.predict(x)

            new_data = np.random.random(100) + 10
            predicted = learner.predict(new_data)
            # confidence compares the given data with the training data
            confidence = learner.confidence(new_data)

        Args:
            degree (int): Degree of polynomial used to fit the data.
        """
        self.degree = degree
        self.best_params = None

    
    def fit(self, features, targets):
        """
        Fit the given data using a polynomial. The degree is given by self.degree,
        which is set in the __init__ function of this class. The goal of this
        function is fit features, a 1D numpy array, to targets, another 1D
        numpy array.
        
        my note:

            def get_best_param(X, y):
                X_transpose = X.T
                best_params = inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
                # normal equation
                # theta_best = (X.T * X)^(-1) * X.T * y
                return best_params # returns a list

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (saves model and training data internally)
        """
        _features = np.ones((len(features),1))
        for i in range(1, self.degree+1):
            _features = np.append(_features,features**i,axis=1)

        self.best_params = np.linalg.inv(_features.transpose().dot(_features)).dot(_features.transpose()).dot(targets)
        # print ("param: ", self.best_params)
        # print ("param shape: ", self.best_params.shape)
        return

    def predict(self, features):
        """
        Given features, a 1D numpy array, use the trained model to predict target 
        estimates. Call this after calling fit.

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        _features = np.ones((len(features),1))
        for i in range(1,self.degree+1):
            _features = np.append(_features,features**i,axis=1)
        _features = _features
        #print("\n#####_features######: ",_features)

        #print("\n#####predict result #####: ", _features.dot(self.best_params))
        return _features.dot(self.best_params)


    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the polynomial fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION. Instead, use plt.savefig().

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (plots to the active figure)
        """
        prediction = self.predict(features)
        test_x = features
        plt.plot(features, targets, "b.")
        plt.plot(test_x, prediction, "r--")
        plt.savefig("polynomial_regression")
