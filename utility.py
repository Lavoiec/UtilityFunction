import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import code

class Utility:
    """
    Main utility class.
    Contains the information necessary for the two-good case.
    """

    def __init__(self):
        """
        px = Price of X
        py = Price of Y
        X = Data
        income = Income
        Utility = Utility
        """
        self.px = 0
        self.py = 0
        self.X = []
        self.income = 0
        self.utility = 0

    def maximize_utility(self, X, utility_function, income_function):
        """
        Uses scipy.optimize.minimize to optimize the utility function subject to the income constraint
        The Utility function is multiplied by negative one, letting us use the minimimize
        function as a maximizer
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Returns a OptimizeResult object
        """
        constraint={'type':'ineq', 'fun':income_function}
        solver = minimize(utility_function, X,constraints=constraint)
        return solver

    def create_budget_constraint(self):
        """
        Creates a Numpy Array representing a budget constraint.
        Useful for plotting.

        Returns a Numpy array
        """
        income=self.income
        px=self.px
        py=self.py
        prices = []
        x_max = int(income/px,)
        y_max = int(income/py)
        income = int(income)

        budget_line = lambda x: (income - px*x)/py
        for x_i in np.arange(0,y_max+1, 0.01):
            prices.append([x_i,budget_line(x_i)])


        return np.array(prices)

    def create_indifference_curve(self, max_goods=100):
        """
        Uses data in the Utility Object to create
        a NumPy array representing an Indifference Curve.

        Needs to be run after having run a Utility Function method.
        Otherwise, utility will be 0, and there will be no explicit_utility function

        Returns an NumPy Array.
        """

        bundles = []
        if self.max_utility:
            utility = self.max_utility
        else:
            utility = -1*int(self.utility)

        if self.utility_func:
            utility_function = self.explicit_utility
        else:
            raise KeyError("Need to have a utility function. Try running a utility function method")

        for x_i in np.arange(1, max_goods, 0.01):
                bundles.append([x_i,utility_function(x_i)]) 
        return np.array(bundles)


    def cobb_douglas(self, alpha=0.5, beta=0.5, income=100, px=1, py=1):
        """
        Python Representation of the Cobb-Douglas Demand Function
        For more info, check WikiPedia
        https://en.wikipedia.org/wiki/Cobb%E2%80%93Douglas_production_function

        Cobb-Douglas function looks like: U(x,y) = (x**alpha)*(y**beta)
        alpha: Exogenous parameter linked to the good x
        beta: Exogenous parameter linked to the good y
        income: ....income
        px: Price of X
        py: Price of Y
        """
        self.X = [1,1]
        self.px = px
        self.py = py
        self.income = income

        self.alpha = alpha
        self.beta = beta
        
        def utility_function(X, alpha=alpha, beta=beta):
            """
            Returns the utility value for the Cobb-Douglas function.
            It returns negative to be used in the maximize_utility() function
            which uses a minimize function.

            Returns a float.
            """
            x = X[0]
            y = X[1]
            return -1*(x**alpha)*(y**beta)

        def income_constraint(X):
            """
            Returns a float to be used as a constraint in the
            Cobb-Douglas function.
            """
            x = X[0]
            y = X[1]
            return self.income - self.px*x - self.py*y

        self.optimize_result = self.maximize_utility(X = self.X, utility_function = utility_function, income_function=income_constraint)
        self.optimal_x = np.round(self.optimize_result.x[0], 0)
        self.optimal_y = np.round(self.optimize_result.x[1],0)
        self.max_utility = -1*(np.round(self.optimize_result.fun, 0))
        self.utility_func = utility_function
        self.explicit_utility = lambda x: (self.max_utility/(x**alpha))**(1/beta)


    def quasi_linear():
        pass


def plot_utility_max(budget_data, indifference_curve_data):
    """
    Plots the Budget Constraint and Indifference Curve, approximating the
    graphical solution to a 2-good utility maximization problem.

    Generates a matplotlib.pyplot plot.
    """
    plt.plot(budget_data[:,0], budget_data[:,1], label='Budget Constraint')
    plt.plot(indifference_curve_data[:,0], indifference_curve_data[:,1], label='Indifference Curve')
    plt.ylim((0,(1.1)*budget_data[0,1]))
    plt.xlim((0,(1.1)*budget_data[0,1]))
    plt.legend()
    plt.title('Indifference Curve')



if __name__ == "__main__":
    a = Utility()
    a.cobb_douglas(alpha=0.7, beta=0.65, income=700, px=14, py=14)
    f = a.create_budget_constraint()
    g = a.create_indifference_curve()
    print("x = {}!".format(a.optimal_x))
    print("y = {}!".format(a.optimal_y))
    plot_utility_max(f,g)
    plt.show()
