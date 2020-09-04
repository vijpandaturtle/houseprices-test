import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

class Pipeline:
    def __init__(self, target, categorical_to_impute, year_variable, numerical_to_impute, numerical_log, categorical_encode,
                 features, test_size=0.1, random_state=0, percentage=0.01, ref_variable='YrSold'):

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        #engineered params
        self.imputing_dict = {}
        self.frequent_category_dict = {}
        self.encoding_dict = {}

        #models
        self.scaler = MinMaxScaler()
        self.model = Lasso(alpha=0.005, random_state=random_state)

        #groups of variables to engineer
        self.target = target
        self.year_variable = year_variable
        self.categorical_to_impute = categorical_to_impute
        self.numerical_to_impute = numerical_to_impute
        self.numerical_log = numerical_log
        self.categorical_encode = categorical_encode
        self.features = features

        #more parameters
        self.test_size = test_size
        self.random_state = random_state
        self.percentage = percentage
        self.ref_variable = ref_variable

    
