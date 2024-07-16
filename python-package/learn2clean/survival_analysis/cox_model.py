import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

class CoxRegressor:
    def __init__(self, dataset, target_goal, time_column, config=None, verbose=False):
        self.dataset = dataset
        self.event_column = target_goal
        self.time_column = time_column
        self.config = config
        self.verbose = verbose
        self.model = None


    def updated_fit(self):

        self.model = CoxPHFitter(penalizer=0.1)

        print(f"IN UPDATED FIT AND SIZE OF DATASET IS::::: {len(self.dataset)}")
        
        x = self.dataset
        x[self.event_column] = x[self.event_column].astype(bool)

        y = x[[self.time_column, self.event_column]]

        print("Building Cox proportional-hazards model.....")

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        self.model.fit(x_train, duration_col=self.time_column, event_col=self.event_column)
        c_index = concordance_index(y_test[self.time_column], -self.model.predict_partial_hazard(x_test))

        print(f" Buidling Cox proportional-hazards model is done\n C-Index score: {c_index:.4f}")
        
        return c_index

    
            
"""
Since Cox PH fit function requires a dataset with numerical values only,
we have a couple of options:

1- Encode data away from the original dataset (Do not know how much this affects the process)
2- Encode the entire dataset (Maybe a better option)

----------------------------------------------------

Another thing is feature_selection class is now equipped with couple 
label encodings because some methods do not accept continous values

"""
