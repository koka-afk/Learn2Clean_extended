#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille <laure.berti@ird.fr>

import warnings
import time
import numpy as np
import pandas as pd


class Feature_selector():
    """
    Select the features for the train dataset using a
    particular strategy and keep the same features in the test dataset

    Parameters
    ----------
    * dataset: input dataset dict
        including dataset['train'] pandas DataFrame, dataset['test']
        pandas DataFrame and dataset['target'] pandas DataSeries
        obtained from train_test_split function of Reader class

    * strategy: str, default = 'LC'
        The choice for the feature selection strategy:
            - 'MR', 'VAR and 'LC' are agnostic to the task
            - 'Tree', 'WR', 'SVC' are used for classification task
            -  'L1', 'IMP' are used  for regression task
            Available strategies=
            'MR': using a default threshold on the missing ratio per variable,
            i.e., variables with 20% (by default) and more missing values
            are removed
            'LC': detects pairs of linearly correlated variables and remove one
            'VAR': uses threshold on the variance
            'Tree': uses decision tree classification as model for feature
                selection given the target set for classification task
                'SVC': uses linear SVC as model for feature selection given
                 the target set for classification task
            'WR': uses the selectKbest (k=10) and Chi2 for feature selection
                given the target set for classification task
            'L1': uses Lasso L1 for feature selection given the target set for
                regression task
            'IMP': uses Random Forest regression for feature selection given
                the target set for regression task

    * exclude: str, default = 'None' name of variable to be excluded from
        feature selection

    * threshold: float, default = '0.3' only for MR, VAR, LC, L1, and IMP

    * verbose: Boolean,  default = 'False' otherwise display information
    about the applied feature selection
    """

    def __init__(self, dataset, time_col=None, event_col=None,  strategy='LC', exclude=None,
                 threshold=0.3, mode="original", config=None, verbose=False): # mode determines original function or survival analysis 

        self.dataset = dataset

        self.time_column = time_col

        self.event_column = event_col

        self.mode = mode

        self.config = config

        self.strategy = strategy

        self.exclude = exclude

        self.threshold = threshold

        self.verbose = verbose

    def get_params(self, deep=True):

        return {'strategy': self.strategy,

                'exclude': self.exclude,

                'threshold':  self.threshold,

                'verbose': self.verbose

                }

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`feature_selector.get_params().keys()`")

            else:

                setattr(self, k, v)

    # Feature selection based on missing values
    def FS_MR_missing_ratio(self, dataset, missing_threshold=.2):

        print("Apply MR feature selection with missing "
              "threshold=", missing_threshold)

        # Find the features with a fraction of missing
        # values above `missing_threshold`
        # Calculate the fraction of missing in each column
        missing_series = dataset.isnull().sum() / dataset.shape[0]

        missing_stats = pd.DataFrame(missing_series).rename(
            columns={'index': 'feature', 0: 'missing_fraction'})

        # Sort with highest number of missing values on top
        missing_stats = missing_stats.sort_values(
            'missing_fraction', ascending=False)

        # Find the columns with a missing percentage above the threshold
        record_missing = pd.DataFrame(missing_series[missing_series >
                                      missing_threshold]).reset_index().\
            rename(columns={'index': 'feature', 0: 'missing_fraction'})

        to_drop = list(record_missing['feature'])

        if self.verbose:

            print(missing_stats)

        print('%d features with greater than %0.2f missing values.\n' %
              (len(to_drop), missing_threshold))

        print('List of variables to be removed :', to_drop)

        to_keep = set(dataset.columns) - set(to_drop)

        if self.verbose:

            print("List of variables to be keep")

            print(list(to_keep))

        return dataset[list(to_keep)]

    def FS_LC_identify_collinear(self, dataset, correlation_threshold=0.8):

        # Finds linear-based correlation between features (LC)
        # For each pair of features with a correlation
        # coefficient greather than `correlation_threshold`,
        # only one of the pair is identified for removal.
        # one attribute can be kept (excluded from feature selection)
        # Using code adapted from: https://gist.github.com/
        # Swarchal/e29a3a1113403710b6850590641f046c
        print("Apply LC feature selection with threshold=",
              correlation_threshold)

        # Calculate the correlations between every column
        corr_matrix = dataset.corr()

        if self.verbose:

            print("Correlation matrix")

            print(corr_matrix)

        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(
            upper[column].abs() > correlation_threshold) and 
            column != self.event_column and 
            column != self.time_column] # ensuring that the event and time columns cannot be dropped
        
        print(f'TO_DROP LIST: ------------------------------> {to_drop}')

        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(
            columns=['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop
        for column in to_drop:

            # Find the correlated features
            corr_features = list(
                upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(
                upper[column][upper[column].abs() > correlation_threshold])

            drop_features = [column for _ in range(len(corr_features))]

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})

            # Add to dataframe
            record_collinear = pd.concat([record_collinear, temp_df], ignore_index=True)

        print('%d features with linear correlation greater than %0.2f.\n' %
              (len(to_drop), correlation_threshold))

        print('List of correlated variables to be removed :', to_drop)

        to_keep = set(dataset.columns) - set(to_drop)

        if self.verbose:

            print("List of numerical variables to be keep")

            print(list(to_keep))

        return dataset[list(to_keep)]

    def FS_WR_identify_best_subset(self, df_train, df_target, k=10):
        # k number of k best feature to keep
        # Feature extraction
        # requires non missing values
        # requires a categorical target variable

        from sklearn.feature_selection import SelectKBest

        from sklearn.feature_selection import chi2, f_regression

        from sklearn.preprocessing import LabelEncoder

        print("Apply WR feature selection")

        if df_train.isnull().sum().sum() > 0:

            df_train = df_train.dropna()

            print('WR requires no missing values, so '
                  'missing values have been removed applying '
                  'DROP on the train dataset.')

        X = df_train.select_dtypes(['number'])

        Y = df_target

        if len(df_train.columns) < 1 or len(df_train) < 1:

            print(
                'Error: Need at least one continous variable '
                'for identifying the best subset of features')

            df = df_train

        else:

            selector = SelectKBest(score_func=f_regression, k='all') # score_func changed from chi2 to f_regression

            lsv = list(X.lt(0).sum().values)

            lis = list(X.lt(0).sum().index)

            for i in range(0, len(lsv)-1):

                if lsv[i] > 0:

                    del lis[i]

            if len(lis) == 0:

                print("Input dataset has no positive variables. "
                      "WR feature selection is not applicable. "
                      "Dataset unchanged.")

                df = df_train

            else:

                X = X[lis]

                print("Input variables must be non-negative. "
                      "WR feature selection is only applied to "
                      "positive variables.")
                # print(X.iloc[:,0])
                # print('\n')
                if self.event_column != None and self.time_column != None:
                    new_data = X # .copy() # .dropna() # dropping NAN Values and working on data copy for now
                    pre_event_labels = LabelEncoder().fit_transform(new_data[self.event_column].astype(bool))
                    time_values = new_data[self.time_column]
                    Y = np.array(list(zip(pre_event_labels, time_values)), dtype=[('event', '?'), ('time', '<f8')])
                    print(f'Y AFTER REASSIGNMENT: {Y} \n')
                    event_labels = LabelEncoder().fit_transform(Y['event'])
                    print(f'X SHAPE:  {X}')
                    print(f'Y SHAPE:  {Y.shape}')
                    selector.fit(new_data, event_labels) # Using event_labels instead of the original Y
                else:
                    selector.fit(X, Y)
                # TODO: ensure that if time and target are not "None"(doing survival analysis) then they will not be removed
                Best_Flist = X.columns[selector.get_support(
                    indices=True)].tolist()

                df = X[Best_Flist]

                if self.verbose:

                    print("Best features to keep", Best_Flist)

        return df

    def FS_SVC_based(self, df_train, df_target):
        # Feature extraction
        # requires non missing value

        from sklearn.svm import LinearSVC

        from sklearn.feature_selection import SelectFromModel

        print("Apply SVC feature selection")

        if df_train.isnull().sum().sum() > 0:

            df_train = df_train.dropna()

            print('SVC requires no missing values, '
                  'so missing values have been removed applying '
                  'DROP on the train dataset.')

        if len(df_train.columns) < 1 or len(df_train) < 1:

            print(
                  'Error: Need at least one continous variable '
                  'for feature selection \n Dataset inchanged')

            df = df_train

        else:

            X = df_train.select_dtypes(['number'])

            Y = df_target

            lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)

            model = SelectFromModel(lsvc, prefit=True)

            Best_Flist = X.columns[model.get_support(indices=True)].tolist()

            df = X[Best_Flist]

            if self.verbose:

                print("Best features to keep", Best_Flist)

        return df

    def FS_Tree_based(self, df_train, df_target):
        # Feature extraction using the decision tree classification as model
        # requires non missing value

        from sklearn.ensemble import ExtraTreesClassifier

        from sklearn.preprocessing import LabelEncoder

        from sklearn.feature_selection import SelectFromModel

        print("Apply Tree-based feature selection ")

        if df_train.isnull().sum().sum() > 0:

            df_train = df_train.dropna()

            print('Tree requires no missing values, so missing '
                  'values have been removed applying DROP '
                  'on the train dataset.')

        if len(df_train.columns) < 1 or len(df_train) < 1:

            print('Error: Need at least one continous variable '
                  'for feature selection \n Dataset inchanged')

            df = df_train

        else:

            X = df_train.select_dtypes(['number'])

            Y = df_target

            clf = ExtraTreesClassifier(n_estimators=50)
            print(f'HERE BEFORE EXTRA TREES ______')
            print(f'\n \n \n{Y}\n\n\n')
            lab = LabelEncoder()
            Y_transformed = lab.fit_transform(Y[self.time_column]) # Y needs to have be discrete values 
            print(f'{Y_transformed}\n\n\n')

            clf = clf.fit(X, Y_transformed)

            model = SelectFromModel(clf, prefit=True)

            Best_Flist = X.columns[model.get_support(indices=True)].tolist()

            if self.event_column not in Best_Flist: # avoid dropping event column
                Best_Flist.append(self.event_column)

            if self.time_column not in Best_Flist:
                Best_Flist.append(self.time_column) # avoid dropping time column

            if self.verbose:

                print("Best features to keep", Best_Flist)

            df = X[Best_Flist]

        return df
    

    def univariate_coxph_selection(self):
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        import numpy as np
        import pandas as pd

        print(">>Feature Selection started with UC method..... ")

        # Drop rows with NaN values
        x = self.dataset.dropna()

        # Ensure the time and event columns are of correct data types
        x[self.time_column] = x[self.time_column].astype(float)
        x[self.event_column] = x[self.event_column].astype(bool)

        print(x)

        # Encode event column
        event_labels = LabelEncoder().fit_transform(x[self.event_column])
        y = np.array(list(zip(event_labels, x[self.time_column])), dtype=[('event', '?'), ('time', '<f8')])

        # Standardize feature columns
        feature_columns = x.columns.difference([self.event_column, self.time_column])
        x[feature_columns] = StandardScaler().fit_transform(x[feature_columns])

        # Perform feature selection using Univariate Cox Proportional Hazards Model
        cph = CoxPHSurvivalAnalysis(alpha=0.1, verbose=10)
        print(y)

        try:
            cph.fit(x[feature_columns], y)
        except ValueError as e:
            print("Error during fitting:", e)
            return None

        # Get the magnitudes of estimated coefficients as feature importance scores
        coefs = cph.coef_
        feature_importances = np.abs(coefs)

        # Select the top features based on importance scores
        k = int(len(feature_importances) * 0.8)  # Select top 80% features
        top_features_indices = np.argsort(feature_importances)[-k:]

        selected_features = np.zeros(len(self.dataset.columns), dtype=bool)

        # Ensure that event and time columns are not removed
        event_column_index = x.columns.get_loc(self.event_column)
        time_column_index = x.columns.get_loc(self.time_column)

        if event_column_index not in top_features_indices:
            top_features_indices = np.append(top_features_indices, event_column_index)

        if time_column_index not in top_features_indices:
            top_features_indices = np.append(top_features_indices, time_column_index)

        # Ensure all indices are within bounds
        top_features_indices = [idx for idx in top_features_indices if idx < len(selected_features)]

        selected_features[top_features_indices] = True

        selected_columns = []

        # Print the selected features
        print("UC Selected Features:")
        for i, feature in enumerate(x.columns):
            if selected_features[i]:
                selected_columns.append(feature)
                if feature != self.event_column and feature != self.time_column:
                    print(feature)

        new_dataset = x[selected_columns]
        return new_dataset

    

    def lasso_selection(self):

        from sklearn.preprocessing import LabelEncoder
        from sklearn.linear_model import LassoCV

        print(">>Feature Selection started with lasso method..... ")

        x = self.dataset.dropna() #TODO I have to include dropna() as NaNs are not allowed and lasso can be a starting state

        # Encode the event column in the target array
        event_labels1 = LabelEncoder().fit_transform(x[self.event_column].astype(bool))
        y = np.array(list(zip(event_labels1, x[self.time_column])), dtype=[('event', '?'), ('time', '<f8')])
        event_labels = LabelEncoder().fit_transform(y["event"])

        # Fit LassoCV with encoded event column and time column
        # warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lasso = LassoCV(cv=5)
        lasso.fit(x, event_labels)
        # Return the selected features based on non-zero coefficient values
        selected_features = lasso.coef_ != 0

        event_column_index = x.columns.get_loc(self.event_column)
        time_column_index = x.columns.get_loc(self.time_column)
        selected_features[event_column_index] = True # Ensuring that event column excluded
        selected_features[time_column_index] = True # Ensuring that time column excluded

        print("LASSO Selected Features:")
        
        selected_columns = []

        for feature, selected in zip(x, selected_features):
            if selected:
                selected_columns.append(feature)
                print(feature)
        
        return x[selected_columns]
    

    def rfe_selection(self):

        from sklearn.preprocessing import LabelEncoder
        from sklearn.feature_selection import RFECV
        from sklearn.linear_model import LassoCV

        print(">>Feature Selection started with RFE method..... ")
        
        x = self.dataset.dropna() #TODO I have to include dropna() as NaNs are not allowed and rfe can be a starting state

        event_labels = LabelEncoder().fit_transform(x[self.event_column].astype(bool))
        
        y = np.array(list(zip(event_labels, x[self.time_column])), dtype=[('event', '?'), ('time', '<f8')])

        # Encode the event column in the target array
        Endcoded_event_labels = LabelEncoder().fit_transform(y["event"])

        # Perform feature selection using Recursive Feature Elimination (RFE)
        #warnings.filterwarnings("ignore", category=ConvergenceWarning)
        rfecv = RFECV(estimator=LassoCV(cv=5), step=1, scoring='neg_mean_squared_error')
        rfecv.fit(x, Endcoded_event_labels)

        selected_features = rfecv.support_

        event_column_index = x.columns.get_loc(self.event_column)
        time_column_index = x.columns.get_loc(self.time_column)
        selected_features[event_column_index] = True # Ensuring that event column excluded
        selected_features[time_column_index] = True # Ensuring that time column excluded

        print("RFE Selected Features:")

        selected_columns = []

        for feature, selected in zip(x, selected_features):
            if selected:
                selected_columns.append(feature)
                print(feature)
    
        return x[selected_columns]
    

    def information_gain_selection(self):

        from sklearn.preprocessing import LabelEncoder
        from sklearn.feature_selection import SelectKBest, f_regression

        print(">>Feature Selection started with information gain method..... ")

        x = self.dataset.dropna()#TODO I have to include dropna() as NaNs are not allowed and IG can be a starting state

        event_labels1 = LabelEncoder().fit_transform(x[self.event_column].astype(bool))
       
        time_values = x[self.time_column]

        y = np.array(list(zip(event_labels1, time_values)), dtype=[('event', '?'), ('time', '<f8')])

        # Encode the event column in the target array
        event_labels = LabelEncoder().fit_transform(y["event"])

        # Perform feature selection using Information Gain
        #warnings.filterwarnings("ignore", category=ConvergenceWarning)

        k_best = SelectKBest(score_func=f_regression, k='all')  # changing from k = 'all' to k = 3
        k_best.fit(x, event_labels)

        selected_features = k_best.get_support()

        event_column_index = x.columns.get_loc(self.event_column)
        time_column_index = x.columns.get_loc(self.time_column)
        selected_features[event_column_index] = True # Ensuring that event column excluded
        selected_features[time_column_index] = True # Ensuring that time column excluded


        print("Information Gain Selected Features:")

        selected_columns = []

        for feature, selected in zip(x, selected_features):
            if selected:
                selected_columns.append(feature)
                print(feature)

        return x[selected_columns]
        

    def transform(self):

        start_time = time.time()

        print()

        print(">>Feature selection ")

        if self.mode == "survival":
            if self.strategy == 'UC':
                selected = self.univariate_coxph_selection()
            
            elif self.strategy == 'LASSO':
                selected = self.lasso_selection()
            
            elif self.strategy == 'RFE':
                selected = self.rfe_selection()
            
            elif self.strategy == 'IG':
                selected = self.information_gain_selection()
            
            else:
                raise ValueError("Invalid feature selection strategy. Please choose from 'UC', 'LASSO', 'RFE', or 'IG'.")
            
            print("Feature selection done -- CPU time: %s seconds" %
                (time.time() - start_time))
            
            return selected
            
        else:

            df = self.dataset['train'].copy()

            fsd = self.dataset

            to_keep = []

            print("Before feature selection:")

            print(self.dataset['train'].shape[1], "features ")

            if (self.strategy == "MR"):

                dn = self.FS_MR_missing_ratio(
                    df, missing_threshold=self.threshold)

            elif (self.strategy == "LC"):

                d = df.select_dtypes(['number'])

                do = df.select_dtypes(exclude=['number'])

                dn = self.FS_LC_identify_collinear(
                    d, correlation_threshold=self.threshold)

                dn = dn.join(do)

            elif (self.strategy == 'VAR'):

                dn = df.select_dtypes(['number'])

                coef = dn.std()

                print("Apply VAR feature selection with "
                    "threshold=", self.threshold)

                abstract_threshold = np.percentile(coef, 100. * self.threshold)

                to_discard = coef[coef < abstract_threshold].index

                dn.drop(to_discard, axis=1)

            elif (not isinstance(self.dataset['target'], dict)):

                dn = df.select_dtypes(['number'])

                if dn.isnull().sum().sum() > 0:

                    dn = dn.dropna()

                    dt = self.dataset['target'].loc[dn.index]

                    print('Warning: This strategy requires no missing values,'
                        ' so missing values have been removed applying '
                        'DROP on the dataset.')
                else:

                    dt = self.dataset['target'].loc[dn.index]

                if (self.strategy == 'L1'):

                    from sklearn.linear_model import Lasso

                    print("Apply L1 feature selection with threshold=",
                        self.threshold)

                    model = Lasso(alpha=100.0, tol=0.01, random_state=0)

                    model.fit(dn, dt)

                    coef = np.abs(model.coef_)

                    abstract_threshold = np.percentile(coef, 100. * self.threshold)

                    to_discard = dn.columns[coef < abstract_threshold]

                    dn = dn.drop(to_discard, axis=1)

                elif (self.strategy == 'IMP'):

                    print("Apply IMP feature selection with"
                        " threshold=", self.threshold)

                    from sklearn.ensemble import RandomForestRegressor

                    model = RandomForestRegressor(n_estimators=50,
                                                n_jobs=-1,
                                                random_state=0)

                    model.fit(dn, dt)

                    coef = model.feature_importances_

                    abstract_threshold = np.percentile(coef, 100. * self.threshold)

                    to_discard = dn.columns[coef < abstract_threshold]

                    dn = dn.drop(to_discard, axis=1)

                elif (self.strategy == "Tree"):

                    dn = self.FS_Tree_based(dn, dt)

                elif (self.strategy == "WR"):

                    dn = self.FS_WR_identify_best_subset(dn, dt)

                elif (self.strategy == "SVC"):

                    dn = self.FS_SVC_based(dn, dt)

                else:

                    print("Strategy invalid. Please choose between "
                        "'Tree', 'WR', 'SVC', 'VAR', 'LC' or 'MR'"
                        " -- No feature selection done on the train dataset")

                    dn = self.dataset['train'].copy()

            else:

                print("Strategy invalid. Please choose between "
                    "'VAR', 'LC', 'MR' if you have no target, or 'Tree', "
                    "'WR', 'SVC'"
                    " or 'L1', 'IMP' if you have a target"
                    "-- No feature selection done on the train dataset")

                dn = self.dataset['train'].copy()

            to_keep = [column for column in dn.columns]

            if (self.exclude is None):

                fsd['train'] = dn[to_keep]

            elif (self.exclude not in self.dataset['train'].columns.values):

                print(
                    "Exclude variable invalid. Please choose a variable"
                    "from the input training dataset.")

            elif (self.exclude in dn.columns.values):

                fsd['train'] = dn[to_keep]

            else:

                print("and keep variable", self.exclude)

                to_keep.append(self.exclude)

                dn = self.dataset['train']

                fsd['train'] = dn[to_keep]

            if (not isinstance(self.dataset['test'], dict)):
                # if not self.dataset['test'].empty:

                df_test = pd.DataFrame.from_dict(self.dataset['test'])

                to_keep = list(set(to_keep) & set(df_test.columns))

                fsd['test'] = df_test[to_keep]

            print("After feature selection:")

            print(len(to_keep), "features remain")

            print(to_keep)

            print("Feature selection done -- CPU time: %s seconds" %
                (time.time() - start_time))

            print()

            return fsd
