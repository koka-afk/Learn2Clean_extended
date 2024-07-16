#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille

import warnings
import time
import sys
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


class Imputer():
    """
    Replace or remove the missing values using a particular strategy

    Parameters
    ----------
    * dataset: input dataset dict
        including dataset['train'] pandas DataFrame, dataset['test']
        pandas DataFrame and dataset['target'] pandas DataSeries
        obtained from train_test_split function of Reader class

    * strategy: str, default = 'DROP'
        The choice for the feature selection strategy:
            - 'EM': only for numerical variables; imputation based on
                expectation maximization
            - 'MICE': only for numerical variables  missing at random (MAR);
                Multivariate Imputation by Chained Equations
            - 'KNN', only for numerical variables; k-nearest neighbor
                imputation (k=4) which weights samples using the mean squared
                difference on features for which two rows both have observed
                data
            - 'RAND', 'MF': both for numerical and categorical variables;
                replace missing values by randomly selected value in the
                variable domain or by the most frequent value in the variable
                domain respectively
            - 'MEAN', 'MEDIAN': only for numerical variables; replace missing
                values by mean or median of the numerical variable respectvely
            - or 'DROP' remove the row with at least one missing value

    * verbose: Boolean,  default = 'False' otherwise display about imputation

    * threshold: float, default =  None

    * exclude: str, default = 'None' name of variable to be excluded
        from imputation
    """

    def __init__(self, dataset, strategy='DROP', verbose=False,
                 exclude=None, time_col=None, event_col=None, config=None, mode="original", threshold=None): # mode determines original function or survival analysis

        self.dataset = dataset

        self.strategy = strategy

        self.mode = mode

        self.time_col = time_col

        self.event_col = event_col

        self.config = config

        self.verbose = verbose

        self.threshold = threshold

        self.exclude = exclude  # to implement

    def get_params(self, deep=True):

        return {'strategy': self.strategy,
                'verbose': self.verbose,
                'exclude': self.exclude,
                'threshold': self.threshold}

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`imputer.get_params().keys()`")

            else:

                setattr(self, k, v)

    # Handling Missing values

    def mean_imputation(self, dataset):
        # for numerical data
        # replace missing numerical values by the mean of
        # the corresponding variable

        df = dataset

        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

            X = dataset.select_dtypes(['number'])

            for i in X.columns:

                X[i] = X[i].fillna(int(X[i].mean()))

            Z = dataset.select_dtypes(exclude=['number'])

            df = pd.DataFrame.from_records(
                X, columns=dataset.select_dtypes(['number']).columns)

            df = pd.concat([X, Z], axis=1)

        else:

            pass

        return df

    def median_imputation(self, dataset):
        # only for numerical data
        # replace missing numerical values by the median
        # of the corresponding variable

        df = dataset

        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

            X = dataset.select_dtypes(['number'])

            for i in X.columns:

                X[i] = X[i].fillna(int(X[i].median()))

            Z = dataset.select_dtypes(include=['object'])

            df = pd.DataFrame.from_records(
                X, columns=dataset.select_dtypes(['number']).columns)

            df = df.join(Z)

        else:

            pass

        return df

    def NaN_drop(self, dataset):
        # for both categorical and numerical data
        # drop observations with missing values

        print("Dataset size reduced from", len(
            dataset), "to", len(dataset.dropna()))

        return dataset.dropna()

    def MF_most_frequent_imputation(self, dataset):
        # for both categorical and numerical data
        # replace missing values by the most frequent value
        # of the corresponding variable

        for i in dataset.columns:

            mfv = dataset[i].value_counts().idxmax()

            dataset[i] = dataset[i].replace(np.nan, mfv)

            if self.verbose:

                print("Most frequent value for ", i, "is:", mfv)

        return dataset

    def NaN_random_replace(self, dataset):
        # for both categorical and numerical data
        # replace missing data with a random observation with data

        M = len(dataset.index)

        N = len(dataset.columns)

        ran = pd.DataFrame(np.random.randn(
            M, N), columns=dataset.columns, index=dataset.index)

        dataset.update(ran)

        return dataset

    def KNN_imputation(self, dataset, k=4):
        # only for numerical values
        # Nearest neighbor imputations which weights samples
        # using the mean squared difference on features for which two
        # rows both have observed data.

        from fancyimpute import KNN
        # Create a copy of the dataset to avoid modifying the original one
        df = dataset.copy()

        # Select numerical columns
        numerical_columns = dataset.select_dtypes(['number']).columns

        # Check if there are any missing values in the numerical columns
        if dataset[numerical_columns].isnull().sum().sum() > 0:
            # Perform KNN imputation on the numerical columns
            X_imputed = KNN(k=k, verbose=False).fit_transform(dataset[numerical_columns])

            # Create a DataFrame from the imputed values
            df_imputed = pd.DataFrame(X_imputed, columns=numerical_columns, index=dataset.index)

            # Replace the original numerical columns with the imputed values
            df[numerical_columns] = df_imputed

        return df

        # df = dataset

        # if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

        #     X = dataset.select_dtypes(['number'])

        #     for i in X.columns:

        #         X[i] = KNN(k=k, verbose=False).fit_transform(X)

        #     Z = dataset.select_dtypes(include=['object'])

        #     df = pd.DataFrame.from_records(
        #         X, columns=dataset.select_dtypes(['number']).columns)

        #     df = df.join(Z)

        # else:

        #     pass

        # return df

    def MICE_imputation(self, dataset):
        # only for numerical values
        # Multivariate Imputation by Chained Equations only suitable
        # for Missing At Random (MAR),
        # which means that the probability that a value is missing
        # depends only on observed values and not on unobserved values

        import impyute as imp

        df = dataset

        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

            X = imp.mice(dataset.select_dtypes(['number']).iloc[:, :].values)

            Z = dataset.select_dtypes(include=['object'])

            df = pd.DataFrame.from_records(
                X, columns=dataset.select_dtypes(['number']).columns)

            df = df.join(Z)

        else:

            pass

        return df

    def EM_imputation(self, dataset):
        # only for numerical values
        # imputes given data using expectation maximization.
        # E-step: Calculates the expected complete data log
        # likelihood ratio.

        import impyute as imp

        df = dataset

        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

            X = imp.em(dataset.select_dtypes(['number']).iloc[:, :].values)

            Z = dataset.select_dtypes(include=['object'])

            df = pd.DataFrame.from_records(

                X, columns=dataset.select_dtypes(['number']).columns)

            df = df.join(Z)

        else:

            pass

        return df
    

    def get_config_dict(self, function_name):
        config_dict = {}
        if self.config is not None:
            if function_name in self.config.keys():
                config_dict = self.config[function_name]
        return config_dict
    

    def get_numerical_columns(self):
        from pandas.api.types import is_numeric_dtype

        data = self.dataset
        numerical_columns = []
        for col_name in data:
            if is_numeric_dtype(data[col_name]):
                numerical_columns.append(col_name)
        
        return numerical_columns


    def handle_categorical(self):

        from sklearn.preprocessing import OrdinalEncoder
        from pandas.api.types import is_numeric_dtype

        data = self.dataset

        print(f"\n\n **BEFORE HANDLE CATEGORICAL** \n\n {data}")

        oe_dict = {}

        for col_name in data:
            if not is_numeric_dtype(data[col_name]):
                oe_dict[col_name] = OrdinalEncoder()
                col = data[col_name]
                col_not_null = col[col.notnull()]
                reshaped_values = col_not_null.values.reshape(-1, 1) # TODO is this reshaping really needed? It might cause problems
                encoded_values = oe_dict[col_name].fit_transform(reshaped_values)
                data.loc[col.notnull(), col_name] = np.squeeze(encoded_values)
        
        print(f"\n\n **AFTER HANDLE CATEGORICAL** \n\n {data}")

        return oe_dict
    

    def inverse_encoding(self, oe_dict):
        data = self.dataset
        for col_name in oe_dict.keys():
            col = data[col_name]
            col_not_null = col[col.notnull()]
            reshaped = col_not_null.values.reshape(-1, 1)
            intermediate_value = oe_dict[col_name].inverse_transform(reshaped)
            temp = []
            for entry in intermediate_value:
                temp.append(entry[0])
            data[col_name] = temp


    def KNN_imputer(self):
        from fancyimpute import KNN
        from sklearn.impute import KNNImputer
        
        oe_dict = self.handle_categorical()
        data = self.dataset
        current_method = sys._getframe().f_code.co_name
        hyperparameters = self.get_config_dict(current_method)

        print(f'\n\n **Starting KNN Imputation** \n\n')

        knn_imputer = KNNImputer(n_neighbors=1) 
        for col in oe_dict.keys(): # Handling Categorical Data with 1 neighbor to avoid generating wrong data
            reshaped = data[col].values.reshape(-1, 1)
            temp = knn_imputer.fit_transform(reshaped)
            data[col] = np.squeeze(temp)
        
        n_neighbors = 5
        n_neighbors = hyperparameters.get("n_neighbors", n_neighbors)

        knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        
        for col in data.columns:
            if col not in oe_dict.keys(): # Handling Numerical Data with several neighbors
                reshaped = data[col].values.reshape(-1, 1)
                temp = knn_imputer.fit_transform(reshaped)
                data[col] = np.squeeze(temp)

        # print(data)
        # self.inverse_encoding(oe_dict)

        # print(f'\n\n {data}')
        # self.handle_categorical()

        return data
    

    def complete_case_analysis(self):
        data = self.dataset
        original_rows = len(data)
        #non_missing_indices = ~np.isnan(data.columns)
        data.dropna(inplace=True)
        remaining_rows = len(data)
        rows_removed = original_rows - remaining_rows
        success_msg = f'Complete Case Analysis (CCA) is successfully completed. Removed {rows_removed} rows, remaining {remaining_rows} rows.'
        print(success_msg)
        self.handle_categorical()
        return data
    

    def multiple_imputation(self):
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        oe_dict = self.handle_categorical()

        # Perform multiple imputation to fill in missing values
        # print(f'\n\n\ Size before MI: {len(self.dataset)} \n\n')
        df = self.dataset

        current_method = sys._getframe().f_code.co_name
        max_iter = 10
        random_state = 0
        min_value = 0
        hyperparameters = self.get_config_dict(current_method)
        
        max_iter = hyperparameters.get("max_iter", max_iter)
        random_state = hyperparameters.get("random_state", random_state)
        min_value = hyperparameters.get("min_value", min_value)
        
        multiple_imputer = IterativeImputer(max_iter=max_iter, random_state=random_state, min_value=min_value)
        imputed_values = multiple_imputer.fit_transform(df)
        imputed_values = pd.DataFrame(imputed_values, columns=df.columns)        
        self.dataset = imputed_values
        df = imputed_values

        # print(f"\n\n After Imputing::: {df} \n\n")
        # print(f"\n\n Before Inverse Encoding::: \n\n {df}")

        # print(f'\n\n\ Size after MI: {len(df)} \n\n')
        # #self.inverse_encoding(oe_dict)
        # print(f"\n\n After Inverse Encoding::: \n\n {df}")
        return df
    

    def inverse_probability_weighting(self, weights):
        # Apply inverse probability weighting to handle missing data
        print(f'\n\n\ Size before IPW: {len(self.dataset)} ')

        test = self.dataset.copy()
        test = test.dropna()

        print(f'\n\n\ Size before IPW but with dropping NaNs: {len(test)} ')


        # Make a copy of the dataset to avoid modifying the original dataset
        weighted_df = self.dataset

       # Impute missing values using mean imputation and handle infinite values
        for column in weighted_df.columns:
            if weighted_df[column].isnull().sum() > 0:
                mean_value = weighted_df[column].mean()
                weighted_df[column].fillna(mean_value, inplace=True)
            if np.isinf(weighted_df[column]).sum() > 0:
                finite_values_mean = weighted_df[column].replace([np.inf, -np.inf], np.nan).mean()
                weighted_df[column].replace([np.inf, -np.inf], finite_values_mean, inplace=True)

        # Apply weights
        weighted_df = weighted_df * weights[:, np.newaxis]

        test = weighted_df.copy()
        test = test.dropna()
        print(f'\n\n\ Size after IPW but with dropping NaNs: {len(test)} ')


        print(f'\n\n\ Size after IPW: {len(weighted_df)} \n\n')

        return weighted_df
    

    def simple_mean_imputation(self):
        from sklearn.impute import SimpleImputer

        imputed_data = self.dataset
        mode_imputer = SimpleImputer(strategy='most_frequent')

        numerical_cols = self.get_numerical_columns()

        for col in numerical_cols:
            imputed_data[col].fillna((imputed_data[col].mean()), inplace=True)

        

        for col in imputed_data:
            if col not in numerical_cols:
                reshaped = imputed_data[col].values.reshape(-1, 1)
                temp = mode_imputer.fit_transform(reshaped)
                imputed_data[col] = np.squeeze(temp)
        
        self.handle_categorical()
                
        return imputed_data
    

    def simple_median_imputation(self):
        from sklearn.impute import SimpleImputer

        imputed_data = self.dataset
        mode_imputer = SimpleImputer(strategy='most_frequent')

        numerical_cols = self.get_numerical_columns()

        for col in numerical_cols:
            imputed_data[col].fillna((imputed_data[col].median()), inplace=True)

        

        for col in imputed_data:
            if col not in numerical_cols:
                reshaped = imputed_data[col].values.reshape(-1, 1)
                temp = mode_imputer.fit_transform(reshaped)
                imputed_data[col] = np.squeeze(temp)
        
        self.handle_categorical()
                
        return imputed_data
    

    def transform(self):

        start_time = time.time()

        print(">>Imputation ")
        if self.mode == "survival":
            if self.strategy == "CCA":
                dn = self.complete_case_analysis()

            elif self.strategy == "MI":
                dn = self.multiple_imputation()

            elif self.strategy == "IPW":
                weights = np.random.rand(len(self.dataset))
                dn = self.inverse_probability_weighting(weights)

            elif self.strategy == "Mean":
                dn = self.simple_mean_imputation()
            
            elif self.strategy == "Median":
                dn = self.simple_median_imputation()
            
            elif self.strategy == "KNN":
                dn = self.KNN_imputer()
            
            else:
                raise ValueError("Strategy invalid. Please "
                                         "choose between "
                                         "'CCA', 'MI', 'IPW', 'Mean' or 'Median'")
            

            print("After missing values handling:")

            print("Total", self.dataset.isnull(
                    ).sum().sum(), "rows are affected")
            print("Missing values handling done with -- CPU time: %s seconds" %
              (time.time() - start_time))
            #print(dn)
            return dn
        

        else:
            
            impd = self.dataset

            for key in ['train', 'test']:

                if (not isinstance(self.dataset[key], dict)):

                    d = self.dataset[key].copy()

                    print("* For", key, "dataset")

                    total_missing_before = d.isnull().sum().sum()

                    Num_missing_before = d.select_dtypes(
                        include=['number']).isnull().sum().sum()

                    NNum_missing_before = d.select_dtypes(
                        exclude=['number']).isnull().sum().sum()

                    print("Before imputation:")

                    if total_missing_before == 0:

                        print("No missing values in the given data")

                    else:

                        print("Total", total_missing_before, "missing values in",
                            d.columns[d.isnull().any()].tolist())

                        if Num_missing_before > 0:

                            print("-", Num_missing_before,
                                "numerical missing values in",
                                d.select_dtypes(['number']).
                                columns[d.select_dtypes(['number']).
                                        isnull().any()].tolist())

                        if NNum_missing_before > 0:

                            print("-", NNum_missing_before,
                                "non-numerical missing values in",
                                d.select_dtypes(['object']).
                                columns[d.select_dtypes(['object']).
                                        isnull().any()].tolist())

                        if (self.strategy == "EM"):

                            dn = self.EM_imputation(d)

                        elif (self.strategy == "MICE"):

                            dn = self.MICE_imputation(d)

                        elif (self.strategy == "KNN"):

                            dn = self.KNN_imputation(d)

                        elif (self.strategy == "RAND"):

                            dn = self.NaN_random_replace(d)

                        elif (self.strategy == "MF"):

                            dn = self.MF_most_frequent_imputation(d)

                        elif (self.strategy == "MEAN"):

                            dn = self.mean_imputation(d)

                        elif (self.strategy == "MEDIAN"):

                            dn = self.median_imputation(d)

                        elif (self.strategy == "DROP"):

                            dn = self.NaN_drop(d)

                        else:

                            raise ValueError("Strategy invalid. Please "
                                            "choose between "
                                            "'EM', 'MICE', 'KNN', 'RAND', 'MF', "
                                            "'MEAN', 'MEDIAN', or 'DROP'")

                        impd[key] = dn

                        print("After imputation:")

                        print("Total", impd[key].isnull(
                        ).sum().sum(), "missing values")

                        print("-", impd[key].select_dtypes(include=['number']
                                                        ).isnull().sum().sum(),
                            "numerical missing values")

                        print("-", impd[key].select_dtypes(exclude=['number']
                                                        ).isnull().sum().sum(),
                            "non-numerical missing values")

                else:

                    print("No", key, "dataset, no imputation")

            print("Imputation done -- CPU time: %s seconds" %
                (time.time() - start_time))

            print()

            return impd
