#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille <laure.berti@ird.fr>

import warnings
import time
import numpy as np


class Outlier_detector():
    """
    Identify and remove outliers using a particular strategy

    Parameters
    ----------
    * dataset: input dataset dict
        including dataset['train'] pandas DataFrame, dataset['test']
        pandas DataFrame and dataset['target'] pandas DataSeries
        obtained from train_test_split function of Reader class

    * threshold: float, default = '0.3' for any outlying value in a row"
        or a value in [0,1] for multivariate "
        outlying row. For example, with threshold=0.5
        if a row has outlying values in half of the attribute set and more,
        it is considered as an outlier and removed"

    * strategy: str, default = 'ZSB'
        The choice for outlier detection and removal strategy:
            - 'ZSB', 'IQR and 'LOF' for numerical values
            Available strategies =
            'ZS': detects outliers using the robust Zscore as a function
            of median and median absolute deviation (MAD)
            'IQR': detects outliers using Q1 and Q3 +/- 1.5*InterQuartile Range
            'LOF': detects outliers using Local Outlier Factor

    * verbose: Boolean,  default = 'False' otherwise display
        about outlier detected and removed

    * exclude: str, default = 'None' name of variable to be
        excluded from outlier detection
    """

    def __init__(self, dataset, strategy='ZSB', threshold=0.3, mode="original", time_col=None,
                 event_col=None, config=None, verbose=False, exclude=None): # mode determines original function or survival analysis

        self.dataset = dataset

        self.strategy = strategy

        self.threshold = threshold

        self.mode = mode

        self.time_column = time_col

        self.event_column = event_col

        self.config = config

        self.verbose = verbose

        self.exclude = exclude  # to implement

    def get_params(self, deep=True):

        return {'strategy': self.strategy,

                'threshold': self.threshold,

                'verbose': self.verbose,

                'exclude': self.exclude

                }

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`outlier_detector.get_params().keys()`")

            else:

                setattr(self, k, v)

    def IQR_outlier_detection(self, dataset, threshold):

        X = dataset.select_dtypes(['number'])

        Y = dataset.select_dtypes(['object'])

        if len(X.columns) < 1:

            print(
                "Error: Need at least one numeric variable for LOF"
                "outlier detection\n Dataset inchanged")

        Q1 = X.quantile(0.25)

        Q3 = X.quantile(0.75)

        IQR = Q3 - Q1

        outliers = X[((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))]

        to_drop = X[outliers.sum(axis=1)/outliers.shape[1] >
                    threshold].index

        to_keep = set(X.index) - set(to_drop)

        if (threshold == -1):

            X = X[~((X < (Q1 - 1.5 * IQR)) |
                  (X > (Q3 + 1.5 * IQR))).any(axis=1)]

        else:

            X = X.loc[list(to_keep)]

        df = X.join(Y)

        print(len(to_drop), "outlying rows have been removed")

        if len(to_drop) > 0:

            if self.verbose:

                print("with indexes:", list(to_drop))

                print()

                print("Outliers:")

                print(dataset.loc[to_drop])

                print()

        return df

    def ZSB_outlier_detection(self, dataset, threshold):
        # Robust Zscore as a function of median and median
        # absolute deviation (MAD)  defined as
        # z-score = |x – median(x)| / mad(x)

        X = dataset.select_dtypes(['number'])

        Y = dataset.select_dtypes(['object'])

        median = X.apply(np.median, axis=0)

        median_absolute_deviation = 1.4296 * \
            np.abs(X - median).apply(np.median, axis=0)

        modified_z_scores = (X - median) / median_absolute_deviation

        outliers = X[np.abs(modified_z_scores) > 1.6]

        to_drop = outliers[(outliers.count(axis=1) /
                            outliers.shape[1]) > threshold].index

        to_keep = set(X.index) - set(to_drop)

        if (threshold == -1):

            X = X[~(np.abs(modified_z_scores) > 1.6).any(axis=1)]

        else:
            # e.g., remove rows where  40% of variables have zscore
            # above a threshold = 0.4
            X = X.loc[list(to_keep)]

        df = X.join(Y)

        print(len(to_drop), "outlying rows have been removed:")

        if len(to_drop) > 0:

            if self.verbose:

                print("with indexes:", list(to_drop))

                print()

                print("Outliers:")

                print(dataset.loc[to_drop])

                print()

        return df

    def LOF_outlier_detection(self, dataset, threshold):
        # requires no missing value
        # select top 10 outliers

        from sklearn.neighbors import LocalOutlierFactor

        if dataset.isnull().sum().sum() > 0:

            dataset = dataset.dropna()

            print(
                "LOF requires no missing values, so missing values "
                "have been removed using DROP.")

        X = dataset.select_dtypes(['number'])

        Y = dataset.select_dtypes(['object'])

        k = int(threshold*100)

        if len(X.columns) < 1 or len(X) < 1:

            print(
                'Error: Need at least one continous variable for'
                'LOF outlier detection\n Dataset inchanged')

            df = dataset

        else:
            # fit the model for outlier detection (default)
            clf = LocalOutlierFactor(n_neighbors=4, contamination=0.1)
            # use fit_predict to compute the predicted labels of
            # the training samples (when LOF is used for
            # outlier detection, the estimator has no predict,
            # decision_function and score_samples methods).

            clf.fit_predict(X)
            # The higher, the more normal.

            LOF_scores = clf.negative_outlier_factor_
            # Inliers tend to have a negative_outlier_factor_
            # close to -1, while outliers tend to have a larger  score.

            top_k_idx = np.argsort(LOF_scores)[-k:]

            top_k_values = [LOF_scores[i] for i in top_k_idx]

            data = X[LOF_scores < top_k_values[0]]

            to_drop = X[~(LOF_scores < top_k_values[0])].index

            df = data.join(Y)

            print(k, "outlying rows have been removed")

            if len(to_drop) > 0:

                if self.verbose:

                    print("with indexes:", list(to_drop))

                    print()

                    print("Outliers:")

                    print(dataset.loc[to_drop])

                    print()

        return df
    

    def survival_analysis_with_fdr_control(self): # TODO Fix this method from NaNs and return correct data

        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        from statsmodels.stats import multitest

        #self.dataset.dropna(inplace=True)

        # Split the data into groups for comparison (e.g., treatment vs. control)
        groups = self.dataset[self.event_column].unique()

        # Initialize Kaplan-Meier estimator
        kmf = KaplanMeierFitter()

        # Create a dictionary to store results for each group
        results_dict = {}

        for group in groups:
            # Select data for the current group
            group_data = self.dataset[self.dataset[self.event_column] == group]

            # Fit the Kaplan-Meier survival curve for the current group
            kmf.fit(group_data[self.time_column], event_observed=group_data[self.event_column])

            # Perform log-rank test for the current group
            results = logrank_test(group_data[self.time_column], group_data[self.time_column])

            # Store the results for the current group
            results_dict[group] = results

        # Correct p-values for multiple testing using the Benjamini-Hochberg procedure
        p_values = [result.p_value for result in results_dict.values()]
        _, adjusted_p_values, _, _ = multitest.multipletests(p_values)

        # Threshold for controlling FDR
        fdr_threshold = 0.05  # Adjust as needed

        # Identify outliers based on adjusted p-values
        outliers = [group for group, adj_p in zip(groups, adjusted_p_values) if adj_p < fdr_threshold]

        print(outliers)

        # Number of detected outliers
        num_outliers = len(outliers)

        # Number of remaining rows
        num_remaining_rows = len(self.dataset) - num_outliers

        # Print results
        print("Number of Detected Outliers:", num_outliers)
        print("Number of Remaining Rows:", num_remaining_rows)
        return self.dataset

    def martingale_residuals(self): # TODO Fix this method by actually discovering outliers correctly

        from lifelines import KaplanMeierFitter

        x = self.dataset #.dropna()
        kmf = KaplanMeierFitter()
        kmf.fit(x[self.time_column], event_observed=x[self.event_column])
        
        observed_events = x[self.event_column]
        expected_events = kmf.event_table['observed'][:-1]
        
        x['martingale_residuals'] = observed_events - expected_events
        
        # Calculate outliers based on a threshold (e.g., absolute value > 1)
        outlier_threshold = 1
        x['is_outlier'] = abs(x['martingale_residuals']) > outlier_threshold
        
        num_dropped_outliers = x['is_outlier'].sum()
        num_remaining_rows = len(x) - num_dropped_outliers
        
        print("Number of Dropped Outliers (Martingale Residuals):", num_dropped_outliers)
        print("Number of Remaining Rows (Martingale Residuals):", num_remaining_rows)
        
        # Drop outliers from the dataset
        data = x[~x['is_outlier']]

        data.drop(["is_outlier", "martingale_residuals"], axis=1, inplace=True)
        
        return data
    

    def multivariate_outliers(self):

        from sklearn.covariance import EllipticEnvelope

        dataset = self.dataset #.dropna()
        print(dataset)
        envelope = EllipticEnvelope()

        needed_values = dataset[[self.time_column, self.event_column]]
        temp = envelope.fit_predict(needed_values)
        dataset['multivariate_outliers'] = temp
        
        num_dropped_outliers = dataset[dataset['multivariate_outliers'] != 1].shape[0]
        num_remaining_rows = len(dataset) - num_dropped_outliers
        
        print("Number of Dropped Outliers (Multivariate Outliers):", num_dropped_outliers)
        print("Number of Remaining Rows (Multivariate Outliers):", num_remaining_rows)
        
        # Drop outliers from the dataset
        dataset = dataset[dataset['multivariate_outliers'] == 1]

        dataset.drop(['multivariate_outliers'], axis=1, inplace=True)

        return dataset
    

    def transform(self):

        start_time = time.time()

        print()

        print(">>Outlier detection and removal:")

        if self.mode == "survival":
            if (self.strategy == "CR"):
                dn = self.survival_analysis_with_fdr_control()

            elif(self.strategy == "MR"):
                dn = self.martingale_residuals()

            elif(self.strategy == "MUO"):
                dn = self.multivariate_outliers()

            else:
                raise ValueError("Strategy invalid."
                                "Please choose between "
                                "'CR', 'MR' or 'MUO'")
                
            print("Outlier detection and removal done --  CPU time: %s seconds" % (time.time() - start_time))

            print()

            return dn

        else:

            osd = self.dataset

            for key in ['train', 'test']:

                if (not isinstance(self.dataset[key], dict)):

                    if not self.dataset[key].empty:

                        print("* For", key, "dataset")

                        d = self.dataset[key]

                        if (self.strategy == "ZSB"):

                            dn = self.ZSB_outlier_detection(d, self.threshold)

                        elif (self.strategy == 'IQR'):

                            dn = self.IQR_outlier_detection(d, self.threshold)

                        elif (self.strategy == "LOF"):

                            dn = self.LOF_outlier_detection(d, self.threshold)

                        else:

                            raise ValueError("Threshold invalid. "
                                            "Please choose between "
                                            "'-1' for any outlying value in "
                                            "a row or a value in [0,1] for "
                                            "multivariate outlying row. For "
                                            "example,  with threshold=0.5 "
                                            "if a row has outlying values in "
                                            "half of the attribute set and more, "
                                            "it is considered as an outlier and "
                                            "removed")
                        osd[key] = dn
                        # if key == 'test':
                        #     print(f"IN OUTLIER DETECTION ----->    {osd['target_test']} \n\n\n\n")
                        #     osd['target_test'] = dn  # ensuring that target_test and test are of the same size
                        #     print(f"IN OUTLIER DETECTION AFTER PROCESS ----->    {osd['target_test']}")

                    else:

                        print("No outlier detection for", key, "dataset")

                else:

                    print("No outlier detection for", key, "dataset")

            print("Outlier detection and removal done -- CPU time: %s seconds" %
                (time.time() - start_time))

            print()

            return osd
