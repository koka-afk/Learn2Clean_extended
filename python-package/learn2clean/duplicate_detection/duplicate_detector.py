#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille <laure.berti@ird.fr>

import time
import warnings
import numpy as np
import jellyfish as jf
import py_stringsimjoin as ssj
import py_stringmatching as sm
import pandas as pd
pd.options.mode.chained_assignment = None


def add_key_reindex(dataset, rand=False):

    if rand:

        dataset = dataset.reindex(np.random.permutation(dataset.index))

    dataset['New_ID'] = range(1, 1+len(dataset))

    dataset['New_ID'].apply(str)

    return(dataset)


class Duplicate_detector():
    """
    Remove the duplicate records from the dataset

    Parameters
    ----------
    * dataset: input dataset dict
        including dataset['train'] pandas DataFrame, dataset['test']
        pandas DataFrame and dataset['target'] pandas DataSeries
        obtained from train_test_split function of Reader class

    * threshold: float, default = '0.6' only for 'AD' strategy

    * strategy: str, default = 'ED'
        The choice for the deduplication strategy : 'ED', 'AD' or 'METRIC'
        Available strategies =
        'ED':  exact duplicate detection/removal or
        'AD':  for aproximate duplicate records detection and removal
        based on Jaccard similarity or
        'METRIC': using a particular distance specificied in 'metric':
                'DL' (by default) for  Damerau Levenshtein Distance
                'LM for Levenshtein Distance or
                'JW' for Jaro-Winkler Distance

    * metric: str, default = 'DL'  only used for 'AD' strategy

    * verbose: Boolean,  default = 'False' otherwise display the list of
        duplicate rows that have been removed

    * exclude: str, default = 'None' name of variable to
        be excluded from deduplication
    """

    def __init__(self, dataset, strategy='ED', threshold=0.6, time_col=None,
                 event_col=None, metric='DL', mode="original", config=None,  verbose=False, exclude=None): # mode determines original function or survival analysis

        self.dataset = dataset

        self.strategy = strategy

        self.threshold = threshold

        self.time_column = time_col

        self.event_column = event_col

        self.mode = mode

        self.config = config

        self.metric = metric

        self.verbose = verbose

        self.exclude = exclude  # to implement

    def get_params(self, deep=True):

        return {'strategy': self.strategy,

                'metric': self.metric,

                'threshold':  self.threshold,

                'verbose': self.verbose,

                'exclude': self.exclude

                }

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`duplicate_detector.get_params().keys()`")

            else:

                setattr(self, k, v)

    def ED_Exact_duplicate_removal(self, dataset):

        if not dataset.empty:

            df = dataset.drop_duplicates()

            print('Initial number of rows:', len(dataset))

            print('After deduplication: Number of rows:', len(df))

        else:

            print("No duplicate detection, empty dataframe")

        return df

    def AD_Approx_string_duplicate_removal(self, dataset,
                                           threshold, metric="DL"):
        # only for non numerical data
        # NOT OPTIMIZED - TODO

        dataset = add_key_reindex(dataset, rand=True)

        data = dataset.applymap(str)

        data = data.apply(lambda x: '*'.join(x.values.tolist()),
                          axis=1)

        data = data.astype(str)

        data = data.str.replace(" ", "")

        # delchars = ''.join(c for c in map(chr,
        # range(256)) if not c.isalnum())
        for row in data.index:

            data[row] = data[row].lower()

        out = pd.DataFrame(columns=["Dup_ID1", "Dup_ID2", "Dup_1", "Dup_2"])

        if metric == "DL":  # Damerau Levenshtein Distance

            res = {_d: [] for _d in data}

            for _d in res.keys():

                for row in data.index:

                    if _d != data[row] \
                        and jf.damerau_levenshtein_distance(_d, data[row]) < \
                            ((len(_d) + len(data[row])/2)*threshold):

                        res[_d].append(data[row])

                        out.loc[len(out)] = (
                            _d.split("*")[-1], row, _d, data[row])

        elif metric == "LM":  # Levenshtein Distance

            res = {_d: [] for _d in data}

            for _d in res.keys():

                for row in data.index:

                    if _d != data[row] \
                        and jf.levenshtein_distance(_d, data[row]) < \
                            ((len(_d) + len(data[row])/2)*threshold):

                        res[_d].append(data[row])

                        out.loc[len(out)] = (
                            _d.split("*")[-1], row, _d, data[row])

        elif metric == "JW":  # Jaro-Winkler Distance

            res = {_d: [] for _d in data}

            for _d in res.keys():

                for row in data.index:

                    if _d != data[row] and jf.jaro_winkler(_d, data[row]) >  \
                            ((len(_d) + len(data[row])/2)*threshold):

                        res[_d].append(data[row])

                        out.loc[len(out)] = (
                            _d.split("*")[-1], row, _d, data[row])

        filtered = {k: v for k, v in res.items() if len(v) > 0}

        out = out[~out[["Dup_ID1", "Dup_ID2"]].apply(
            frozenset, axis=1).duplicated()]

        out.reset_index(drop=True, inplace=True)
        # d = dataset['New_ID'].astype(str)
        if self.verbose:

            print("Duplicates IDs:", out)

            dups = pd.DataFrame.from_dict(filtered, orient='index')

            print("Duplicates:", dups)

            print("Duplicates removed: ",
                  dataset[dataset['New_ID'].isin(out['Dup_ID2'])])

        df = dataset[~dataset['New_ID'].isin(out['Dup_ID2'])]

        print("Number of duplicate rows removed:", len(dataset)-len(df))

        return df

    def jaccard_similarity(self, dataset, threshold):

        df = add_key_reindex(dataset)
        # concatenate all columns and convert as one string
        # for each row with '*' as separator

        A = dataset.applymap(str)

        A = A.apply(lambda x: '*'.join(x.values.tolist()), axis=1)

        A = A.astype(str)

        A = A.str.replace(" ", "")

        df['row'] = A

        ssj.profile_table_for_join(df)

        ws = sm.WhitespaceTokenizer(return_set=True)

        # auto join
        output_pairs = ssj.jaccard_join(df, df, 'New_ID',
                                        'New_ID', 'row', 'row', ws,
                                        threshold, l_out_attrs=['row'],
                                        r_out_attrs=['row'], n_jobs=-1)

        dup = output_pairs[output_pairs['l_New_ID']
                           != output_pairs['r_New_ID']]

        dataset = df[~df['New_ID'].isin(dup['r_New_ID'])]

        dataset.drop(["New_ID", "row"], axis=1, inplace=True)

        print("Number of duplicate rows removed:", len(set(dup['r_New_ID'])))

        return dataset
    

    # Function for generating unique event IDs
    def generate_event_ids(self):
        # Combine selected columns to create a unique event identifier
        data = self.dataset #.copy()
        data['event_id'] = data[self.time_column].astype(str) + data[self.event_column].astype(str)
        return data
    

    # Function for Unique Event Identifier-Based Deduplication
    def deduplicate_by_event_id(self, df_with_event_ids):
        dataset = df_with_event_ids

        initial_rows = dataset.shape[0]  # Get the initial number of rows

        df_deduplicated = dataset.drop_duplicates(subset=["event_id"])

        final_rows = df_deduplicated.shape[0]  # Get the final number of rows after deduplication

        num_duplicates = initial_rows - final_rows  # Calculate the number of duplicate rows removed

        print(f"Number of Duplicate Rows identified by event id: {num_duplicates}")

        print(f"Number of Rows After Deduplication by event id: {final_rows}")

        print(df_deduplicated)

        # Remove the 'event_id' column from df_deduplicated
        df_deduplicated.drop(["event_id"], axis=1, inplace=True)

        return df_deduplicated
    

    # Function for Timestamp-Based Deduplication
    def deduplicate_by_timestamp(self):

        dataset = self.dataset

        initial_rows = dataset.shape[0]  # Get the initial number of rows

        new_col = "timestamp"

        dataset[new_col] = pd.to_datetime(dataset[self.time_column])

        df_deduplicated = dataset.sort_values(by=self.time_column).drop_duplicates(subset=new_col)

        final_rows = df_deduplicated.shape[0]  # Get the final number of rows after deduplication

        num_duplicates = initial_rows - final_rows  # Calculate the number of duplicate rows removed

        print(f"Number of Duplicate Rows identified by time: {num_duplicates}")
        print(f"Number of Rows After Deduplication identified by time of event: {final_rows}")

        dataset.drop([new_col], axis=1, inplace=True)
        df_deduplicated.drop([new_col], axis=1, inplace=True) # no need for the column any more because it causes problems

        return df_deduplicated
    

    def Exact_duplicate_removal(self):
        
        dataset = self.dataset

        if len(dataset) > 0:

            df = dataset.drop_duplicates()

            print('Initial number of rows:', len(dataset))

            print('After deduplication: Number of rows:', len(df))

        else:
            df = dataset
            print("No duplicate detection, empty dataframe")

        return df
    

    def transform(self):

        start_time = time.time()

        print()

        print(">>Duplicate detection and removal:")

        if self.mode == "survival":
            if (self.strategy == "DBID"):
                print (" started using event ID based deduplication .....")
                event_id = self.generate_event_ids()
                print(event_id)
                dn = self.deduplicate_by_event_id(event_id)

            elif (self.strategy == 'DBT'):
                print(" started using timestamp-based deduplication .....")
                dn = self.deduplicate_by_timestamp()

            elif (self.strategy == "ED"):
                print(" started using exact duplicate removal mehtod .....")
                dn = self.Exact_duplicate_removal()

            else:
                raise ValueError("Strategy invalid."
                                "Please choose between "
                                "'DBID', 'DBT' or 'ED'")
            
            print("Deduplication done -- CPU time: %s seconds" %
              (time.time() - start_time))
            print()

            return dn

        else:

            dedup = self.dataset

            for key in ['train', 'test']:

                if (not isinstance(self.dataset[key], dict)):

                    if not self.dataset[key].empty:

                        print("* For", key, "dataset")

                        if (self.strategy == "ED"):

                            if self.metric:

                                print("Metric is not considered for 'ED'.")

                            dn = self.ED_Exact_duplicate_removal(self.dataset[key])

                        elif (self.strategy == "AD"):

                            if self.metric:

                                print("Metric is not considered for 'AD'.")

                            dn = self.jaccard_similarity(
                                    self.dataset[key], self.threshold)

                        elif (self.strategy == "METRIC"):

                            if self.metric not in ('DL', 'JW', 'LM'):

                                print("Metric invalid. "
                                    "Please choose between 'LM', 'JW' or 'DL'.")

                            dn = self.AD_Approx_string_duplicate_removal(
                                    self.dataset[key],
                                    metric=self.metric,
                                    threshold=self.threshold)

                        else:
                            raise ValueError("Strategy invalid."
                                            "Please choose between "
                                            "'ED', 'METRIC' or 'AD'")

                        dedup[key] = dn
                        
                        # if key == 'test':
                        #     dedup['target_test'] = dn # ensuring that target_test and test are of the same size

                    else:

                        print("No", key, "dataset, no duplicate detection")

                else:

                    print("No", key, "dataset, no duplicate detection")

            print("Deduplication done -- CPU time: %s seconds" %
                (time.time() - start_time))
            print()

            return dedup
