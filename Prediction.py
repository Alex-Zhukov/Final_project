import dask.dataframe as dd
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        return X[self.columns]


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("DataFrame не содердит следующие колонки: %s" % cols_error)


def merge_data(df, features):
    data_merged = pd.merge(df, features, on='id')

    duplicated_data = data_merged[data_merged['id'].duplicated(keep=False)].sort_values(by='id')
    duplicated_data['time_diff'] = abs(duplicated_data['buy_time_x'] - duplicated_data['buy_time_y'])
    duplicated_data = duplicated_data.sort_values(by=['Unnamed: 0_x', 'time_diff'])

    before_clean = duplicated_data.index.tolist()
    after_clean = duplicated_data.drop_duplicates(subset=['Unnamed: 0_x', 'id'], keep='first').index.tolist()
    to_delete = list(set(before_clean) - set(after_clean))

    data_merged.drop(to_delete, axis=0, inplace=True)
    data_merged.drop(columns=['Unnamed: 0_x','Unnamed: 0_y', 'buy_time_y'], inplace=True)
    data_merged.rename(columns={"buy_time_x": "buy_time"}, inplace=True)

    del duplicated_data
    return data_merged


TEST_PATH = 'data_test.csv'
FEATURES_PATH = 'features.csv'

df_test = pd.read_csv(TEST_PATH)

df_features = dd.read_csv(FEATURES_PATH, sep="\t")

features_test = df_features.loc[df_features['id'].isin(df_test['id'])].compute()

del df_features

merged_test = merge_data(df_test, features_test)

del features_test

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict_proba(merged_test)

df_test['predictions'] = predictions[:, 1]
df_test.drop(columns=['Unnamed: 0'], inplace=True)

df_test.to_csv('answers_test.csv')
