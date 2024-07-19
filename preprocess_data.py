import pandas as pd
import os
import pickle
import numpy as np
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from db import write_in_db

load_dotenv()
dest_path = os.getenv("CLEAR_DATA_PATH")


def read_dataframe(filename: str):
    df = pd.read_csv(filename)
    df = df.drop(columns=['loan_id'])
    return df

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)
    
def create_preprocess_pipeline(df):
    simple_imputer = SimpleImputer(strategy='median')
    pipe_num = Pipeline([('imputer', simple_imputer)])

    s_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
    ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    pipe_cat = Pipeline([('imputer', s_imputer), ('encoder', ohe_encoder)])

    col_transformer = ColumnTransformer([('num_preproc', pipe_num, [x for x in df.columns if df[x].dtype != 'object']),
                                        ('cat_preproc', pipe_cat, [x for x in df.columns if df[x].dtype == 'object'])])
    
    col_transformer.fit(df)

    return col_transformer

def preprocess_answer(y):
    return np.array(list(map(lambda res: 1 if res == "Approved" else 0, y.values)))


def run_data_prep():
    raw_data_path = "data/raw_data/loan_approval.csv"
    # Load data
    df = read_dataframe(raw_data_path)

    # split data
    train_val_df, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(train_val_df, test_size=0.25, random_state=42)

    # Extract the target
    target = 'loan_status'
    y_train = preprocess_answer(df_train[target])
    y_val = preprocess_answer(df_val[target])
    # y_test = preprocess_answer(df_test[target])

    # drop target column
    df_train = df_train.drop(columns=[target])
    df_val = df_val.drop(columns=[target])
    # df_test = df_test.drop(columns=[target])

    # preprocess data
    pipeline = create_preprocess_pipeline(df_train)
    X_train = pipeline.transform(df_train)
    X_val = pipeline.transform(df_val)
    # X_test = pipeline.transform(df_test)

    # Save DictVectorizer and datasets
    dump_pickle(pipeline, os.path.join(dest_path, "transformer.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))

    # write test data to db
    # y_test = y_test.reshape((len(y_test), 1))
    write_in_db(df_test)


if __name__ == '__main__':
    run_data_prep()