import pandas as pd
import os
import pickle
import click
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


load_dotenv()
dest_path = os.getenv("CLEAR_DATA_PATH")


def read_dataframe(filename: str):
    df = pd.read_csv(filename)
    df = df.drop(columns=['loan_id'])

    mapping = {
        'self_employed': {"Yes": 1, "No": 0},
        'loan_status': {"Approved": 1, "Rejected": 0},
        'education': {"Graduate": 1, "NotGraduate": 0}
    }
    for column, map_dict in mapping.items():
        df[column] = df[column].map(map_dict)
    df.fillna(0, inplace=True)

    return df


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)
    
def preprocess_data(df):
    simple_imputer = SimpleImputer(strategy='median')
    pipe_num = Pipeline([('imputer', simple_imputer)])

    s_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
    ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    pipe_cat = Pipeline([('imputer', s_imputer), ('encoder', ohe_encoder)])

    col_transformer = ColumnTransformer([('num_preproc', pipe_num, [x for x in df.columns if df[x].dtype!='object']),
                                        ('cat_preproc', pipe_cat, [x for x in df.columns if df[x].dtype=='object'])])

    return col_transformer.fit_transform(df)


@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the raw data was saved"
)
def run_data_prep(raw_data_path: str):
    # Load data
    df = read_dataframe(raw_data_path)
    train_val_df, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(train_val_df, test_size=0.25, random_state=42)


    # Extract the target
    target = 'loan_status'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    X_train = preprocess_data(df_train)
    X_val = preprocess_data(df_val)
    X_test = preprocess_data(df_test)

    # Save DictVectorizer and datasets
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == '__main__':
    run_data_prep()