import psycopg
import pandas as pd

create_table_statement = """
    drop table if exists testdata;
    create table testdata(
        no_of_dependents float,
        education TEXT,
        self_employed TEXT,
        income_annum float,
        loan_amount float,
        loan_term float,
        cibil_score float,
        residential_assets_value float,
        commercial_assets_value float,
        luxury_assets_value float,
        bank_asset_value float,
        loan_status INT
    )
"""

create_accuracy_table = """
    DROP TABLE IF EXISTS accuracy;
    CREATE TABLE accuracy(
        model_id SERIAL PRIMARY KEY,
        roc_auc FLOAT,
        precision FLOAT,
        recall FLOAT,
        f1_score FLOAT,
        gini_coefficient FLOAT,
        true_positive INT,
        true_negative INT,
        false_positive INT,
        false_negative INT
);
"""

def setup_db():
    with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
        with psycopg.connect("host=localhost port=5432 dbname=testdata user=postgres password=example") as conn:
            conn.execute(create_table_statement)
            conn.execute(create_accuracy_table)


def write_in_db(df_test):
    # prepare for inserting
    df_test["loan_status"] = df_test["loan_status"].map({"Approved": 1, "Rejected": 0},)
    df_test = tuple(map(tuple, df_test.to_numpy()))

    # send data
    with psycopg.connect("host=localhost port=5432 dbname=testdata user=postgres password=example", autocommit=True) as conn:
        with conn.cursor() as curr:
            curr.executemany(
                """
                INSERT INTO 
                    testdata (
                        no_of_dependents,
                        education ,
                        self_employed ,
                        income_annum ,
                        loan_amount ,
                        loan_term ,
                        cibil_score ,
                        residential_assets_value ,
                        commercial_assets_value ,
                        luxury_assets_value ,
                        bank_asset_value ,
                        loan_status
                    ) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                df_test
            )

def get_test_data():
    with psycopg.connect("host=localhost port=5432 dbname=testdata user=postgres password=example", autocommit=True) as conn:
        query = "SELECT * FROM testdata"
        return pd.read_sql_query(query, conn)

def insert_metrics(metrics):
    query = """
        INSERT INTO accuracy (
            roc_auc, precision, recall, f1_score, gini_coefficient,
            true_positive, true_negative,
            false_positive, false_negative
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    with psycopg.connect("host=localhost port=5432 dbname=testdata user=postgres password=example", autocommit=True) as conn:
        with conn.cursor() as curr:
            curr.execute(query, metrics)
    
    return True


if __name__ == "__main__":
    setup_db()
