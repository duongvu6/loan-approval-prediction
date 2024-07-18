import psycopg
import requests

class TestConnection:

    def test_postgresql_connection(self):
        is_connected = False
        try:
            with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
                is_connected = True
        except:
            ...

        assert is_connected is True, "PostgresSQL is not running on port 5432"

    def test_grafana_connection(self):
        try:
            response = requests.get("http://localhost:3000/login")
        except requests.ConnectionError:
            ...

        assert response.status_code == 200, "Grafana is not running on port 3000"

    def test_mlflow_connection(self):
        try:
            response = requests.get("http://localhost:5000")
        except requests.ConnectionError:
            ...

        assert response.status_code == 200, "MLFlow is not running on port 5000"
