import psycopg2
from enum import Enum


class DBConnection(Enum):
    host = 'localhost'
    database = 'political_party_classifier'
    user = 'postgres'
    password = 'admin'
    port = '5432'


def make_conn():
    return psycopg2.connect(
        database=DBConnection.database.value,
        user=DBConnection.user.value,
        password=DBConnection.password.value,
        host=DBConnection.host.value,
        port=DBConnection.port.value
    )
