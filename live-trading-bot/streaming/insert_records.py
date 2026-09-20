import psycopg2

def connect_to_db():
    print("Connecting to Postgre SQL Database ..")
    try:
        conn = psycopg2.connect(
            host="db",
            port=5432,
            dbname="fin",
            user="dbuser",
            password="dbpassword"
        )
        print(conn)
        return conn
    except psycopg2.Error as e:
        print(f"Database Connection Failed {e}")
        raise

def create_table(conn):
    print("Create table if not exists..")
    try:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE SCHEMA IF NOT EXISTS fin;
        CREATE TABLE IF NOT EXISTS fin.stock_quotes(
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        bid_price NUMERIC,
        bid_size INTEGER,
        bid_exchange VARCHAR(5),
        ask_price NUMERIC,
        ask_size INTEGER,
        ask_exchange VARCHAR(5),
        conditions VARCHAR(20),
        tape VARCHAR(5),
        inserted_at TIMESTAMPTZ DEFAULT NOW()
        )
        """)
        conn.commit()
        print("Table was created")
    except psycopg2.Error as e:
        print(f"Failed to create table {e}")
        raise

def insert_quote(conn, quote):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO fin.stock_quotes
        (symbol, timestamp, bid_price, bid_size, bid_exchange,
         ask_price, ask_size, ask_exchange, conditions, tape)
        VALUES (%(symbol)s, %(timestamp)s, %(bid_price)s, %(bid_size)s, %(bid_exchange)s,
                %(ask_price)s, %(ask_size)s, %(ask_exchange)s, %(conditions)s, %(tape)s)
    """, quote)
    conn.commit()
