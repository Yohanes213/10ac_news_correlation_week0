import psycopg2

def create_table(conn):
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE ml_features (
            document_id SERIAL PRIMARY KEY,
            article TEXT,
            title_sentiment TEXT
        )
    """)
    conn.commit()

def insert_data(conn, data):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ml_features (document_id, article, title_sentiment)
        VALUES (%s, %s, %s)
    """, data)
    conn.commit()
