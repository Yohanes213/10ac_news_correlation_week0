import psycopg2
from database import create_table, insert_data
from csv_handler import read_csv
from datetime import datetime
import os

def parse_datetime(dt_str):
    if dt_str:
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
    else:
        return None

def main():
    # Read database credentials from environment variables
    dbname = os.environ.get("DBNAME")
    user = os.environ.get("DBUSER")
    password = os.environ.get("DBPASSWORD")
    host = os.environ.get("DBHOST")
    port = os.environ.get("DBPORT")

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )

    # Create table
    create_table(conn)

    # Read data from CSV
    csv_data = read_csv('data/sample_csv.csv')

    # Insert data into database
    for row in csv_data:
        # Convert None values to NULL
        row = {k: v if v != 'None' else None for k, v in row.items()}
        
        # Convert datetime string to datetime object
        #row['published_at'] = parse_datetime(row['published_at'])
        
        # Extract relevant features
        document_id = row['article_id']
        document_text = row['article']
        label = row['title_sentiment']  # No event probability provided in your DataFrame
        
        # Insert data into database
        insert_data(conn, (
            document_id,
            document_text,
            label
        ))

    # Close connection
    conn.close()

if __name__ == "__main__":
    main()
