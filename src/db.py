import mysql.connector
import pandas as pd

def get_connection():
    return mysql.connector.connect(
        host     = "localhost",
        user     = "root",
        password = "Tanushka@2006",  # change this
        database = "churn_db"
    )

def log_prediction(tenure, city_tier, satisfaction,
                   complain, order_count, cashback,
                   churn_prob, risk_level, prediction):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions
        (tenure, city_tier, satisfaction, complain,
         order_count, cashback, churn_prob, risk_level, prediction)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (tenure, city_tier, satisfaction, complain,
          order_count, cashback, churn_prob, risk_level, prediction))
    conn.commit()
    cursor.close()
    conn.close()

def load_predictions():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM predictions ORDER BY created_at DESC", conn)
    conn.close()
    return df

def insert_customers(df_rfm):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE customers")
    for _, row in df_rfm.iterrows():
        cursor.execute("""
            INSERT INTO customers
            (customer_id, recency, frequency, monetary, segment, churn)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (int(row['CustomerID']), float(row['Recency']),
              float(row['Frequency']), float(row['Monetary']),
              str(row['Segment']), int(row['Churn'])))
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Inserted {len(df_rfm)} customers into MySQL")

if __name__ == "__main__":
    rfm = pd.read_csv('data/rfm_segments.csv')
    insert_customers(rfm)
    print("Database setup complete!")