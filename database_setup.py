import sqlite3


def init_db():
    conn = sqlite3.connect('dairy_farm.db')
    cursor = conn.cursor()

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS cow_logs
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       timestamp
                       DATETIME
                       DEFAULT
                       CURRENT_TIMESTAMP,
                       cow_id
                       TEXT
                       DEFAULT
                       'COW_001',
                       age
                       INTEGER,
                       breed
                       TEXT,
                       farm_cleanliness
                       TEXT,
                       body_temp
                       REAL,
                       ambient_temp
                       REAL,
                       humidity
                       REAL,
                       activity
                       REAL,
                       activity_state
                       TEXT,
                       milk_ec
                       REAL,
                       milk_yield
                       REAL,
                       risk_percentage
                       REAL
                   )
                   ''')

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Local database 'dairy_farm.db' initialized successfully.")