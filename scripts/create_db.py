"""Create MySQL database if it does not exist."""

import os
import pymysql


DB_NAME = os.getenv("DB_NAME", "poultry_ai")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "kana")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")


def main() -> None:
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        autocommit=True,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    finally:
        conn.close()

    print(f"Database ensured: {DB_NAME}")


if __name__ == "__main__":
    main()
