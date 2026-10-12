#!/usr/bin/env python3
import sqlite3
import sys

def view_database(db_path='model.db'):
    """View contents of SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print(f"No tables found in {db_path}")
            return

        print(f"\n{'='*60}")
        print(f"Database: {db_path}")
        print(f"{'='*60}\n")

        # Display each table
        for table in tables:
            table_name = table[0]

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            # Get schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            print(f"\n{'─'*60}")
            print(f"Table: {table_name} ({row_count} rows)")
            print(f"{'─'*60}")

            # Show columns
            print("\nColumns:")
            for col in columns:
                col_id, col_name, col_type, notnull, default, pk = col
                print(f"  • {col_name:20} {col_type:15} {'NOT NULL' if notnull else ''}")

            # Show sample data
            print(f"\nSample data (first 5 rows):")
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            rows = cursor.fetchall()

            if rows:
                # Print header
                col_names = [description[0] for description in cursor.description]
                print("  " + " | ".join(f"{name:20}" for name in col_names))
                print("  " + "─" * (len(col_names) * 22))

                # Print rows
                for row in rows:
                    print("  " + " | ".join(f"{str(val)[:20]:20}" for val in row))
            else:
                print("  (No data)")

        print(f"\n{'='*60}\n")
        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    db_file = sys.argv[1] if len(sys.argv) > 1 else 'model.db'
    view_database(db_file)
