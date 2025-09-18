import sqlite3
from typing import List, Dict


def list_collections(db_path: str = "milvus_tgps.db") -> List[str]:
    """
    Return a list of table names (collections) present in the SQLite database.

    Args:
        db_path: Path to the SQLite .db file.

    Returns:
        List of table names.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        rows = cursor.fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def get_collection_fields(db_path: str = "milvus_tgps.db") -> Dict[str, List[Dict[str, str]]]:
    """
    For every table in the database, return its fields (columns) and basic metadata.

    Args:
        db_path: Path to the SQLite .db file.

    Returns:
        Dict mapping table name -> list of column metadata dictionaries with keys:
            - cid: column id (int)
            - name: column name (str)
            - type: declared type (str)
            - notnull: 0 or 1
            - dflt_value: default value or None
            - pk: 0 or 1 indicating if it is part of primary key
    """
    collections = list_collections(db_path)
    result: Dict[str, List[Dict[str, str]]] = {}
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        for table in collections:
            cursor.execute(f"PRAGMA table_info('{table}');")
            cols = cursor.fetchall()
            # PRAGMA table_info returns rows: cid, name, type, notnull, dflt_value, pk
            result[table] = [
                {
                    "cid": col[0],
                    "name": col[1],
                    "type": col[2],
                    "notnull": col[3],
                    "dflt_value": col[4],
                    "pk": col[5],
                }
                for col in cols
            ]
    finally:
        conn.close()
    return result


if __name__ == "__main__":
    db = "milvus_tgps.db"
    collections = list_collections(db)
    print("Collections (tables) found in", db)
    for c in collections:
        print(" -", c)
    print()
    fields = get_collection_fields(db)
    for table, cols in fields.items():
        print(f"Fields for {table}:")
        for col in cols:
            print(f"   {col['cid']}: {col['name']} ({col['type']}) notnull={col['notnull']} pk={col['pk']} dflt={col['dflt_value']}")
        print()