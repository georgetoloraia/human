import sqlite3

DB_PATH = "/home/greendragon/Desktop/human/graph.db"  # adjust if you move it

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Counts
cur.execute("SELECT COUNT(*) FROM neurons"); print("neurons:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM edges"); print("edges:", cur.fetchone()[0])

# Simple keyword lookup + neighbors
term = "socket"
cur.execute("SELECT id FROM neurons WHERE label = ? LIMIT 200", (term,))
ids = [row[0] for row in cur.fetchall()]
if ids:
    placeholders = ",".join("?" for _ in ids)
    cur.execute(f"""
        SELECT n2.label, n2.source
        FROM edges e
        JOIN neurons n2 ON n2.id = e.dst
        WHERE e.src IN ({placeholders}) AND e.kind='sequence'
        LIMIT 200
    """, ids)
    print(cur.fetchall())

conn.close()
