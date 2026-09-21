"""One-off: turn the 540 MB hf_space demo data into a compact bundle that fits a 512 MB host.

- chunks: keep artist/country/label yearly-grain chunks (drop 344K per-song chunks) -> int8 matrix
- chunk text -> SQLite (read on demand, not held in RAM)
- SQL tables -> trimmed, ZSTD parquet (only the columns the SQL router uses)
"""
import json
import sqlite3
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

SRC = Path(sys.argv[1])          # hf_space/demo_data
OUT = Path(sys.argv[2])          # data/
OUT.mkdir(parents=True, exist_ok=True)
KEEP = ("artist_performance", "country_performance", "label_performance_enhanced")

pf = pq.ParquetFile(SRC / "gold_chunks.parquet")
ids, tables, keys, texts, blocks = [], [], [], [], []
for batch in pf.iter_batches(batch_size=20000):
    d = batch.to_pydict()
    sel = [i for i, t in enumerate(d["source_table"]) if t in KEEP]
    if not sel:
        continue
    emb = np.asarray([d["embedding"][i] for i in sel], dtype=np.float32)
    blocks.append(emb)
    tables += [d["source_table"][i] for i in sel]
    keys += [d["source_key"][i] for i in sel]
    texts += [d["chunk_text"][i] for i in sel]

matrix = np.concatenate(blocks)
n = len(matrix)
scale = 127.0 / float(np.abs(matrix).max())
q = np.clip(np.rint(matrix * scale), -127, 127).astype(np.int8)
deq = q.astype(np.float32) / scale
np.save(OUT / "chunks_q8.npy", q)
np.save(OUT / "norms_sq.npy", (deq * deq).sum(axis=1).astype(np.float32))
json.dump({"scale": scale, "rows": n}, open(OUT / "meta.json", "w"))

db = sqlite3.connect(OUT / "chunks.sqlite")
db.execute("DROP TABLE IF EXISTS chunks")
db.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, source_table TEXT, source_key TEXT, chunk_text TEXT)")
db.executemany("INSERT INTO chunks VALUES (?,?,?,?)", zip(range(n), tables, keys, texts))
db.execute("CREATE INDEX ix_entity ON chunks(source_table, source_key)")
db.commit()
db.close()

con = duckdb.connect()
G = SRC / "gold_sql"
def dump(name, cols, out):
    con.execute(f"""COPY (SELECT {cols} FROM read_parquet('{G}/{name}/**/*.parquet', hive_partitioning=1))
                    TO '{OUT / out}' (FORMAT parquet, COMPRESSION zstd)""")
dump("country_performance", "country_name, CAST(year AS INT) AS year, total_streams", "country.parquet")
dump("artist_performance", "artist_uri, artist_name, CAST(year AS INT) AS year, total_streams", "artist.parquet")
dump("label_performance_enhanced", "standardized_label", "label.parquet")
dump("kpi_song", "uri", "song.parquet")
print("rows", n, "scale", scale)
