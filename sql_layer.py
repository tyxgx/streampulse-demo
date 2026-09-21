"""SQL router: exact counts / totals / top-N computed with DuckDB over trimmed Parquet.
Same principle as production — compute with SQL, let the LLM only phrase the result."""
import re
from pathlib import Path

import duckdb

_D = Path(__file__).parent / "data"
_con = duckdb.connect()
_con.execute("PRAGMA threads=2")
_con.execute("PRAGMA memory_limit='150MB'")

_COUNTRY = f"read_parquet('{_D}/country.parquet')"
_ARTIST = f"read_parquet('{_D}/artist.parquet')"
_LABEL = f"read_parquet('{_D}/label.parquet')"
_SONG = f"read_parquet('{_D}/song.parquet')"

_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_TOPN_RE = re.compile(r"\btop\s+(\d{1,2})\b")
_COUNT_KEYWORDS = ("how many", "total number of", "number of")
_TOPN_KEYWORDS = ("top ", "highest", "most streamed", "best performing")

_countries = None


def _country_names():
    global _countries
    if _countries is None:
        _countries = [r[0] for r in _con.execute(f"SELECT DISTINCT country_name FROM {_COUNTRY}").fetchall()]
    return _countries


def find_country(lowered):
    # longest name first so "Niger" doesn't shadow "Nigeria"
    for name in sorted(_country_names(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered):
            return name
    return None


def detect_sql_intent(question):
    lowered = question.lower()
    m = _YEAR_RE.search(lowered)
    year = int(m.group(1)) if m else None
    country = find_country(lowered)

    if any(k in lowered for k in _COUNT_KEYWORDS):
        if "countr" in lowered:
            return {"kind": "count", "src": _COUNTRY, "col": "country_name", "label": "countries"}
        if "artist" in lowered:
            return {"kind": "count", "src": _ARTIST, "col": "artist_uri", "label": "artists"}
        if "label" in lowered:
            return {"kind": "count", "src": _LABEL, "col": "standardized_label", "label": "labels"}
        if "song" in lowered or "track" in lowered:
            return {"kind": "count", "src": _SONG, "col": "uri", "label": "tracks"}

    if ("total streams" in lowered or "how many streams" in lowered) and country:
        return {"kind": "sum_streams", "country": country, "year": year}

    if any(k in lowered for k in _TOPN_KEYWORDS):
        n = int(_TOPN_RE.search(lowered).group(1)) if _TOPN_RE.search(lowered) else 5
        return {"kind": "top_n", "table": "artist" if "artist" in lowered else "country", "n": min(n, 20), "year": year}
    return None


def run_sql_intent(intent):
    kind = intent["kind"]
    if kind == "count":
        n = _con.execute(f"SELECT COUNT(DISTINCT {intent['col']}) FROM {intent['src']}").fetchone()[0]
        return n, intent["label"]
    if kind == "sum_streams":
        sql = f"SELECT SUM(total_streams) FROM {_COUNTRY} WHERE country_name = ?"
        params = [intent["country"]]
        if intent["year"]:
            sql += " AND year = ?"
            params.append(intent["year"])
        total = _con.execute(sql, params).fetchone()[0]
        return total, intent["country"] + (f" in {intent['year']}" if intent["year"] else "")
    if kind == "top_n":
        src, name = (_COUNTRY, "country_name") if intent["table"] == "country" else (_ARTIST, "artist_name")
        where, params = ("WHERE year = ?", [intent["year"]]) if intent["year"] else ("", [])
        rows = _con.execute(
            f"SELECT {name}, SUM(total_streams) t FROM {src} {where} GROUP BY {name} ORDER BY t DESC LIMIT {intent['n']}",
            params).fetchall()
        label = "countries" if intent["table"] == "country" else "artists"
        return rows, f"top {intent['n']} {label} by streams" + (f" in {intent['year']}" if intent["year"] else "")
    return None, None


def format_streams(n):
    n = float(n or 0)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.2f}{suf}"
    return f"{n:.0f}"
