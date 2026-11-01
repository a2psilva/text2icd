import sqlite3


def test_sqlite_has_fts5_with_bm25() -> None:
    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE VIRTUAL TABLE terms USING fts5(title, tokenize='unicode61 remove_diacritics 2')"
    )
    con.executemany(
        "INSERT INTO terms VALUES (?)",
        [
            ("Pneumonia, aspiration, due to food",),
            ("Diabetes, type 2, with chronic kidney disease",),
        ],
    )
    best = con.execute(
        "SELECT title FROM terms WHERE terms MATCH 'kidney chronic' ORDER BY bm25(terms)"
    ).fetchone()
    assert best == ("Diabetes, type 2, with chronic kidney disease",)
