"""Small deterministic database used by documentation and CI smoke tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEMO_SQL = """
PRAGMA foreign_keys = ON;
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price_cents INTEGER NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    ordered_at TEXT NOT NULL
);
CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    PRIMARY KEY (order_id, product_id)
);
INSERT INTO customers VALUES (1, 'Alice', 'Lyon'), (2, 'Bob', 'Paris');
INSERT INTO products VALUES (1, 'Notebook', 900), (2, 'Pen', 250), (3, 'Bag', 4200);
INSERT INTO orders VALUES (1, 1, '2026-09-01'), (2, 2, '2026-09-02');
INSERT INTO order_items VALUES (1, 1, 2), (1, 2, 3), (2, 3, 1);
"""


def create_demo_database(destination: str | Path, *, force: bool = False) -> Path:
    path = Path(destination)
    if path.exists() and not force:
        raise FileExistsError(f"database already exists: {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as connection:
        connection.executescript(DEMO_SQL)
    return path

