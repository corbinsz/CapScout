"""Local accounts. Passwords are hashed. Emails are stored only so a person
can sign in and so a paid tier can be attached to them. The database file is
gitignored.
"""
import os
import sqlite3
from datetime import datetime, timezone

from werkzeug.security import check_password_hash, generate_password_hash

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT 'free',
    stripe_customer_id TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    city_key TEXT NOT NULL,
    min_score REAL NOT NULL,
    cadence TEXT NOT NULL DEFAULT 'weekly',
    created_at TEXT NOT NULL,
    UNIQUE(email, city_key)
);
CREATE TABLE IF NOT EXISTS event_counts (
    name TEXT NOT NULL,
    day TEXT NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (name, day)
);
"""


def _connect():
    os.makedirs(os.path.dirname(config.ACCOUNTS_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.ACCOUNTS_DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript(_SCHEMA)
    return conn


def _public(row):
    if row is None:
        return None
    return {
        'id': row['id'],
        'email': row['email'],
        'tier': row['tier'],
        'created_at': row['created_at'],
    }


def create_user(email, password):
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    try:
        with _connect() as conn:
            cur = conn.execute(
                'INSERT INTO users (email, password_hash, tier, created_at) VALUES (?, ?, ?, ?)',
                (email, generate_password_hash(password), 'free', now),
            )
            conn.commit()
            row = conn.execute('SELECT * FROM users WHERE id = ?', (cur.lastrowid,)).fetchone()
            return _public(row), None
    except sqlite3.IntegrityError:
        return None, 'An account with that email already exists.'


def authenticate(email, password):
    with _connect() as conn:
        row = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    if row is None or not check_password_hash(row['password_hash'], password):
        return None
    return _public(row)


def get_user(user_id):
    if not user_id:
        return None
    with _connect() as conn:
        row = conn.execute('SELECT * FROM users WHERE id = ?', (int(user_id),)).fetchone()
    return _public(row)


def set_tier(user_id, tier, stripe_customer_id=None):
    if tier not in {'free', 'pro'}:
        raise ValueError('tier must be free or pro')
    with _connect() as conn:
        if stripe_customer_id:
            conn.execute(
                'UPDATE users SET tier = ?, stripe_customer_id = ? WHERE id = ?',
                (tier, stripe_customer_id, int(user_id)),
            )
        else:
            conn.execute('UPDATE users SET tier = ? WHERE id = ?', (tier, int(user_id)))
        conn.commit()
        row = conn.execute('SELECT * FROM users WHERE id = ?', (int(user_id),)).fetchone()
    return _public(row)


def save_alert(email, city_key, min_score, cadence='weekly'):
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with _connect() as conn:
        conn.execute(
            '''
            INSERT INTO alerts (email, city_key, min_score, cadence, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(email, city_key) DO UPDATE SET
                min_score = excluded.min_score,
                cadence = excluded.cadence,
                created_at = excluded.created_at
            ''',
            (email, city_key, float(min_score), cadence, now),
        )
        conn.commit()
    return {'city': city_key, 'min_score': float(min_score), 'cadence': cadence}


def record_event(name):
    day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    with _connect() as conn:
        conn.execute(
            '''
            INSERT INTO event_counts (name, day, hits) VALUES (?, ?, 1)
            ON CONFLICT(name, day) DO UPDATE SET hits = hits + 1
            ''',
            (name, day),
        )
        conn.commit()


def find_by_email(email):
    with _connect() as conn:
        row = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    return _public(row)
