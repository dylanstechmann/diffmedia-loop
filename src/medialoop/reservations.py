"""Transactional local reservations and idempotent proposal recovery."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from medialoop.planner import _read, _propose
from medialoop.space import NAMES


def propose_and_reserve(candidates_path, observations_path=None, pending_path=None, *,
                        ledger_path, request_id, seed=0, noise=0.02):
    """Reserve before returning; retry a request ID to retrieve its saved result.

    All cooperating callers must use the same local SQLite file. Reservations
    persist even after completion, so a condition cannot be issued twice. Input
    CSVs are snapshots, not transactionally managed experimental records.
    """
    if (not isinstance(request_id, str) or not 1 <= len(request_id) <= 128
            or any(not 33 <= ord(char) <= 126 for char in request_id)):
        raise ValueError("request_id must contain 1-128 printable ASCII characters without spaces")
    ledger = Path(ledger_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    connection = None
    try:
        connection = sqlite3.connect(ledger, timeout=30, isolation_level=None)
        connection.execute("PRAGMA synchronous=FULL")
        # SQLite releases this lock on process exit; no stale lock-file removal
        # is required. Selection and insertion are serialized by one transaction.
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS reservations ("
                           "request_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL UNIQUE, "
                           "configuration TEXT NOT NULL, proposal TEXT NOT NULL)")
        rows, candidate_hash = _read(candidates_path, ["candidate_id", *NAMES])
        bound = connection.execute("SELECT value FROM metadata WHERE key='candidates_sha256'").fetchone()
        if bound is not None and bound[0] != candidate_hash:
            raise ValueError("candidate table differs from this ledger; keep its original candidate table")
        configuration = json.dumps({"seed": seed, "noise": noise}, sort_keys=True, allow_nan=False)
        saved = connection.execute("SELECT configuration, proposal FROM reservations WHERE request_id=?",
                                   (request_id,)).fetchone()
        if saved is not None:
            if saved[0] != configuration:
                raise ValueError("request_id already exists with different seed or noise")
            result = json.loads(saved[1])
        else:
            observations, observations_hash = _read(observations_path, ["candidate_id", "response"]) if observations_path else ([], None)
            pending, pending_hash = _read(pending_path, ["candidate_id"]) if pending_path else ([], None)
            hashes = {"candidates": candidate_hash, "observations": observations_hash, "pending": pending_hash}
            reserved = [row[0] for row in connection.execute("SELECT candidate_id FROM reservations")]
            result = _propose(rows, observations, pending, hashes, seed=seed, noise=noise, reserved_ids=reserved)
            result["reservation"] = {"request_id": request_id, "candidates_sha256": candidate_hash}
            result["note"] = ("Maximizes the supplied response. A proposal is not an executable protocol. "
                              "Reserved in the ledger; reuse this request ID to recover the original proposal.")
            if bound is None:
                connection.execute("INSERT INTO metadata VALUES ('candidates_sha256', ?)", (candidate_hash,))
            connection.execute("INSERT INTO reservations VALUES (?, ?, ?, ?)",
                               (request_id, result["candidate_id"], configuration,
                                json.dumps(result, allow_nan=False)))
        connection.commit()
        return result
    except sqlite3.Error as exc:
        raise ValueError(f"reservation ledger could not be updated: {exc}") from exc
    finally:
        if connection is not None:
            # An uncommitted insertion is rolled back, including on interruption.
            connection.close()
