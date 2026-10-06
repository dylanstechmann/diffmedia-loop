"""Transactional local reservations and idempotent proposal recovery."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from medialoop.constraints import enforce_constraints
from medialoop.planner import _read, _propose
from medialoop.space import NAMES


def propose_and_reserve(candidates_path, observations_path=None, pending_path=None, *,
                        ledger_path, request_id, seed=0, noise=0.02, constraints_path=None,
                        batch_size=1, batch_strategy="kriging_believer"):
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
        existing = connection.execute("PRAGMA table_info(reservations)").fetchall()
        if existing:
            primary_key_positions = {row[1]: row[5] for row in existing if row[5]}
            if primary_key_positions == {"request_id": 1}:
                legacy_exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='reservations_legacy'"
                ).fetchone()
                if legacy_exists:
                    raise ValueError("reservation ledger has conflicting legacy and current tables")
                connection.execute("ALTER TABLE reservations RENAME TO reservations_legacy")
        connection.execute("CREATE TABLE IF NOT EXISTS reservations ("
                           "request_id TEXT NOT NULL, candidate_id TEXT NOT NULL UNIQUE, "
                           "configuration TEXT NOT NULL, proposal TEXT NOT NULL, "
                           "PRIMARY KEY (request_id, candidate_id))")
        legacy_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='reservations_legacy'"
        ).fetchone()
        if legacy_exists:
            legacy_rows = connection.execute(
                "SELECT request_id, candidate_id, configuration, proposal FROM reservations_legacy"
            ).fetchall()
            for old_request_id, candidate_id, old_configuration, proposal in legacy_rows:
                saved_configuration = json.loads(old_configuration)
                saved_configuration.setdefault("batch_size", 1)
                saved_configuration.setdefault("batch_strategy", "kriging_believer")
                connection.execute(
                    "INSERT OR IGNORE INTO reservations VALUES (?, ?, ?, ?)",
                    (old_request_id, candidate_id,
                     json.dumps(saved_configuration, sort_keys=True, allow_nan=False), proposal),
                )
        rows, candidate_hash = _read(candidates_path, ["candidate_id", *NAMES])
        constraints_provenance = enforce_constraints(constraints_path, rows) if constraints_path else None
        bound = connection.execute("SELECT value FROM metadata WHERE key='candidates_sha256'").fetchone()
        if bound is not None and bound[0] != candidate_hash:
            raise ValueError("candidate table differs from this ledger; keep its original candidate table")
        configuration = json.dumps({"seed": seed, "noise": noise, "batch_size": batch_size,
                                    "batch_strategy": batch_strategy}, sort_keys=True, allow_nan=False)
        saved = connection.execute("SELECT configuration, proposal FROM reservations WHERE request_id=? LIMIT 1",
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
            result = _propose(rows, observations, pending, hashes, seed=seed, noise=noise,
                              reserved_ids=reserved, observations_path=observations_path,
                              batch_size=batch_size, batch_strategy=batch_strategy)
            if constraints_provenance is not None:
                result["constraints"] = constraints_provenance
            candidate_ids = result.get("candidates", [result["candidate_id"]])
            result["reservation"] = {"request_id": request_id, "candidate_ids": candidate_ids,
                                     "candidates_sha256": candidate_hash}
            result["note"] = ("Maximizes the supplied response. A proposal is not an executable protocol. "
                              "All listed conditions were reserved atomically; reuse this request ID and settings "
                              "to recover the original proposal.")
            if bound is None:
                connection.execute("INSERT INTO metadata VALUES ('candidates_sha256', ?)", (candidate_hash,))
            proposal = json.dumps(result, allow_nan=False)
            connection.executemany("INSERT INTO reservations VALUES (?, ?, ?, ?)",
                                   [(request_id, candidate_id, configuration, proposal)
                                    for candidate_id in candidate_ids])
        connection.commit()
        return result
    except sqlite3.Error as exc:
        raise ValueError(f"reservation ledger could not be updated: {exc}") from exc
    finally:
        if connection is not None:
            # An uncommitted insertion is rolled back, including on interruption.
            connection.close()
