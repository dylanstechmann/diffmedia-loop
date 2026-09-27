import contextlib
import io
import json
import multiprocessing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from medialoop.planner import main, propose
from medialoop.reservations import propose_and_reserve
from medialoop.space import NAMES


def reserve_worker(candidates, ledger, request_id, start, messages):
    messages.put(("ready", request_id))
    start.wait(15)
    try:
        result = propose_and_reserve(candidates, ledger_path=ledger, request_id=request_id)
        messages.put(("result", result))
    except Exception as exc:
        messages.put(("error", repr(exc)))


def interrupted_writer(ledger, ready):
    connection = sqlite3.connect(ledger)
    connection.execute("BEGIN IMMEDIATE")
    connection.execute("INSERT INTO reservations VALUES ('interrupted', 'bogus', '{}', '{}')")
    ready.set()
    # The parent terminates this process with an uncommitted write and held lock.
    multiprocessing.Event().wait(30)


class ReservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.candidates = self.root / "candidates.csv"
        self.candidates.write_text("candidate_id," + ",".join(NAMES) +
                                   "\na,0,0,0,0\nb,3,1,0,0\nc,6,2,0,0\nd,9,3,0,0\n")
        self.ledger = self.root / "reservations.sqlite3"

    def reserve(self, request_id, **kwargs):
        return propose_and_reserve(self.candidates, ledger_path=self.ledger,
                                   request_id=request_id, **kwargs)

    def test_repeated_requests_are_unique_and_preview_remains_read_only(self):
        before = self.candidates.read_bytes()
        preview = propose(self.candidates)
        self.assertFalse(self.ledger.exists())
        first = self.reserve("first")
        self.assertEqual(first["candidate_id"], preview["candidate_id"])
        results = [first] + [self.reserve(str(i)) for i in range(3)]
        self.assertEqual(len({result["candidate_id"] for result in results}), 4)
        with self.assertRaisesRegex(ValueError, "no unevaluated"):
            self.reserve("exhausted")
        self.assertEqual(self.candidates.read_bytes(), before)

    def test_retry_recovers_original_proposal_even_when_observations_change(self):
        observations = self.root / "observations.csv"
        observations.write_text("candidate_id,response\na,0.1\n")
        first = self.reserve("round-1", observations_path=observations)
        observations.unlink()
        self.assertEqual(first, self.reserve("round-1", observations_path=observations))
        with self.assertRaisesRegex(ValueError, "different seed or noise"):
            self.reserve("round-1", seed=1)

    def test_pending_completed_and_reserved_conditions_are_excluded(self):
        first = self.reserve("first")
        remaining = sorted({"a", "b", "c", "d"} - {first["candidate_id"]})
        observations = self.root / "observations.csv"
        observations.write_text(f"candidate_id,response\n{first['candidate_id']},0.2\n{remaining[0]},0.4\n")
        pending = self.root / "pending.csv"
        pending.write_text(f"candidate_id\n{remaining[1]}\n")
        result = self.reserve("second", observations_path=observations, pending_path=pending)
        self.assertEqual(result["candidate_id"], remaining[2])
        self.assertEqual(result["n_observed"], 2)
        self.assertEqual(result["n_pending"], 1)

    def test_candidate_table_is_bound_to_ledger(self):
        self.reserve("first")
        self.candidates.write_text(self.candidates.read_text().replace("d,9", "d,10"))
        for request_id in ("first", "second"):
            with self.assertRaisesRegex(ValueError, "candidate table differs"):
                self.reserve(request_id)

    def test_commit_failure_rolls_back_insert_and_request_id(self):
        self.reserve("initial")
        connect = sqlite3.connect

        class FailingCommit(sqlite3.Connection):
            def commit(self):
                raise sqlite3.OperationalError("simulated disk failure")

        def failing_connect(*args, **kwargs):
            return connect(*args, factory=FailingCommit, **kwargs)

        with patch("medialoop.reservations.sqlite3.connect", side_effect=failing_connect):
            with self.assertRaisesRegex(ValueError, "simulated disk failure"):
                self.reserve("retry")
        with connect(self.ledger) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM reservations").fetchone()[0], 1)
        self.reserve("retry", seed=2)

    def test_concurrent_processes_reserve_distinct_candidates(self):
        context = multiprocessing.get_context("spawn")
        start, messages = context.Event(), context.Queue()
        processes = [context.Process(target=reserve_worker,
                     args=(self.candidates, self.ledger, str(i), start, messages)) for i in range(4)]
        try:
            for process in processes:
                process.start()
            for _ in processes:
                self.assertEqual(messages.get(timeout=15)[0], "ready")
            start.set()
            replies = [messages.get(timeout=20) for _ in processes]
            self.assertTrue(all(kind == "result" for kind, _ in replies), replies)
            self.assertEqual(len({result["candidate_id"] for _, result in replies}), 4)
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
            messages.close()

    def test_terminated_writer_releases_lock_and_rolls_back(self):
        self.reserve("first")
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        process = context.Process(target=interrupted_writer, args=(self.ledger, ready))
        process.start()
        try:
            self.assertTrue(ready.wait(15))
        finally:
            process.terminate()
            process.join(timeout=5)
        second = self.reserve("second")
        self.assertIn(second["candidate_id"], {"a", "b", "c", "d"})
        with sqlite3.connect(self.ledger) as connection:
            self.assertIsNone(connection.execute("SELECT * FROM reservations WHERE request_id='interrupted'").fetchone())

    def test_cli_export_failure_keeps_recoverable_reservation(self):
        output = self.root / "proposal.json"
        output.write_text("existing unrelated output")
        args = ["--candidates", str(self.candidates), "--reserve-ledger", str(self.ledger),
                "--request-id", "round-1", "--out", str(output)]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args)
        output.unlink()
        self.assertEqual(main(args), 0)
        result = json.loads(output.read_text())
        self.assertEqual(result, self.reserve("round-1"))
        self.assertEqual(main(args), 0)
        with sqlite3.connect(self.ledger) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM reservations").fetchone()[0], 1)

    def test_cli_requires_request_key_and_preserves_preview_output(self):
        output = self.root / "preview.json"
        args = ["--candidates", str(self.candidates), "--out", str(output)]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args + ["--reserve-ledger", str(self.ledger)])
        self.assertFalse(self.ledger.exists())
        self.assertEqual(main(args), 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args)


if __name__ == "__main__":
    unittest.main()
