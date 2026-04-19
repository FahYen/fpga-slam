"""Two-process round-trip with the zero-copy acquire/release API.

The producer process writes frames into shm; the consumer process attaches
in a separate process, uses ``Consumer.frame()`` (acquire/release context
manager), and pushes per-frame metadata + checksums back via a queue for
the test to assert.
"""

import multiprocessing as mp
import os
import sys
import time
import unittest
import uuid

import numpy as np

# Allow running directly without the cmake-set PYTHONPATH.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PYDIR = os.path.normpath(os.path.join(_HERE, "..", "python"))
if _PYDIR not in sys.path:
    sys.path.insert(0, _PYDIR)

from rnsg_ipc import (  # noqa: E402
    Consumer,
    Producer,
    RingBusy,
    RingTimeout,
    FLAG_RAW_SEMANTICKITTI_LABELS,
    unlink,
)


def _consumer_proc(name: str, expected_frames: int, q: mp.Queue, ready) -> None:
    try:
        cons = Consumer.open(name)
    except Exception as exc:  # pragma: no cover - surfaced via queue
        q.put(("error", repr(exc)))
        ready.set()
        return

    # Signal "attached" only after open. The consumer's consumer_next
    # initializes from head_seq at open time, so we must block the producer
    # until we've attached, otherwise we'd silently skip early publishes.
    ready.set()

    try:
        for _ in range(expected_frames):
            with cons.frame(timeout_s=5.0) as frame:
                pts_writable = frame.points.flags.writeable
                lbl_writable = frame.labels.flags.writeable
                q.put(
                    (
                        "frame",
                        frame.frame_id,
                        frame.consumed_index,
                        frame.capture_ns,
                        frame.flags,
                        frame.skipped_before,
                        frame.points.shape,
                        frame.labels.shape,
                        int(frame.labels.sum()),
                        float(frame.points.sum()),
                        bool(pts_writable),
                        bool(lbl_writable),
                    )
                )
        q.put(("drop_count", cons.drop_count))
    except Exception as exc:
        q.put(("error", repr(exc)))
    finally:
        cons.close()


class TwoProcessRoundTrip(unittest.TestCase):
    def test_inorder_delivery_zero_copy(self) -> None:
        name = f"/rnsg_pytest_{uuid.uuid4().hex[:8]}"
        unlink(name)

        slot_count = 4
        capacity = 1024
        n_frames = 6

        producer = Producer.create(name, slot_count=slot_count, capacity_points=capacity)
        try:
            ctx = mp.get_context("spawn")
            q: mp.Queue = ctx.Queue()
            ready = ctx.Event()
            cp = ctx.Process(target=_consumer_proc, args=(name, n_frames, q, ready))
            cp.start()

            self.assertTrue(ready.wait(timeout=10.0),
                            "consumer process never signaled attached")

            for f in range(n_frames):
                view = producer.lease()
                n = 256 + f
                view.points[:n, 0] = np.arange(n, dtype=np.float32) + 1000.0 * f
                view.points[:n, 1] = 0.5
                view.points[:n, 2] = 0.25
                view.points[:n, 3] = 0.125
                view.labels[:n] = (np.arange(n, dtype=np.int32) % 19) + (f * 100)
                producer.publish(
                    num_points=n,
                    capture_ns=10_000 + f,
                    frame_id=42 + f,
                    flags=FLAG_RAW_SEMANTICKITTI_LABELS,
                )
                # Pace slightly so the consumer can keep up with this small ring.
                time.sleep(0.02)

            cp.join(timeout=10)
            self.assertFalse(cp.is_alive(), "consumer did not finish in time")

            results = []
            while not q.empty():
                results.append(q.get_nowait())

            errors = [r for r in results if r[0] == "error"]
            self.assertFalse(errors, f"consumer errors: {errors}")

            frames = [r for r in results if r[0] == "frame"]
            self.assertEqual(len(frames), n_frames)
            for f, rec in enumerate(frames):
                (
                    _, frame_id, consumed_index, capture_ns, flags, skipped,
                    pts_shape, lbl_shape, lbl_sum, _pts_sum,
                    pts_writable, lbl_writable,
                ) = rec
                self.assertEqual(frame_id, 42 + f)
                # consumed_index is the dense per-consumer counter starting at 0.
                self.assertEqual(consumed_index, f)
                self.assertEqual(capture_ns, 10_000 + f)
                self.assertEqual(flags & FLAG_RAW_SEMANTICKITTI_LABELS,
                                 FLAG_RAW_SEMANTICKITTI_LABELS)
                self.assertEqual(skipped, 0)
                n = 256 + f
                self.assertEqual(pts_shape, (n, 4))
                self.assertEqual(lbl_shape, (n,))
                expected_sum = int(((np.arange(n) % 19) + (f * 100)).sum())
                self.assertEqual(lbl_sum, expected_sum)
                # Zero-copy views are exposed as read-only on the consumer.
                self.assertFalse(pts_writable)
                self.assertFalse(lbl_writable)

            drops = [r for r in results if r[0] == "drop_count"]
            self.assertEqual(drops, [("drop_count", 0)])
        finally:
            producer.close()
            unlink(name)

    def test_zero_copy_aliases_shm(self) -> None:
        """In-process: consumer sees producer's writes through the same memory."""
        name = f"/rnsg_pytest_{uuid.uuid4().hex[:8]}"
        unlink(name)
        prod = Producer.create(name, slot_count=4, capacity_points=64)
        try:
            cons = Consumer.open(name)
            try:
                view = prod.lease()
                view.points[0, 0] = 1.5
                view.labels[0] = 7
                prod.publish(num_points=1, frame_id=0)

                with cons.frame(timeout_s=1.0) as f1:
                    self.assertEqual(f1.frame_id, 0)
                    self.assertEqual(f1.points.shape, (1, 4))
                    self.assertEqual(f1.labels.shape, (1,))
                    self.assertEqual(float(f1.points[0, 0]), 1.5)
                    self.assertEqual(int(f1.labels[0]), 7)
                    # Writes through the read-only view must raise.
                    with self.assertRaises(ValueError):
                        f1.points[0, 0] = 99.0

                # After release, slot is available for producer reuse.
                v2 = prod.lease()
                v2.points[0, 0] = 9.0
                v2.labels[0] = 13
                prod.publish(num_points=1, frame_id=1)

                with cons.frame(timeout_s=1.0) as f2:
                    self.assertEqual(f2.frame_id, 1)
                    self.assertEqual(float(f2.points[0, 0]), 9.0)
                    self.assertEqual(int(f2.labels[0]), 13)
            finally:
                cons.close()
        finally:
            prod.close()
            unlink(name)

    def test_busy_errors(self) -> None:
        name = f"/rnsg_pytest_{uuid.uuid4().hex[:8]}"
        unlink(name)
        prod = Producer.create(name, slot_count=4, capacity_points=8)
        try:
            cons = Consumer.open(name)
            try:
                # release without acquire -> BUSY
                with self.assertRaises(RingBusy):
                    cons.release()

                v = prod.lease()
                v.labels[0] = 1
                prod.publish(num_points=1, frame_id=0)

                f1 = cons.acquire(timeout_s=1.0)
                self.assertEqual(f1.frame_id, 0)
                # acquire while holding -> BUSY
                with self.assertRaises(RingBusy):
                    cons.acquire(timeout_s=0)
                cons.release()
            finally:
                cons.close()
        finally:
            prod.close()
            unlink(name)

    def test_consumed_index_dense_under_overrun(self) -> None:
        """consumed_index must increment by 1 per acquire even when frames drop.

        This is the structural counter SG-SLAM uses for poses_vec_, GTSAM
        keys, and per-frame artifact filenames. It must NOT have gaps even
        when frame_id has gaps from drop-oldest.
        """
        name = f"/rnsg_pytest_{uuid.uuid4().hex[:8]}"
        unlink(name)
        # slot_count=4 so 10 publishes overflow into 6 dropped frames.
        prod = Producer.create(name, slot_count=4, capacity_points=8)
        try:
            cons = Consumer.open(name)
            try:
                for f in range(10):
                    v = prod.lease()
                    v.labels[0] = f
                    prod.publish(num_points=1, frame_id=f)

                # Drain whatever survived. With slot_count=4 and no consumer
                # active, exactly 4 frames survive: frame_ids 6, 7, 8, 9.
                seen = []
                for _ in range(4):
                    with cons.frame(timeout_s=0.5) as frame:
                        seen.append(
                            (frame.frame_id, frame.consumed_index, frame.skipped_before)
                        )

                # frame_id is sparse: jumps from nothing to 6, then 7,8,9.
                self.assertEqual([s[0] for s in seen], [6, 7, 8, 9])
                # consumed_index is dense: 0, 1, 2, 3 regardless of drops.
                self.assertEqual([s[1] for s in seen], [0, 1, 2, 3])
                # skipped_before reports the gap once, on the first acquire.
                self.assertEqual([s[2] for s in seen], [6, 0, 0, 0])
                self.assertEqual(cons.drop_count, 6)
            finally:
                cons.close()
        finally:
            prod.close()
            unlink(name)

    def test_timeout_when_idle(self) -> None:
        name = f"/rnsg_pytest_{uuid.uuid4().hex[:8]}"
        unlink(name)
        prod = Producer.create(name, slot_count=4, capacity_points=64)
        try:
            cons = Consumer.open(name)
            try:
                t0 = time.monotonic()
                with self.assertRaises(RingTimeout):
                    cons.acquire(timeout_s=0.1)
                elapsed = time.monotonic() - t0
                self.assertGreaterEqual(elapsed, 0.08)
                self.assertLess(elapsed, 1.0)
            finally:
                cons.close()
        finally:
            prod.close()
            unlink(name)


if __name__ == "__main__":
    unittest.main()
