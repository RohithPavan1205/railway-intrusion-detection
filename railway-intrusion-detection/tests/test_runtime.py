import time
import unittest

from dashboard.runtime import ProcessingWorker


class FakeSystem:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.steps = 0

    def start_processing(self):
        self.started += 1

    def stop_processing(self):
        self.stopped += 1

    def run_step(self):
        self.steps += 1


class TestProcessingWorker(unittest.TestCase):
    def test_start_is_idempotent_and_stop_releases_once(self):
        system = FakeSystem()
        worker = ProcessingWorker(system)
        self.assertTrue(worker.start())
        self.assertFalse(worker.start())
        time.sleep(0.03)
        self.assertGreater(system.steps, 0)
        self.assertTrue(worker.stop())
        self.assertFalse(worker.is_running)
        self.assertEqual(system.started, 1)
        self.assertEqual(system.stopped, 1)
