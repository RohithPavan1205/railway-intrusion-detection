"""Non-Streamlit worker lifecycle helpers."""

import threading
import time


class ProcessingWorker:
    def __init__(self, system):
        self.system = system
        self._stop = threading.Event()
        self._thread = None

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.is_running:
            return False
        self._stop.clear()
        self.system.start_processing()
        self._thread = threading.Thread(target=self._run, name="railway-processing", daemon=True)
        self._thread.start()
        return True

    def _run(self):
        while not self._stop.is_set():
            self.system.run_step()
            self._stop.wait(0.01)

    def stop(self):
        if not self.is_running:
            self.system.stop_processing()
            return False
        self._stop.set()
        self._thread.join(timeout=3)
        self.system.stop_processing()
        self._thread = None
        return True
