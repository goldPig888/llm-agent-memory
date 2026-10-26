"""LIBERO worker selection for the richer harness; baseline worker is unchanged."""
from pathlib import Path
from inspect_robots_libero import embodiment as module


class HarnessEmbodiment(module.LiberoEmbodiment):
    def _start_worker(self):
        original = module.WORKER
        try:
            module.WORKER = Path(__file__).with_name('worker.py')
            super()._start_worker()
        finally:
            module.WORKER = original
