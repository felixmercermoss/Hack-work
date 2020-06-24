import time

class Timer:
    def __init__(self, message):
        self.message = message
        self.start = None

    def __enter__(self):
        print(self.message + "...", end="")
        self.start = time.perf_counter()

    def __exit__(self, type, value, traceback):
        duration = time.perf_counter() - self.start
        print(" took {0:.3f} seconds".format(duration))
