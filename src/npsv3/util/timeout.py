# Adapted from: https://gist.github.com/TySkby/143190ad1b88c6115597c45f996b030c

import errno
import os
import signal

DEFAULT_TIMEOUT_MESSAGE = os.strerror(errno.ETIME)

class Timeout:
    def __init__(self, seconds: int, timeout_message=DEFAULT_TIMEOUT_MESSAGE):
        self.seconds = seconds
        self.timeout_message = timeout_message

    def _handle_timeout(self, signum, frame):
        raise TimeoutError("Timeout occurred")

    def __enter__(self):
        if self.seconds > 0:
            signal.signal(signal.SIGALRM, self._handle_timeout)
            signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.seconds > 0:
            signal.alarm(0)
