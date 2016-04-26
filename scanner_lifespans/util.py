import re
import pathlib
from concurrent import futures
import multiprocessing

ROW_NAMES_384 = 'ABCDEFGHIJKLMNOP'
COL_NAMES_384 = ['{:02d}'.format(i) for i in range(1, 25)]

_well_regex = re.compile('[A-P][0-9][0-9]?')
def split_image_name(image_path):
    image_path = pathlib.Path(image_path)
    well_match = _well_regex.match(image_path.stem)
    well = well_match.group()
    rest = image_path.stem[well_match.end():]
    if rest.startswith('_'):
        rest = rest[1:]
    return well, rest

class BackgroundRunner:
    """Class for running jobs in background processes.Use submit() to add jobs,
    and then wait() to get the results for each submitted job:
    either the return value on successful completion, or an exception object.

    Wait() will only wait until the first exception, and after that will
    attempt to cancel all pending jobs.

    If max_workers is 1, then just run the job in this process. Useful
    for debugging, where a proper traceback from a foreground exception
    can be helpful.
    """
    def __init__(self, max_workers=None):
        if max_workers is None:
            max_workers = multiprocessing.cpu_count() - 1
        if max_workers == 1:
            self.executor = None
        else:
            self.executor = futures.ProcessPoolExecutor(max_workers)
        self.futures = []

    def submit(self, fn, *args, **kwargs):
        if self.executor:
            self.futures.append(self.executor.submit(fn, *args, **kwargs))
        else:
            fn(*args, **kwargs)

    def wait(self):
        if not self.executor:
            return []
        try:
            futures.wait(self.futures, return_when=futures.FIRST_EXCEPTION)
        except KeyboardInterrupt:
            for future in self.futures:
                future.cancel()
                raise

        # If there was an exception, cancel all the rest of the jobs.
        # If there was no exception, can "cancel" the jobs anyway, because canceling does
        # nothing if the job is done.
        for future in self.futures:
            future.cancel()
        results = []
        error_indices = []
        cancelled_indices = []
        for i, future in enumerate(self.futures):
            if future.cancelled():
                results.append(futures.CancelledError())
                cancelled_indices.append(i)
            else:
                exc = future.exception()
                if exc is not None:
                    results.append(exc)
                    error_indices.append(i)
                else:
                    results.append(future.result())
        self.futures = []
        was_error = len(error_indices) > 0 or len(cancelled_indices) > 0
        return results, was_error, error_indices, cancelled_indices
