"""Reference implementations per block, for catch-up cells and solutions.

    from claims_lab.reference import block1
    block1.agent(cl.load_package("PKG-002"))
"""

import functools
import sys


class _Watch:
    """Passes everything on to the real output and notes whether anything was printed."""
    def __init__(self, out):
        self.out, self.wrote = out, False

    def write(self, text):
        self.wrote = self.wrote or bool(text)
        return self.out.write(text)

    def __getattr__(self, name):
        return getattr(self.out, name)


def blank_line_after(fn):
    """A check helper prints its verdict; an empty line after it keeps the verdict apart from the output that follows."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        watch = sys.stdout = _Watch(sys.stdout)
        try:
            return fn(*args, **kwargs)
        finally:
            sys.stdout = watch.out
            if watch.wrote:
                print()
    return wrapper
