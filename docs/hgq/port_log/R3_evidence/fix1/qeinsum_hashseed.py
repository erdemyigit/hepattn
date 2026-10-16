"""HGQ2 0.1.9 QEinsum expands '...' from a Python set: print the serialised equation (vary PYTHONHASHSEED)."""

import os

from hgq.layers import QEinsum

layer = QEinsum("...d,...d->...d")
layer.build([(None, 21, 64), (None, 21, 64)])
print(os.environ.get("PYTHONHASHSEED"), layer.get_config()["equation"])
