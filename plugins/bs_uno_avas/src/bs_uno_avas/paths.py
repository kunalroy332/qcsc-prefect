"""Where relative paths go: the workspace (inputs, output, chkfiles, logs) and the package data.

- :func:`workspace`: ``$BS_UNO_AVAS_WORKSPACE`` if set, else ``<plugin>/workspace`` of a checkout
  (``plugins/bs_uno_avas/workspace``, editable install or ``PYTHONPATH=src``), else the current
  directory (wheel install). Relative config paths (``[structure] pdb``, ``[output] dir`` /
  ``work_dir``) resolve against it unless the config sets ``base_dir``.
- :func:`package_data`: files shipped with the package (``data/AOs``: the ``file:AOs`` targets).
"""

import os
from importlib import resources
from pathlib import Path

ENV = "BS_UNO_AVAS_WORKSPACE"


def workspace():
    env = os.environ.get(ENV)
    if env:
        return Path(env).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2] / "workspace"
    return checkout if checkout.is_dir() else Path.cwd()


def package_data(*parts):
    return Path(str(resources.files("bs_uno_avas").joinpath("data", *parts)))


def ao_dir(spec, base_dir):
    """Directory of ``target_aos = "file:<spec>"``: absolute as given; relative from the package
    data (``file:AOs`` = the shipped dzvp targets), else from ``base_dir``."""
    p = Path(spec).expanduser()
    if p.is_absolute():
        return p
    shipped = package_data(spec)
    return shipped if shipped.is_dir() else Path(base_dir) / p
