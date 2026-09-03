"""Detection of Totem's optional dependencies.

Several tests exercise how the runtime degrades when an optional dependency
is not installed. Those tests can only assert a degradation path on a machine
where the dependency is genuinely missing, so they are skipped elsewhere
rather than asserting absence — otherwise the suite fails on any machine that
happens to have the package installed.
"""

import importlib.util

import pytest


def module_available(name):
    """Return True when ``name`` can be imported without importing it."""

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # A namespace package shadowing the name, or a parent package that
        # cannot be imported, both mean the dependency is unusable here.
        return False


HAS_CRYPTOGRAPHY = module_available("cryptography")
HAS_NETWORKX = module_available("networkx")
HAS_MATPLOTLIB = module_available("matplotlib")
HAS_PYDOT = module_available("pydot")

# visualize_graph needs both networkx and matplotlib, so it only degrades when
# at least one of them is missing.
HAS_VISUALIZATION = HAS_NETWORKX and HAS_MATPLOTLIB

requires_no_cryptography = pytest.mark.skipif(
    HAS_CRYPTOGRAPHY,
    reason="cryptography is installed; its absence cannot be exercised here",
)

requires_no_visualization = pytest.mark.skipif(
    HAS_VISUALIZATION,
    reason="networkx and matplotlib are installed; "
    "the missing-dependency path cannot be exercised here",
)

requires_no_pydot = pytest.mark.skipif(
    HAS_PYDOT,
    reason="pydot is installed; its absence cannot be exercised here",
)
