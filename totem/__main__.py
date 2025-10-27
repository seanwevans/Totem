"""Module entry-point for ``python -m totem``."""

from .runtime import main


def _run() -> None:
    import sys

    main(sys.argv[1:])


if __name__ == "__main__":
    _run()
