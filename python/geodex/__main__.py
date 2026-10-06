"""Print the geodex version and the components this install provides.

Run it with ``python -m geodex``.
"""

import geodex
from geodex import _geodex_core

COMPONENTS = (
    ("planning", "plan"),
    ("built-in robots", "robots"),
    ("collision checking", "Scene"),
)


def main() -> None:
    print("geodex", geodex.__version__)
    for label, name in COMPONENTS:
        print(f"{label}: {'yes' if hasattr(_geodex_core, name) else 'no'}")


if __name__ == "__main__":
    main()
