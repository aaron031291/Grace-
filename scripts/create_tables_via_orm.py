"""Create all ORM tables into the dev SQLite DB using SQLAlchemy.

This imports `grace.db.models_phase_a|b|c` and calls Base.metadata.create_all()
against the SQLite path (GRACE_SQLITE_PATH) so devs can quickly provision the DB.
"""
from __future__ import annotations

import os
from sqlalchemy import create_engine


def main():
    sqlite_path = os.environ.get("GRACE_SQLITE_PATH", "./grace_dev.sqlite3")
    engine = create_engine(f"sqlite:///{sqlite_path}")

    # Import the model modules so their Base classes are registered
    # We expect each models_phase_*.py to expose a Base variable.
    from grace.db import models_phase_a as a
    from grace.db import models_phase_b as b
    from grace.db import models_phase_c as c

    # For simplicity, call create_all on each Base
    a.Base.metadata.create_all(engine)
    b.Base.metadata.create_all(engine)
    c.Base.metadata.create_all(engine)

    print("Created ORM tables in", sqlite_path)


if __name__ == '__main__':
    main()
