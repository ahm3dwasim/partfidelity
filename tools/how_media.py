"""CLI for the /how-it-works frame provisioning; the logic lives in app/how_media.py (it ships with the app)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.how_media import main  # noqa: E402

if __name__ == "__main__":
    main()
