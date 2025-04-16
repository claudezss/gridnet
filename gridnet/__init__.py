from pathlib import Path

ROOT = Path(__file__).parent

VERSION = (ROOT / "VERSION").read_text().strip()
