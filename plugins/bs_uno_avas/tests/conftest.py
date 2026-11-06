import sys
from pathlib import Path

# the checkout's src/ (no install needed); an installed bs_uno_avas is shadowed
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
