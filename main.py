"""Run without installing the commands:  python main.py <command> [options]

    python main.py calibrate region|lives|score|digits|scale|motion|preview
    python main.py aim [--stop-at N] [--max-shots N]
    python main.py record
    python main.py fit [--plot]
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

MODULES = {"calibrate": "calibrate", "aim": "aim", "record": "record", "fit": "fit"}

if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
    raise SystemExit(__doc__)
module = MODULES[sys.argv[1]]
sys.argv = [f"main.py {sys.argv[1]}"] + sys.argv[2:]
runpy.run_module(f"hoops_bot.{module}", run_name="__main__")
