"""Compatibility entrypoint; the public evaluator uses the online output contract."""
import os
import sys
from pathlib import Path
os.execv(sys.executable,[sys.executable,str(Path(__file__).parent/'train/evaluate.py'),*sys.argv[1:]])
