"""Print the sandbox worker's own error for a known-good model (Windows debugging)."""
import os, platform, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssc.sandbox import _run_worker, parse_candidate

code = (Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ducks_ok.py").read_text(encoding="utf-8")
print("python:", sys.executable, platform.python_version(), "os:", os.name)
res = _run_worker("sandbox_worker", code, 60)
print("worker status:", res.get("_status"), "returncode:", res.get("returncode"))
print("defects:", res.get("defects"))
print("worker error:\n", res.get("error") or res.get("stderr"))
print("parse_candidate defects:", parse_candidate(code)["defects"])
