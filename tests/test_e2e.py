"""Offline end-to-end test: mock LLM -> generate -> extract -> score -> SCIP
evaluate -> report.  Run:  python -m pytest tests -q   (or python tests/test_e2e.py)"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ssc.evaluate import evaluate_benchmark, score_benchmark  # noqa: E402
from ssc.extract import extract_benchmark  # noqa: E402
from ssc.generate import generate_benchmark  # noqa: E402
from ssc.judge import judge_benchmark  # noqa: E402
from ssc.llm import LLMClient  # noqa: E402
from ssc.metrics import format_report, load_results, summarize  # noqa: E402
from tests.mock_llm import PROBLEMS, mock_llm  # noqa: E402


def test_pipeline(tmp_path=None):
    out = Path(tmp_path or ROOT / "runs" / "_toy")
    if out.exists():
        shutil.rmtree(out)
    llm = LLMClient(model="mock", cache_dir=out / "cache", mock=mock_llm)
    generate_benchmark(llm, PROBLEMS, out, problem_workers=2, recipe_workers=4, log=print)
    extract_benchmark(llm, PROBLEMS, out, log=print)
    score_benchmark(PROBLEMS, out, run_lp=True, log=print)
    evaluate_benchmark(PROBLEMS, out, log=print)
    judge_benchmark(llm, PROBLEMS, out, log=print)
    rows = load_results(out, PROBLEMS)
    summ = summarize(rows)
    print(format_report("toy", summ))
    assert summ["n"] == 2
    assert summ["coverage"] == 1.0
    assert summ["accuracy"]["SSC"] == 1.0, summ["accuracy"]
    # requirement filtering dropped the out-of-vocabulary token
    reqs = json.load(open(out / "requirements" / "toy_0001.json"))["requirements"]
    assert all(all(t.split(":")[0] in ("objective", "var", "constraint") for t in r["features"]) for r in reqs)
    assert len(reqs) == 4
    pool = json.load(open(out / "pools" / "toy_0001.json"))
    assert len(pool["candidates"]) == 10
    assert sorted(c["position"] for c in pool["candidates"]) == list(range(10))
    ev = json.load(open(out / "eval" / "toy_0001.json"))
    n_correct = sum(c["correct"] for c in ev["candidates"])
    assert 4 <= n_correct <= 9, [c["objective"] for c in ev["candidates"]]
    sc = json.load(open(out / "scores" / "toy_0001.json"))
    sel = sc["selected"]["SSC"]
    assert ev["candidates"][sel]["correct"]
    print("E2E OK")


if __name__ == "__main__":
    test_pipeline()
