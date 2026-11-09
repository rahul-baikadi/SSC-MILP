"""Offline test of the verifier pipeline with a mock Claude client."""
import json
from ssc import verify as V
from ssc.anthropic_client import AnthropicClient
from ssc.prompts_verify import SYSTEM_FULL, verifier_blocks


def test_parse_verdict():
    r = V.parse_verdict('blah {"verdict": "incorrect", "p_correct": 0.2, "errors": ["x"], "assumptions": [], "reason": "r",}')
    assert r["parsed"] and r["accept"] is False and r["p_correct"] == 0.2
    assert V.parse_verdict("no json")["parsed"] is False


def test_prompt_has_no_gold_and_shows_solution():
    b = verifier_blocks("Q?", "code", {"status": "optimal", "objective": 17.4, "vars": {"x[Chicken]": 4.6}})
    txt = SYSTEM_FULL + "".join(x["text"] for x in b)
    assert "x[Chicken] = 4.6" in txt and "gold" not in txt.lower() and b[0]["cache"] is True


def test_mock_client_cache(tmp_path):
    mock = lambda s, b, t: '{"verdict": "correct", "p_correct": 0.9, "errors": [], "assumptions": [], "reason": "ok"}'
    c = AnthropicClient(model="mock", cache_dir=tmp_path, mock=mock)
    r1 = c.chat("s", [{"text": "a"}], tag="t"); r2 = c.chat("s", [{"text": "a"}], tag="t")
    assert not r1["cached"] and r2["cached"] and V.parse_verdict(r2["content"])["accept"]
