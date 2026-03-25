"""Tests for working memory."""
from corza_agents import WorkingMemory


def test_store_and_get():
    wm = WorkingMemory(session_id="s1")
    wm.store("key1", "value1")
    assert wm.get("key1") == "value1"
    assert wm.get("missing") is None
    assert wm.get("missing", "default") == "default"


def test_has_and_remove():
    wm = WorkingMemory()
    wm.store("k", "v")
    assert wm.has("k")
    wm.remove("k")
    assert not wm.has("k")


def test_findings():
    wm = WorkingMemory()
    wm.add_finding("Found anomaly", category="anomaly", evidence={"table": "orders"})
    wm.add_finding("Normal range", category="normal")

    assert len(wm.findings) == 2
    assert len(wm.findings_by_category("anomaly")) == 1


def test_artifacts():
    wm = WorkingMemory()
    wm.set_artifact("report", "Report content")
    assert wm.get_artifact("report") == "Report content"
    assert wm.get_artifact("missing") is None


def test_counters():
    wm = WorkingMemory()
    assert wm.get_count("calls") == 0
    wm.increment("calls")
    wm.increment("calls", 2)
    assert wm.get_count("calls") == 3


def test_tags():
    wm = WorkingMemory()
    wm.tag("urgent", "review")
    assert wm.has_tag("urgent")
    assert not wm.has_tag("done")
    assert wm.tags == {"urgent", "review"}


def test_context_for_llm():
    wm = WorkingMemory()
    wm.add_finding("Found something", category="test")
    wm.store("data_key", {"rows": 100})
    wm.tag("active")

    context = wm.get_context_for_llm()
    assert "Found something" in context
    assert "data_key" in context
    assert "active" in context


def test_context_for_llm_truncation():
    wm = WorkingMemory()
    wm.store("big_key", "x" * 10000)
    context = wm.get_context_for_llm(max_chars=500)
    assert len(context) <= 500


def test_clear():
    wm = WorkingMemory()
    wm.store("k", "v")
    wm.add_finding("f")
    wm.set_artifact("a", "v")
    wm.increment("c")
    wm.tag("t")
    wm.clear()
    assert wm.keys() == []
    assert wm.findings == []
    assert wm.artifacts == {}
    assert wm.get_count("c") == 0
    assert wm.tags == set()
