"""python -X utf8 -m pytest tests -q   (or: python -X utf8 tests/test_basic.py)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lifemate.agent import Agent, extract_json  # noqa: E402
from lifemate.data import catalog_index, load_catalog, load_samples, load_zaic_tools, data_root  # noqa: E402
from lifemate.scorer import calls_match, score_sample, token_f1  # noqa: E402

FZ = {"sms.send": {"content"}, "smart_home.light_set": set()}


def test_token_f1():
    assert token_f1("Mai chạy bộ 9h", "mai chạy bộ 9h") == 1.0
    assert token_f1("abc", "xyz") == 0.0


def test_match_rules():
    g = {"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "nhớ mang hộ cái sạc nhé"}}
    assert calls_match({"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "nhớ mang hộ cái sạc"}}, g, FZ)
    assert not calls_match({"name": "sms.send", "arguments": {"contact_id": "c_02", "content": "nhớ mang hộ cái sạc nhé"}}, g, FZ)
    assert not calls_match({"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "gọi lại"}}, g, FZ)
    light = {"name": "smart_home.light_set", "arguments": {"device_id": "l1", "power": "off"}}
    assert not calls_match({"name": "smart_home.light_set", "arguments": {"device_id": "l1", "power": "off", "brightness": 0}}, light, FZ)


def test_score_extra_read_costs_precision():
    g = [{"name": "smart_home.get_status", "arguments": {}}, {"name": "smart_home.light_set", "arguments": {"device_id": "l1", "power": "off"}}]
    s = g + [{"name": "weather.get_current", "arguments": {}}]
    r = score_sample(s, "h", g, "h", FZ)
    assert abs(r["score"] - 0.8) < 1e-9  # P=2/3, R=1
    assert score_sample(s, "x", g, "h", FZ)["score"] == 0.0
    assert score_sample([], "h", [], "h", FZ)["score"] == 1.0
    assert score_sample(g, "h", [], "h", FZ)["score"] == 0.0


def test_extract_json():
    assert extract_json('<think>x {"a":1}</think>```json\n{"name": "done"}\n```') == {"name": "done"}
    assert extract_json('call: {"name": "a.b", "arguments": {"q": "}"}} trailing') == {"name": "a.b", "arguments": {"q": "}"}}
    assert extract_json("không có json") is None


def test_agent_never_executes_invalid_calls():
    zt = load_zaic_tools("train")
    s = next(x for x in load_samples("train") if "smart_home" in x["context"]["installed_apps"])
    tools, apps = catalog_index(load_catalog("train"))
    env = zt.ToolEnv("train", s["id"], root=data_root())
    a = Agent(s, env, tools, apps, s["audio_text"], [], max_generations=8)
    a.feed("xin chào")                                                    # not JSON -> feedback only
    a.feed('{"name": "smart_home.light_set", "arguments": {"power": "off"}}')  # missing device_id -> feedback only
    a.feed('{"name": "banking_fake.pay", "arguments": {}}')              # unknown tool -> feedback only
    assert env.get_tool_calls() == []
    a.feed('{"name": "smart_home.light_set", "arguments": {"device_id": "light_made_up", "power": "off"}}')  # invented ID
    assert env.get_tool_calls() == []
    a.feed('{"name": "smart_home.get_status", "arguments": {}}')
    a.feed('{"name": "smart_home.get_status", "arguments": {}}')         # duplicate read -> cached
    assert [c["name"] for c in env.get_tool_calls()] == ["smart_home.get_status"]
    a.feed('{"name": "done"}')
    assert a.done


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
