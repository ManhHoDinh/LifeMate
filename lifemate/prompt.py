"""Prompt rendering: compact tool schemas, user context, conversation history.

The JSON schemas of a sample's tools are ~12K chars on median (24K max); rendered as one-line signatures they take
roughly a third of that, which keeps the whole conversation inside the 16K-token limit.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
HISTORY_RESULT_CHARS = 500
TOOL_RESULT_CHARS = 4000

SYSTEM_RULES = """Bạn là trợ lý giọng nói LifeMate. Người dùng nói một yêu cầu tiếng Việt; bạn thực hiện nó bằng cách gọi các tool của những app đã cài, từng tool một.

Quy tắc:
1. Mỗi lượt chỉ trả về MỘT đối tượng JSON, không viết gì thêm: {"name": "<app.tool>", "arguments": {...}}. Khi đã xong (hoặc không cần gọi tool nào) trả về {"name": "done"}.
2. ID (device_id, contact_id, bill_id, event_id...) chỉ được lấy từ kết quả tool đã gọi; muốn có ID thì gọi tool đọc tương ứng trước (get_status, list_..., search...). Không bịa ID.
3. Không gọi tool thừa: không gọi lại tool đọc đã có kết quả, không gọi tool đọc khi không cần. Chỉ làm đúng những gì người dùng yêu cầu.
4. Người liên hệ: tìm bằng contacts.search với tên như người dùng nói. Nếu ra nhiều người trùng tên mà không biết là ai, gọi sys.ask_missing_info với tool cần gọi và tham số còn thiếu (ví dụ ["contact_id"]).
5. Thiếu thông tin bắt buộc (số tiền, thời gian, đối tượng...) thì làm trước các phần làm được, rồi gọi sys.ask_missing_info {"tool_name": "<app.tool>", "params": [<tên tham số thiếu>]} ở cuối và dừng.
6. Nếu không app nào đã cài làm được yêu cầu, hoặc người dùng đổi ý / bảo thôi, trả về {"name": "done"} ngay, không gọi tool nào.
7. Yêu cầu có điều kiện ("nếu mai mưa thì...", "nếu ví đủ tiền thì..."): gọi tool đọc để kiểm tra, rồi chỉ làm nhánh đúng.
8. Định dạng: thời điểm ISO-8601 kèm múi giờ (2026-11-16T17:00:00+07:00), ngày YYYY-MM-DD, giờ HH:MM, tiền là số nguyên VND (500 nghìn = 500000, 1 triệu rưỡi = 1500000). Tính "mai", "tối nay", "thứ Sáu tuần sau" từ thời gian hiện tại.
9. Nội dung tin nhắn, nhắc nhở, ghi chú: viết ngắn gọn, giữ đúng ý và từ ngữ của người dùng."""


def _type_str(p: dict) -> str:
    if p.get("enum"):
        return "|".join(str(x) for x in p["enum"])
    t = p.get("type", "any")
    if t == "array":
        it = p.get("items") or {}
        return f"{_type_str(it)}[]" if it else "array"
    if p.get("format"):
        return f"{t}<{p['format']}>"
    return t


def _param_str(name: str, p: dict, required: bool) -> str:
    s = f"{name}{'*' if required else ''}: {_type_str(p)}"
    extras = []
    for k, sym in (("minimum", ">="), ("exclusiveMinimum", ">"), ("maximum", "<="), ("exclusiveMaximum", "<")):
        if k in p:
            extras.append(f"{sym}{p[k]}")
    if p.get("description"):
        extras.append(p["description"])
    return s + (f" ({'; '.join(extras)})" if extras else "")


def render_tool(t: dict) -> str:
    params = t.get("parameters") or {}
    req = set(params.get("required", []))
    props = params.get("properties", {})
    sig = ", ".join(_param_str(k, p, k in req) for k, p in props.items())
    line = f"- {t['name']} [{t.get('kind', 'write')}] {t.get('description', '')}\n    args: {sig or '(không có)'}"
    if params.get("x-one-of"):
        line += "\n    cần một trong: " + " | ".join(" + ".join(g) for g in params["x-one-of"])
    if t.get("returns"):
        line += f"\n    trả về: {t['returns']}"
    return line


def render_tools(sample: dict, tools: dict, apps: dict) -> str:
    out = []
    for app_id, a in sample["context"]["installed_apps"].items():
        meta = apps.get(app_id, {})
        out.append(f"## {app_id} ({meta.get('name_vi', '')}): {meta.get('description', '')}")
        out += [render_tool(tools[n]) for n in a["tools"] if n in tools]
    out.append("## sys (luôn có)")
    out.append(render_tool(tools["sys.ask_missing_info"]))
    return "\n".join(out)


def render_context(ctx: dict) -> str:
    now = datetime.fromisoformat(ctx["current_time"])
    tomorrow = now + timedelta(days=1)
    prof = ctx.get("user_profile") or {}
    lines = [
        f"Thời gian hiện tại: {ctx['current_time']} ({WEEKDAYS[now.weekday()]}, {now:%d/%m/%Y %H:%M}), múi giờ {ctx.get('timezone', '')}",
        f"Ngày mai: {tomorrow:%Y-%m-%d} ({WEEKDAYS[tomorrow.weekday()]})",
    ]
    if ctx.get("location"):
        lines.append(f"Vị trí hiện tại: {json.dumps(ctx['location'], ensure_ascii=False)}")
    for k, label in (("name", "Tên người dùng"), ("home_address", "Địa chỉ nhà"), ("work_address", "Nơi làm việc"), ("city", "Thành phố")):
        if prof.get(k):
            lines.append(f"{label}: {prof[k]}")
    for k, v in prof.items():
        if k not in ("name", "home_address", "work_address", "city"):
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    return "\n".join(lines)


def _short(obj, n: int) -> str:
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return s if len(s) <= n else s[:n] + "…"


def render_history(history: list[dict], user_texts: list[str]) -> str:
    """user_texts[i] is the transcript of the i-th user turn of history."""
    if not history:
        return ""
    lines, u = [], 0
    for h in history:
        if h["role"] == "user":
            lines.append(f"Người dùng: {user_texts[u] if u < len(user_texts) else ''}")
            u += 1
            continue
        results = h.get("tool_results") or []
        for i, c in enumerate(h.get("tool_calls") or []):
            r = results[i] if i < len(results) else None
            lines.append(f"  [gọi] {_short(c, 300)}" + (f" -> {_short(r, HISTORY_RESULT_CHARS)}" if r is not None else ""))
        if h.get("text"):
            lines.append(f"Trợ lý: {h['text']}")
    return "\n".join(lines)


def build_messages(sample: dict, tools: dict, apps: dict, request_text: str, history_texts: list[str]) -> list[dict]:
    system = (f"{SYSTEM_RULES}\n\n# Ngữ cảnh người dùng\n{render_context(sample['context'])}\n\n"
              f"# Các app đã cài và tool (* = bắt buộc)\n{render_tools(sample, tools, apps)}")
    hist = render_history(sample["history"], history_texts)
    user = (f"# Lịch sử hội thoại (các thao tác này đã được thực hiện)\n{hist}\n\n" if hist else "") + \
           f"# Yêu cầu hiện tại của người dùng\n{request_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def tool_result_message(result: dict) -> dict:
    return {"role": "user", "content": f"Kết quả: {_short(result, TOOL_RESULT_CHARS)}\nGọi tool tiếp theo, hoặc {{\"name\": \"done\"}} nếu đã xong."}
