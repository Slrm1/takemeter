"""Collect public r/nfl comments via the Arctic Shift mirror.

Arctic Shift republishes public Reddit data. This script does not assign labels.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw_comments.jsonl"
UA = "takemeter-course-project/0.1 (public r/nfl discourse research)"
BASE = "https://arctic-shift.photon-reddit.com"


def get_json(path: str, params: dict) -> dict:
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"failed {url}: {last_err}")


def source_type(title: str) -> str:
    low = (title or "").lower()
    if "game thread" in low or "redzone" in low:
        return "game_thread"
    if "post game" in low or "post-game" in low or "postgame" in low:
        return "postgame"
    if any(word in low for word in ("power ranking", "rank", "prediction", "hot take")):
        return "ranking_or_prediction"
    if any(word in low for word in ("discussion", "free talk", "who would", "what are")):
        return "discussion"
    return "news_or_other"


def main() -> None:
    comments: list[dict] = []
    seen: set[str] = set()
    before: str | None = None
    pages = 0
    while len(comments) < 1200 and pages < 25:
        params = {"subreddit": "nfl", "limit": 100}
        if before:
            params["before"] = before
        payload = get_json("/api/comments/search", params)
        batch = payload.get("data") or []
        if not batch:
            break
        pages += 1
        oldest = batch[-1]["created_utc"]
        for item in batch:
            body = (item.get("body") or "").replace("\n", " ").strip()
            author = item.get("author") or ""
            if author in {"AutoModerator", "[deleted]", "nfl_mod"}:
                continue
            if body in {"[deleted]", "[removed]"} or len(body) < 40 or len(body) > 900:
                continue
            if body.lower().startswith("i am a bot"):
                continue
            key = " ".join(body.lower().split())
            if key in seen:
                continue
            seen.add(key)
            comments.append(
                {
                    "comment_id": item.get("id"),
                    "text": body,
                    "score": item.get("score"),
                    "created_utc": item.get("created_utc"),
                    "link_id": (item.get("link_id") or "").removeprefix("t3_"),
                    "source_url": "https://www.reddit.com" + (item.get("permalink") or ""),
                }
            )
        before = str(int(oldest))
        print(f"page {pages}: batch={len(batch)} kept={len(comments)} before={before}")
        time.sleep(0.4)

    link_ids = sorted({row["link_id"] for row in comments if row["link_id"]})
    titles: dict[str, str] = {}
    for i in range(0, len(link_ids), 50):
        chunk = link_ids[i : i + 50]
        # Arctic Shift accepts repeated ids via a comma-separated ids query on posts.
        payload = get_json("/api/posts/ids", {"ids": ",".join(chunk)})
        for post in payload.get("data") or []:
            titles[post.get("id")] = post.get("title") or ""
        print(f"titles {i + len(chunk)}/{len(link_ids)}")
        time.sleep(0.3)

    for row in comments:
        title = titles.get(row["link_id"], "")
        row["thread_title"] = title
        row["source_type"] = source_type(title)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in comments:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(comments)} -> {OUT}")


if __name__ == "__main__":
    main()
