"""Collect YouTube full-fight metadata and matching UFCStats strike counts.

This script is intentionally conservative with YouTube media. By default it
records video URLs and metadata, then scrapes UFCStats JSON for matched fights.
Use `--download-videos` only for videos you are allowed to store locally.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from bs4 import BeautifulSoup

from src.scraper import RAW_DIR, _out_path, _session, scrape_fight


PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "youtube_fights_manifest.json"
EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"
DEFAULT_QUERY = "UFC FULL FIGHT"
OFFICIAL_UPLOADERS = ("ufc", "ufc india", "ufc on paramount+")


@dataclass
class YouTubeFightCandidate:
    video_id: str
    url: str
    title: str
    uploader: str
    duration_seconds: float | None
    view_count: int | None
    fighter_a_query: str
    fighter_b_query: str


@dataclass
class UFCStatsIndexEntry:
    event_name: str
    event_date: str
    event_url: str
    fight_url: str
    fighter_a: str
    fighter_b: str


@dataclass
class CollectedFight:
    youtube: YouTubeFightCandidate
    ufcstats: UFCStatsIndexEntry
    stats_json: str
    video_path: str | None = None


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def normalize_name(text: str) -> str:
    text = _ascii(text).lower()
    text = re.sub(r"\b\d+\b$", "", text).strip()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _name_tokens(text: str) -> set[str]:
    return {token for token in normalize_name(text).split() if token}


def names_match(query_name: str, stats_name: str) -> bool:
    query_tokens = _name_tokens(query_name)
    stats_tokens = _name_tokens(stats_name)
    if not query_tokens or not stats_tokens:
        return False
    if query_tokens == stats_tokens:
        return True
    if query_tokens.issubset(stats_tokens) or stats_tokens.issubset(query_tokens):
        return True
    return bool(query_tokens & stats_tokens) and _last_token(query_name) == _last_token(stats_name)


def _last_token(text: str) -> str:
    tokens = normalize_name(text).split()
    return tokens[-1] if tokens else ""


def pair_matches(candidate: YouTubeFightCandidate, entry: UFCStatsIndexEntry) -> bool:
    direct = names_match(candidate.fighter_a_query, entry.fighter_a) and names_match(
        candidate.fighter_b_query, entry.fighter_b
    )
    reversed_pair = names_match(candidate.fighter_a_query, entry.fighter_b) and names_match(
        candidate.fighter_b_query, entry.fighter_a
    )
    return direct or reversed_pair


def pair_key(fighter_a: str, fighter_b: str) -> tuple[str, str]:
    return tuple(sorted((normalize_name(fighter_a), normalize_name(fighter_b))))  # type: ignore[return-value]


def parse_full_fight_title(title: str) -> tuple[str, str] | None:
    head = re.split(r"\s*[|:\-]\s*(?:full\s+fight|ufc|free\s+fight)\b", title, flags=re.I)[0]
    match = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", head, flags=re.I)
    if not match:
        return None

    fighter_a = clean_title_name(match.group(1))
    fighter_b = clean_title_name(match.group(2))
    if not fighter_a or not fighter_b:
        return None
    return fighter_a, fighter_b


def clean_title_name(text: str) -> str:
    text = re.sub(r"\bfull\s+fight\b", "", text, flags=re.I)
    text = re.sub(r"\bufc\b.*$", "", text, flags=re.I)
    text = re.sub(r"\b\d+\b$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -|:")


def _require_yt_dlp():
    try:
        import yt_dlp
    except ImportError as exc:
        raise ImportError(
            "yt-dlp is required for YouTube collection. Install it with `pip install yt-dlp`."
        ) from exc
    return yt_dlp


def search_youtube_full_fights(
    *,
    query: str = DEFAULT_QUERY,
    search_limit: int = 50,
    min_duration_seconds: int = 300,
) -> list[YouTubeFightCandidate]:
    yt_dlp = _require_yt_dlp()
    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
    }

    candidates: list[YouTubeFightCandidate] = []
    seen_pairs: set[tuple[str, str]] = set()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{search_limit}:{query}", download=False)

    for entry in info.get("entries", []):
        if not entry:
            continue
        title = entry.get("title") or ""
        uploader = entry.get("uploader") or entry.get("channel") or ""
        duration = entry.get("duration")
        if "full fight" not in title.lower():
            continue
        if duration is not None and float(duration) < min_duration_seconds:
            continue
        if uploader and not uploader.lower().startswith(OFFICIAL_UPLOADERS):
            continue

        parsed = parse_full_fight_title(title)
        if parsed is None:
            continue
        fighter_a, fighter_b = parsed
        key = pair_key(fighter_a, fighter_b)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        video_id = entry.get("id") or ""
        url = entry.get("webpage_url") or entry.get("url") or ""
        if video_id and not url.startswith("http"):
            url = f"https://www.youtube.com/watch?v={video_id}"

        candidates.append(
            YouTubeFightCandidate(
                video_id=video_id,
                url=url,
                title=title,
                uploader=uploader,
                duration_seconds=float(duration) if duration is not None else None,
                view_count=entry.get("view_count"),
                fighter_a_query=fighter_a,
                fighter_b_query=fighter_b,
            )
        )

    return candidates


def iter_completed_events(session, *, max_events: int | None = None) -> Iterable[tuple[str, str, str]]:
    response = session.get(EVENTS_URL, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    count = 0
    for row in soup.select("tr.b-statistics__table-row"):
        link = row.select_one("a[href*='event-details']")
        if not link:
            continue
        event_name = link.get_text(" ", strip=True)
        date_el = row.select_one(".b-statistics__date")
        event_date = date_el.get_text(" ", strip=True) if date_el else ""
        yield event_name, _to_iso_date(event_date), link.get("href")
        count += 1
        if max_events is not None and count >= max_events:
            return


def _to_iso_date(text: str) -> str:
    try:
        return datetime.strptime(text, "%B %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        return text


def parse_event_fights(session, event_name: str, event_date: str, event_url: str) -> list[UFCStatsIndexEntry]:
    response = session.get(event_url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    fights: list[UFCStatsIndexEntry] = []
    for row in soup.select("tr.b-fight-details__table-row"):
        link = row.select_one("a[href*='fight-details']")
        if not link:
            continue
        texts = [p.get_text(" ", strip=True) for p in row.select("p.b-fight-details__table-text")]
        if len(texts) < 3:
            texts = [text for text in row.get_text(" | ", strip=True).split(" | ") if text]
        if len(texts) < 3:
            continue

        fights.append(
            UFCStatsIndexEntry(
                event_name=event_name,
                event_date=event_date,
                event_url=event_url,
                fight_url=link.get("href"),
                fighter_a=texts[1],
                fighter_b=texts[2],
            )
        )
    return fights


def find_ufcstats_matches(
    candidates: Sequence[YouTubeFightCandidate],
    *,
    limit: int,
    max_events: int | None = None,
    delay: float = 0.2,
) -> list[tuple[YouTubeFightCandidate, UFCStatsIndexEntry]]:
    session = _session()
    unmatched = list(candidates)
    matches: list[tuple[YouTubeFightCandidate, UFCStatsIndexEntry]] = []

    for event_name, event_date, event_url in iter_completed_events(session, max_events=max_events):
        fights = parse_event_fights(session, event_name, event_date, event_url)
        for entry in fights:
            for candidate in list(unmatched):
                if pair_matches(candidate, entry):
                    matches.append((candidate, entry))
                    unmatched.remove(candidate)
                    break
            if len(matches) >= limit:
                return matches
        if delay > 0:
            time.sleep(delay)

    return matches


def write_fight_stats(entry: UFCStatsIndexEntry, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = _session()
    data = scrape_fight(session, entry.fight_url, entry.event_date)
    path = _out_path(data, out_dir)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def download_video(url: str, video_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / f"{video_id}.%(ext)s")
    cmd = [
        "python",
        "-m",
        "yt_dlp",
        "--no-playlist",
        "-f",
        "bv*[height<=720]+ba/b[height<=720]/b",
        "-o",
        output_template,
        url,
    ]
    subprocess.run(cmd, check=True)
    matches = sorted(out_dir.glob(f"{video_id}.*"))
    if not matches:
        raise FileNotFoundError(f"yt-dlp completed but no file was found for {video_id}")
    return matches[0]


def collect(
    *,
    limit: int = 5,
    query: str = DEFAULT_QUERY,
    search_limit: int = 50,
    max_events: int | None = 350,
    out_dir: Path = RAW_DIR,
    manifest_path: Path = MANIFEST_PATH,
    download_videos: bool = False,
    video_dir: Path = VIDEO_DIR,
    delay: float = 0.2,
) -> list[CollectedFight]:
    candidates = search_youtube_full_fights(query=query, search_limit=search_limit)
    matches = find_ufcstats_matches(candidates, limit=limit, max_events=max_events, delay=delay)

    collected: list[CollectedFight] = []
    for youtube, ufcstats in matches:
        stats_path = write_fight_stats(ufcstats, out_dir)
        video_path = None
        if download_videos:
            video_path = str(download_video(youtube.url, youtube.video_id, video_dir))
        collected.append(
            CollectedFight(
                youtube=youtube,
                ufcstats=ufcstats,
                stats_json=str(stats_path),
                video_path=video_path,
            )
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps([asdict(item) for item in collected], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect official YouTube full-fight metadata and matching UFCStats JSON."
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--search-limit", type=int, default=50)
    parser.add_argument("--max-events", type=int, default=350)
    parser.add_argument("--output", type=Path, default=RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--download-videos", action="store_true")
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    args = parser.parse_args()

    collected = collect(
        limit=args.limit,
        query=args.query,
        search_limit=args.search_limit,
        max_events=args.max_events,
        out_dir=args.output,
        manifest_path=args.manifest,
        download_videos=args.download_videos,
        video_dir=args.video_dir,
        delay=args.delay,
    )

    for item in collected:
        print(
            f"{item.youtube.fighter_a_query} vs {item.youtube.fighter_b_query} -> "
            f"{Path(item.stats_json).name} | {item.youtube.url}"
        )
    print(f"Collected {len(collected)} fight(s); manifest: {args.manifest}")


if __name__ == "__main__":
    main()
