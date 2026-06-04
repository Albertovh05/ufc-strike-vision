"""UFCStats scraper — per-round strike stats for every fight on an event card."""

import argparse
import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
_DELAY = 0.8  # seconds between requests — be polite


# ── HTTP ───────────────────────────────────────────────────────────────────

def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "ufc-tracker/0.1 (research)"
    return s


# ── Parsing helpers ────────────────────────────────────────────────────────

def _parse_of(text: str) -> dict[str, int]:
    """'25 of 51' → {'landed': 25, 'attempted': 51}. Returns zeros on failure."""
    m = re.search(r"(\d+)\s+of\s+(\d+)", text)
    if m:
        return {"landed": int(m.group(1)), "attempted": int(m.group(2))}
    return {"landed": 0, "attempted": 0}


def _safe_int(text: str) -> int:
    m = re.search(r"\d+", text)
    return int(m.group()) if m else 0


def _cell(td, fighter_idx: int) -> str:
    """
    UFCStats per-round rows: each <td> holds TWO <p> children, one per fighter.
    fighter_idx 0 = fighter A (red corner), 1 = fighter B (blue corner).
    """
    ps = [p.get_text(" ", strip=True) for p in td.select("p")]
    if len(ps) > fighter_idx:
        return ps[fighter_idx]
    # fallback for cells that don't follow the two-<p> pattern
    return td.get_text(" ", strip=True) if fighter_idx == 0 else ""


# ── Row parsers ────────────────────────────────────────────────────────────
#
# Totals table column order (0-indexed):
#   0  Fighter names
#   1  KD
#   2  Sig. Str.  (landed of attempted)
#   3  Sig. Str. %
#   4  Total Str. (landed of attempted)
#   5  Td         (landed of attempted)
#   6  Td %
#   7  Sub. Att
#   8  Rev.
#   9  Ctrl
#
# Significant Strikes table column order:
#   0  Fighter names
#   1  Sig. Str. total (same as Totals col 2)
#   2  Head
#   3  Body
#   4  Leg
#   5  Distance
#   6  Clinch
#   7  Ground

def _parse_totals(tds: list, fi: int) -> dict:
    def v(i: int) -> str:
        return _cell(tds[i], fi) if i < len(tds) else ""

    return {
        "knockdowns": _safe_int(v(1)),
        "sig_strikes": _parse_of(v(2)),
        "sig_strikes_pct": v(3),
        "total_strikes": _parse_of(v(4)),
        "takedowns": _parse_of(v(5)),
        "takedown_pct": v(6),
        "submission_attempts": _safe_int(v(7)),
        "reversals": _safe_int(v(8)),
        "ctrl_time": v(9),
    }


def _parse_sig(tds: list, fi: int) -> dict:
    def v(i: int) -> str:
        return _cell(tds[i], fi) if i < len(tds) else ""

    return {
        "sig_strikes_head": _parse_of(v(2)),
        "sig_strikes_body": _parse_of(v(3)),
        "sig_strikes_leg": _parse_of(v(4)),
        "sig_strikes_distance": _parse_of(v(5)),
        "sig_strikes_clinch": _parse_of(v(6)),
        "sig_strikes_ground": _parse_of(v(7)),
    }


# ── Section finders ────────────────────────────────────────────────────────
#
# UFCStats fight pages have four <section class="b-fight-details__section">
# blocks:
#   • Per-fight Totals         — heading class ends in _tot
#   • Per-fight Sig Strikes    — heading class ends in _tot
#   • Per-ROUND Totals         — heading class ends in _rnd
#   • Per-ROUND Sig Strikes    — heading class ends in _rnd
#
# We want the _rnd sections; fall back to _tot if the page only has one set.

def _data_rows(section) -> list:
    """Table rows that are not header rows (contain <td>, not <th>)."""
    return [r for r in section.select("tr.b-fight-details__table-row") if r.select("td")]


def _find_rows(soup: BeautifulSoup, keyword: str) -> list:
    """Return data rows from the per-round section whose heading matches keyword."""
    for cls_suffix in ("_rnd", "_tot"):
        for section in soup.select("section.b-fight-details__section"):
            link = section.select_one(f"p[class*='collapse-link{cls_suffix}']")
            if link and keyword.lower() in link.get_text(strip=True).lower():
                rows = _data_rows(section)
                if rows:
                    return rows
    return []


# ── Fight scraper ──────────────────────────────────────────────────────────

def scrape_fight(session: requests.Session, fight_url: str, date: str) -> dict:
    """Scrape one fight page and return the canonical fight dict."""
    resp = session.get(fight_url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    fight_id = fight_url.rstrip("/").split("/")[-1]

    # Fighter names — prefer the <a> inside the name <h3>; fall back to <h3> text
    name_els = soup.select("div.b-fight-details__person h3.b-fight-details__person-name a")
    if len(name_els) < 2:
        name_els = soup.select("div.b-fight-details__person h3.b-fight-details__person-name")
    if len(name_els) < 2:
        raise ValueError(f"Could not find two fighter names at {fight_url}")

    fighter_a = name_els[0].get_text(strip=True)
    fighter_b = name_els[1].get_text(strip=True)

    tot_rows = _find_rows(soup, "Totals")
    sig_rows = _find_rows(soup, "Significant Strikes")

    if not tot_rows:
        raise ValueError(f"No Totals rows found at {fight_url}")
    if not sig_rows:
        raise ValueError(f"No Significant Strikes rows found at {fight_url}")

    rounds: list[dict] = []
    for rnum, (tr_tot, tr_sig) in enumerate(zip(tot_rows, sig_rows), start=1):
        tot_tds = tr_tot.select("td")
        sig_tds = tr_sig.select("td")

        fa = _parse_totals(tot_tds, 0)
        fa.update(_parse_sig(sig_tds, 0))

        fb = _parse_totals(tot_tds, 1)
        fb.update(_parse_sig(sig_tds, 1))

        rounds.append({"round": rnum, "fighter_a": fa, "fighter_b": fb})

    return {
        "fight_id": fight_id,
        "date": date,
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "rounds": rounds,
    }


# ── Event scraper ──────────────────────────────────────────────────────────

def scrape_event(session: requests.Session, event_url: str) -> tuple[str, list[str]]:
    """Return (ISO-date, [fight_url, ...]) for an event listing page."""
    resp = session.get(event_url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Date is in a <li class="b-list__box-list-item"> that contains "Date:"
    date_str = ""
    for li in soup.select("li.b-list__box-list-item"):
        text = li.get_text(" ", strip=True)
        if "Date:" in text:
            raw = text.replace("Date:", "").strip()
            try:
                date_str = datetime.strptime(raw, "%B %d, %Y").strftime("%Y-%m-%d")
            except ValueError:
                date_str = raw  # keep whatever came back rather than losing it
            break

    # Fight links live inside <tr> rows as <a href="…/fight-details/…">
    seen: set[str] = set()
    fight_urls: list[str] = []
    for a in soup.select("tr.b-fight-details__table-row td a"):
        href = a.get("href", "")
        if href.startswith("http://ufcstats.com/fight-details/") and href not in seen:
            seen.add(href)
            fight_urls.append(href)

    return date_str, fight_urls


# ── File naming ────────────────────────────────────────────────────────────

def _lastname(full_name: str) -> str:
    """ASCII last word of a fighter's name, lower-cased, non-alnum stripped."""
    ascii_name = unicodedata.normalize("NFKD", full_name).encode("ascii", "ignore").decode()
    parts = ascii_name.strip().split()
    return re.sub(r"[^a-z0-9]", "", parts[-1].lower()) if parts else "unknown"


def _out_path(data: dict, out_dir: Path) -> Path:
    la = _lastname(data["fighter_a"])
    lb = _lastname(data["fighter_b"])
    digits = re.sub(r"[^0-9]", "", data["date"])[:8]  # YYYYMMDD
    return out_dir / f"{la}_{lb}_{digits}.json"


# ── Orchestrator ───────────────────────────────────────────────────────────

def run(event_url: str, out_dir: Path = RAW_DIR, delay: float = _DELAY) -> list[Path]:
    """Scrape every fight on an event card and write JSON files to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session = _session()

    print(f"Fetching event: {event_url}")
    date, fight_urls = scrape_event(session, event_url)
    print(f"  date: {date or '(not found)'}  |  {len(fight_urls)} fight(s) found")

    saved: list[Path] = []
    for url in fight_urls:
        time.sleep(delay)
        try:
            data = scrape_fight(session, url, date)
            path = _out_path(data, out_dir)
            if path.exists():
                print(f"  skip (exists): {path.name}")
                continue
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            saved.append(path)
            print(f"  saved: {path.name}  ({len(data['rounds'])} round(s))")
        except requests.HTTPError as exc:
            print(f"  HTTP {exc.response.status_code} — {url}")
        except ValueError as exc:
            print(f"  parse error — {url}: {exc}")
        except Exception as exc:
            print(f"  unexpected error — {url}: {exc}")

    return saved


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scrape per-round strike stats from a UFCStats event page."
    )
    ap.add_argument("--url", required=True, help="UFCStats event URL")
    ap.add_argument("--output", default=str(RAW_DIR), help="Output directory for JSON files")
    ap.add_argument(
        "--delay",
        type=float,
        default=_DELAY,
        help="Seconds between requests (default: %(default)s)",
    )
    args = ap.parse_args()
    run(event_url=args.url, out_dir=Path(args.output), delay=args.delay)


if __name__ == "__main__":
    main()
