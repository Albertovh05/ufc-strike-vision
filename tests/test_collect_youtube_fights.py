"""Tests for YouTube/UFCStats collection helpers."""

from src.collect_youtube_fights import (
    UFCStatsIndexEntry,
    YouTubeFightCandidate,
    names_match,
    pair_matches,
    parse_full_fight_title,
)


def test_parse_full_fight_title():
    assert parse_full_fight_title("Conor McGregor vs Max Holloway 1 | FULL FIGHT") == (
        "Conor McGregor",
        "Max Holloway",
    )


def test_names_match_handles_accents_and_suffix_numbers():
    assert names_match("Jiri Prochazka 2", "Jiří Procházka")


def test_pair_matches_either_order():
    candidate = YouTubeFightCandidate(
        video_id="abc",
        url="https://www.youtube.com/watch?v=abc",
        title="Josh Hokit vs Curtis Blaydes | FULL FIGHT",
        uploader="UFC",
        duration_seconds=1000,
        view_count=1,
        fighter_a_query="Josh Hokit",
        fighter_b_query="Curtis Blaydes",
    )
    entry = UFCStatsIndexEntry(
        event_name="UFC Test",
        event_date="2026-01-01",
        event_url="http://ufcstats.com/event-details/test",
        fight_url="http://ufcstats.com/fight-details/test",
        fighter_a="Curtis Blaydes",
        fighter_b="Josh Hokit",
    )

    assert pair_matches(candidate, entry)
