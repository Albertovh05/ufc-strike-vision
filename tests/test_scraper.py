"""Tests for src/scraper.py — unit, schema-validation, and mocked-HTTP integration."""

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scraper import (
    _cell,
    _lastname,
    _out_path,
    _parse_of,
    _safe_int,
    scrape_event,
    scrape_fight,
)
from bs4 import BeautifulSoup


# ── Schema validation ──────────────────────────────────────────────────────
#
# These constants mirror the canonical JSON schema defined in README.md.
# Keep them in sync if the schema changes.

_OF_KEYS = {"landed", "attempted"}
_OF_FIELDS = {"sig_strikes", "total_strikes", "takedowns"}
_OF_SUBFIELDS = {
    "sig_strikes_head", "sig_strikes_body", "sig_strikes_leg",
    "sig_strikes_distance", "sig_strikes_clinch", "sig_strikes_ground",
}
_INT_FIELDS = {"knockdowns", "submission_attempts", "reversals"}
_STR_FIELDS = {"sig_strikes_pct", "takedown_pct", "ctrl_time"}
_TOP_LEVEL = {"fight_id", "date", "fighter_a", "fighter_b", "rounds"}


def validate_schema(data: dict) -> list[str]:
    """Return a list of schema-violation messages; empty list means valid."""
    errors: list[str] = []

    for key in _TOP_LEVEL:
        if key not in data:
            errors.append(f"missing top-level key: '{key}'")

    if not isinstance(data.get("rounds"), list):
        errors.append("'rounds' must be a list")
        return errors

    if not data["rounds"]:
        errors.append("'rounds' is empty")

    for i, rnd in enumerate(data["rounds"]):
        prefix = f"rounds[{i}]"

        if "round" not in rnd:
            errors.append(f"{prefix}: missing 'round'")
        elif not isinstance(rnd["round"], int):
            errors.append(f"{prefix}.round: must be int, got {type(rnd['round']).__name__}")

        for fk in ("fighter_a", "fighter_b"):
            if fk not in rnd:
                errors.append(f"{prefix}: missing '{fk}'")
                continue

            stats = rnd[fk]
            fp = f"{prefix}.{fk}"

            for field in _OF_FIELDS:
                if field not in stats:
                    errors.append(f"{fp}: missing '{field}'")
                elif not isinstance(stats[field], dict):
                    errors.append(f"{fp}.{field}: must be dict, got {type(stats[field]).__name__}")
                else:
                    for sub in _OF_KEYS:
                        if sub not in stats[field]:
                            errors.append(f"{fp}.{field}: missing sub-key '{sub}'")

            for field in _OF_SUBFIELDS:
                if field not in stats:
                    errors.append(f"{fp}: missing '{field}'")
                elif not isinstance(stats[field], dict):
                    errors.append(f"{fp}.{field}: must be dict")
                else:
                    for sub in _OF_KEYS:
                        if sub not in stats[field]:
                            errors.append(f"{fp}.{field}: missing sub-key '{sub}'")

            for field in _INT_FIELDS:
                if field not in stats:
                    errors.append(f"{fp}: missing '{field}'")
                elif not isinstance(stats[field], int):
                    errors.append(f"{fp}.{field}: must be int, got {type(stats[field]).__name__}")

            for field in _STR_FIELDS:
                if field not in stats:
                    errors.append(f"{fp}: missing '{field}'")

    return errors


# ── HTML fixtures ──────────────────────────────────────────────────────────
#
# Minimal HTML that reproduces the UFCStats DOM structure used by the scraper.
# One row = one round; each <td> has two <p> children (fighter A then fighter B).

EVENT_HTML = textwrap.dedent("""\
    <html><body>
    <ul class="b-list__box-list">
      <li class="b-list__box-list-item">
        <i class="b-list__box-item-title">Date:</i>October 06, 2018
      </li>
    </ul>
    <table><tbody>
      <tr class="b-fight-details__table-row">
        <td><a href="http://ufcstats.com/fight-details/abc111">fight 1</a></td>
        <td><a href="http://ufcstats.com/fighter-details/xyz">fighter link (should be ignored)</a></td>
      </tr>
      <tr class="b-fight-details__table-row">
        <td><a href="http://ufcstats.com/fight-details/abc222">fight 2</a></td>
      </tr>
    </tbody></table>
    </body></html>
""")

# One-round fight between McGregor and Khabib
FIGHT_HTML = textwrap.dedent("""\
    <html><body>
    <div class="b-fight-details__person">
      <h3 class="b-fight-details__person-name"><a>Conor McGregor</a></h3>
    </div>
    <div class="b-fight-details__person">
      <h3 class="b-fight-details__person-name"><a>Khabib Nurmagomedov</a></h3>
    </div>

    <!-- Per-round Totals -->
    <section class="b-fight-details__section">
      <p class="b-fight-details__collapse-link b-fight-details__collapse-link_rnd">Totals</p>
      <table><thead>
        <tr class="b-fight-details__table-row">
          <th>Fighter</th><th>KD</th><th>Sig. str.</th><th>Sig. str. %</th>
          <th>Total str.</th><th>Td</th><th>Td %</th><th>Sub. att</th>
          <th>Rev.</th><th>Ctrl</th>
        </tr>
      </thead><tbody>
        <tr class="b-fight-details__table-row">
          <td><p><a>Conor McGregor</a></p><p><a>Khabib Nurmagomedov</a></p></td>
          <td><p>0</p><p>1</p></td>
          <td><p>25 of 51</p><p>29 of 44</p></td>
          <td><p>49%</p><p>65%</p></td>
          <td><p>35 of 65</p><p>37 of 55</p></td>
          <td><p>0 of 1</p><p>2 of 3</p></td>
          <td><p>0%</p><p>66%</p></td>
          <td><p>0</p><p>0</p></td>
          <td><p>0</p><p>0</p></td>
          <td><p>0:00</p><p>4:30</p></td>
        </tr>
      </tbody></table>
    </section>

    <!-- Per-round Significant Strikes -->
    <section class="b-fight-details__section">
      <p class="b-fight-details__collapse-link b-fight-details__collapse-link_rnd">Significant Strikes</p>
      <table><thead>
        <tr class="b-fight-details__table-row">
          <th>Fighter</th><th>Sig. str.</th><th>Head</th><th>Body</th><th>Leg</th>
          <th>Distance</th><th>Clinch</th><th>Ground</th>
        </tr>
      </thead><tbody>
        <tr class="b-fight-details__table-row">
          <td><p><a>Conor McGregor</a></p><p><a>Khabib Nurmagomedov</a></p></td>
          <td><p>25 of 51</p><p>29 of 44</p></td>
          <td><p>18 of 35</p><p>10 of 20</p></td>
          <td><p>5 of 10</p><p>12 of 15</p></td>
          <td><p>2 of 6</p><p>7 of 9</p></td>
          <td><p>15 of 30</p><p>10 of 15</p></td>
          <td><p>5 of 10</p><p>12 of 18</p></td>
          <td><p>5 of 11</p><p>7 of 11</p></td>
        </tr>
      </tbody></table>
    </section>
    </body></html>
""")

# Same structure but with 3 rounds
FIGHT_HTML_3R = FIGHT_HTML.replace(
    # Duplicate the single data row twice more to simulate 3 rounds
    "<tr class=\"b-fight-details__table-row\">\n          <td><p>0</p><p>1</p></td>",
    "\n".join([
        "<tr class=\"b-fight-details__table-row\">",
        "          <td><p><a>Conor McGregor</a></p><p><a>Khabib Nurmagomedov</a></p></td>",
        "          <td><p>0</p><p>0</p></td>",
        "          <td><p>10 of 20</p><p>15 of 22</p></td>",
        "          <td><p>50%</p><p>68%</p></td>",
        "          <td><p>15 of 30</p><p>20 of 28</p></td>",
        "          <td><p>0 of 0</p><p>1 of 2</p></td>",
        "          <td><p>0%</p><p>50%</p></td>",
        "          <td><p>0</p><p>0</p></td>",
        "          <td><p>0</p><p>0</p></td>",
        "          <td><p>0:00</p><p>2:10</p></td>",
        "        </tr>",
        "        <tr class=\"b-fight-details__table-row\">",
        "          <td><p><a>Conor McGregor</a></p><p><a>Khabib Nurmagomedov</a></p></td>",
        "          <td><p>0</p><p>1</p></td>",
    ]),
)


# ── Mock session factory ───────────────────────────────────────────────────

def _mock_session(html_by_url: dict[str, str]) -> MagicMock:
    session = MagicMock()

    def get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = html_by_url.get(url, "<html><body></body></html>")
        resp.raise_for_status = MagicMock()
        return resp

    session.get.side_effect = get
    return session


# ── Unit tests: _parse_of ──────────────────────────────────────────────────

class TestParseOf:
    def test_normal(self):
        assert _parse_of("25 of 51") == {"landed": 25, "attempted": 51}

    def test_zeros(self):
        assert _parse_of("0 of 0") == {"landed": 0, "attempted": 0}

    def test_extra_whitespace(self):
        assert _parse_of("  3  of  10  ") == {"landed": 3, "attempted": 10}

    def test_embedded_in_longer_string(self):
        assert _parse_of("Head: 5 of 12 (42%)") == {"landed": 5, "attempted": 12}

    def test_garbage_returns_zeros(self):
        assert _parse_of("---") == {"landed": 0, "attempted": 0}

    def test_empty_string(self):
        assert _parse_of("") == {"landed": 0, "attempted": 0}


# ── Unit tests: _safe_int ─────────────────────────────────────────────────

class TestSafeInt:
    def test_plain_digit(self):
        assert _safe_int("3") == 3

    def test_embedded(self):
        assert _safe_int("Sub: 2") == 2

    def test_garbage(self):
        assert _safe_int("---") == 0

    def test_empty(self):
        assert _safe_int("") == 0


# ── Unit tests: _lastname ─────────────────────────────────────────────────

class TestLastname:
    def test_two_word_name(self):
        assert _lastname("Conor McGregor") == "mcgregor"

    def test_single_name(self):
        assert _lastname("Volkanovski") == "volkanovski"

    def test_three_word_name(self):
        assert _lastname("Francis Ngannou") == "ngannou"

    def test_accented_chars_stripped(self):
        result = _lastname("José Aldo")
        assert result == "aldo"

    def test_hyphen_removed(self):
        # "Blachowicz" has no hyphen but the regex strips non-alnum
        assert _lastname("Jan Blachowicz") == "blachowicz"


# ── Unit tests: _cell ─────────────────────────────────────────────────────

class TestCell:
    def _td(self, html: str):
        return BeautifulSoup(html, "html.parser").find("td")

    def test_returns_first_p_for_fighter_a(self):
        td = self._td("<td><p>25 of 51</p><p>29 of 44</p></td>")
        assert _cell(td, 0) == "25 of 51"

    def test_returns_second_p_for_fighter_b(self):
        td = self._td("<td><p>25 of 51</p><p>29 of 44</p></td>")
        assert _cell(td, 1) == "29 of 44"

    def test_fallback_when_no_p_children(self):
        td = self._td("<td>bare text</td>")
        assert _cell(td, 0) == "bare text"

    def test_fallback_b_returns_empty_string(self):
        td = self._td("<td>bare text</td>")
        assert _cell(td, 1) == ""


# ── Unit tests: _out_path ─────────────────────────────────────────────────

class TestOutPath:
    def test_filename_format(self, tmp_path):
        data = {
            "fight_id": "x",
            "date": "2018-10-06",
            "fighter_a": "Conor McGregor",
            "fighter_b": "Khabib Nurmagomedov",
            "rounds": [],
        }
        p = _out_path(data, tmp_path)
        assert p.name == "mcgregor_nurmagomedov_20181006.json"

    def test_unicode_in_name(self, tmp_path):
        data = {
            "fight_id": "x",
            "date": "2019-03-02",
            "fighter_a": "José Aldo",
            "fighter_b": "Max Holloway",
            "rounds": [],
        }
        p = _out_path(data, tmp_path)
        assert p.name == "aldo_holloway_20190302.json"


# ── Schema validation tests ───────────────────────────────────────────────

def _make_of() -> dict:
    return {"landed": 5, "attempted": 10}


def _make_fighter_stats() -> dict:
    of = _make_of()
    return {
        "knockdowns": 0,
        "sig_strikes": of.copy(),
        "sig_strikes_pct": "50%",
        "total_strikes": of.copy(),
        "takedowns": of.copy(),
        "takedown_pct": "50%",
        "submission_attempts": 0,
        "reversals": 0,
        "ctrl_time": "1:30",
        "sig_strikes_head": of.copy(),
        "sig_strikes_body": of.copy(),
        "sig_strikes_leg": of.copy(),
        "sig_strikes_distance": of.copy(),
        "sig_strikes_clinch": of.copy(),
        "sig_strikes_ground": of.copy(),
    }


def _make_valid_fight() -> dict:
    stats = _make_fighter_stats()
    return {
        "fight_id": "abc123",
        "date": "2018-10-06",
        "fighter_a": "Conor McGregor",
        "fighter_b": "Khabib Nurmagomedov",
        "rounds": [
            {"round": 1, "fighter_a": stats.copy(), "fighter_b": stats.copy()},
            {"round": 2, "fighter_a": stats.copy(), "fighter_b": stats.copy()},
        ],
    }


class TestSchemaValidation:
    def test_valid_data_has_no_errors(self):
        assert validate_schema(_make_valid_fight()) == []

    def test_saved_and_reloaded_file_passes(self, tmp_path):
        data = _make_valid_fight()
        path = tmp_path / "mcgregor_nurmagomedov_20181006.json"
        path.write_text(json.dumps(data, indent=2))
        reloaded = json.loads(path.read_text())
        assert validate_schema(reloaded) == []

    def test_missing_top_level_key(self):
        data = _make_valid_fight()
        del data["fighter_b"]
        errors = validate_schema(data)
        assert any("fighter_b" in e for e in errors)

    def test_missing_rounds_key(self):
        data = _make_valid_fight()
        del data["rounds"]
        errors = validate_schema(data)
        assert any("rounds" in e for e in errors)

    def test_empty_rounds_flagged(self):
        data = _make_valid_fight()
        data["rounds"] = []
        errors = validate_schema(data)
        assert any("empty" in e for e in errors)

    def test_missing_round_number(self):
        data = _make_valid_fight()
        del data["rounds"][0]["round"]
        errors = validate_schema(data)
        assert any("round" in e for e in errors)

    def test_round_number_wrong_type(self):
        data = _make_valid_fight()
        data["rounds"][0]["round"] = "1"
        errors = validate_schema(data)
        assert any("int" in e for e in errors)

    def test_missing_fighter_key_in_round(self):
        data = _make_valid_fight()
        del data["rounds"][0]["fighter_a"]
        errors = validate_schema(data)
        assert any("fighter_a" in e for e in errors)

    def test_missing_stat_field(self):
        data = _make_valid_fight()
        del data["rounds"][0]["fighter_a"]["ctrl_time"]
        errors = validate_schema(data)
        assert any("ctrl_time" in e for e in errors)

    def test_of_field_wrong_type(self):
        data = _make_valid_fight()
        data["rounds"][0]["fighter_a"]["sig_strikes"] = "25 of 51"
        errors = validate_schema(data)
        assert any("sig_strikes" in e for e in errors)

    def test_of_field_missing_subkey(self):
        data = _make_valid_fight()
        del data["rounds"][0]["fighter_a"]["sig_strikes"]["attempted"]
        errors = validate_schema(data)
        assert any("attempted" in e for e in errors)

    def test_missing_sig_subfield(self):
        data = _make_valid_fight()
        del data["rounds"][0]["fighter_b"]["sig_strikes_ground"]
        errors = validate_schema(data)
        assert any("sig_strikes_ground" in e for e in errors)

    def test_int_field_wrong_type(self):
        data = _make_valid_fight()
        data["rounds"][0]["fighter_a"]["knockdowns"] = "0"
        errors = validate_schema(data)
        assert any("knockdowns" in e for e in errors)


# ── Integration tests (mocked HTTP) ──────────────────────────────────────

EVENT_URL = "http://ufcstats.com/event-details/test_event"
FIGHT_URL = "http://ufcstats.com/fight-details/xyz123"


class TestScrapeEvent:
    def test_parses_iso_date(self):
        session = _mock_session({EVENT_URL: EVENT_HTML})
        date, _ = scrape_event(session, EVENT_URL)
        assert date == "2018-10-06"

    def test_returns_fight_urls(self):
        session = _mock_session({EVENT_URL: EVENT_HTML})
        _, fights = scrape_event(session, EVENT_URL)
        assert "http://ufcstats.com/fight-details/abc111" in fights
        assert "http://ufcstats.com/fight-details/abc222" in fights

    def test_excludes_non_fight_links(self):
        session = _mock_session({EVENT_URL: EVENT_HTML})
        _, fights = scrape_event(session, EVENT_URL)
        assert not any("fighter-details" in u for u in fights)

    def test_deduplicates_fight_urls(self):
        html = EVENT_HTML.replace("abc222", "abc111")
        session = _mock_session({EVENT_URL: html})
        _, fights = scrape_event(session, EVENT_URL)
        assert fights.count("http://ufcstats.com/fight-details/abc111") == 1

    def test_empty_page_returns_empty_list(self):
        session = _mock_session({EVENT_URL: "<html></html>"})
        date, fights = scrape_event(session, EVENT_URL)
        assert fights == []
        assert date == ""


class TestScrapeFight:
    def _scrape(self, html=FIGHT_HTML):
        session = _mock_session({FIGHT_URL: html})
        return scrape_fight(session, FIGHT_URL, "2018-10-06")

    def test_fighter_names(self):
        data = self._scrape()
        assert data["fighter_a"] == "Conor McGregor"
        assert data["fighter_b"] == "Khabib Nurmagomedov"

    def test_fight_id(self):
        data = self._scrape()
        assert data["fight_id"] == "xyz123"

    def test_date_stored(self):
        data = self._scrape()
        assert data["date"] == "2018-10-06"

    def test_one_round_parsed(self):
        data = self._scrape()
        assert len(data["rounds"]) == 1

    def test_round_number(self):
        data = self._scrape()
        assert data["rounds"][0]["round"] == 1

    def test_sig_strikes_fighter_a(self):
        data = self._scrape()
        assert data["rounds"][0]["fighter_a"]["sig_strikes"] == {"landed": 25, "attempted": 51}

    def test_sig_strikes_fighter_b(self):
        data = self._scrape()
        assert data["rounds"][0]["fighter_b"]["sig_strikes"] == {"landed": 29, "attempted": 44}

    def test_head_strikes_fighter_a(self):
        data = self._scrape()
        assert data["rounds"][0]["fighter_a"]["sig_strikes_head"] == {"landed": 18, "attempted": 35}

    def test_knockdowns(self):
        data = self._scrape()
        assert data["rounds"][0]["fighter_a"]["knockdowns"] == 0
        assert data["rounds"][0]["fighter_b"]["knockdowns"] == 1

    def test_takedowns_fighter_b(self):
        data = self._scrape()
        assert data["rounds"][0]["fighter_b"]["takedowns"] == {"landed": 2, "attempted": 3}

    def test_ctrl_time(self):
        data = self._scrape()
        assert data["rounds"][0]["fighter_a"]["ctrl_time"] == "0:00"
        assert data["rounds"][0]["fighter_b"]["ctrl_time"] == "4:30"

    def test_full_output_passes_schema(self):
        data = self._scrape()
        assert validate_schema(data) == []

    def test_missing_fighter_names_raises(self):
        html = "<html><body><section class='b-fight-details__section'></section></body></html>"
        session = _mock_session({FIGHT_URL: html})
        with pytest.raises(ValueError, match="fighter"):
            scrape_fight(session, FIGHT_URL, "2018-10-06")

    def test_missing_totals_raises(self):
        # Remove the Totals section entirely
        html = FIGHT_HTML.split("<!-- Per-round Totals -->")[0] + FIGHT_HTML.split("<!-- Per-round Significant Strikes -->")[1]
        # html now has names but no Totals section → should raise
        # (If split fails the test will error rather than pass silently)
        session = _mock_session({FIGHT_URL: html})
        with pytest.raises(ValueError):
            scrape_fight(session, FIGHT_URL, "2018-10-06")
