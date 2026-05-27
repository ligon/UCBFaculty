import json
import pytest

from ucbfaculty.spelling import identify_case, StrippedDefaultDict, Dictionary


class TestIdentifyCase:
    def test_uppercase(self):
        assert identify_case("HELLO") == "uppercase"

    def test_lowercase(self):
        assert identify_case("hello") == "lowercase"

    def test_titlecase(self):
        assert identify_case("Hello") == "titlecase"

    def test_camelcase(self):
        assert identify_case("helloWorld") == "camelcase"

    def test_unknown(self):
        assert identify_case("HELLo") == "unknown"


class TestStrippedDefaultDict:
    def test_missing_key_returns_stripped(self):
        d = StrippedDefaultDict()
        assert d["  foo  "] == "foo"

    def test_existing_key_returned(self):
        d = StrippedDefaultDict()
        d["key"] = "val"
        assert d["key"] == "val"


class TestDictionary:
    @pytest.fixture()
    def spelling_file(self, tmp_path):
        data = {"Robert": ["Bob", "Rob"], "William": ["Bill", "Will"]}
        p = tmp_path / "spelling.json"
        p.write_text(json.dumps(data))
        return p

    @pytest.fixture()
    def dic(self, spelling_file):
        return Dictionary(spelling_file)

    def test_init_loads_json(self, dic):
        assert "Robert" in dic
        assert dic["Robert"] == ["Bob", "Rob"]

    def test_inverse_lookup(self, dic):
        assert dic.inverse_d["Bob"] == "Robert"
        assert dic.inverse_d["Bob".title()] == "Robert"

    def test_spell_known(self, dic):
        assert dic.spell("Bob") == "Robert"

    def test_spell_unknown_returns_titlecased(self, dic):
        # Per the docstring, an unmatched word falls back to the stripped
        # title-cased form (preserve_case=False is the default).
        assert dic.spell("unknown") == "Unknown"
        assert dic.spell("  whitespace  ") == "Whitespace"

    def test_spell_preserve_case_title(self, dic):
        # "Bob" is in inverse_d -> "Robert"; titlecase preserved
        assert dic.spell("Bob", preserve_case=True) == "Robert"

    def test_spell_preserve_case_unknown_word(self, dic):
        # Not in inverse_d; StrippedDefaultDict returns stripped key
        assert dic.spell("xyzzy", preserve_case=True) == "xyzzy"

    def test_add(self, dic):
        dic.add("Robert", "Bobby")
        assert "Bobby" in dic["Robert"]
        assert dic.inverse_d["Bobby"] == "Robert"

    def test_add_new_key(self, dic):
        dic.add("NewName", "nick")
        assert "nick" in dic["NewName"]
        assert dic.inverse_d["nick"] == "NewName"

    def test_spellings_known(self, dic):
        result = dic.spellings("Bob")
        assert "Bob" in result

    def test_spellings_unknown(self, dic):
        # defaultdict(list) returns [] for unknown keys; KeyError path unreachable
        assert dic.spellings("nobody") == []

    def test_persist_roundtrip(self, dic, tmp_path):
        out = tmp_path / "out.json"
        dic.persist(out)
        reloaded = Dictionary(out)
        assert reloaded["Robert"] == dic["Robert"]


class TestGetSpellingDictionaryFallback:
    """Behaviour of ucbfaculty.get_spelling_dictionary path resolution."""

    def _write_config(self, tmp_path, **entries):
        import yaml
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(entries))
        return cfg_path

    def test_falls_back_to_packaged_when_default_missing(self, tmp_path):
        """No config entry and no file on disk => packaged copy wins."""
        from pathlib import Path
        import ucbfaculty
        from ucbfaculty import get_spelling_dictionary

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = self._write_config(tmp_path, data_directory=str(data_dir))

        resolved = get_spelling_dictionary(target=cfg)

        expected = Path(ucbfaculty.__file__).resolve().parent / "spelling_dictionary.json"
        assert resolved == expected
        assert resolved.exists()
        loaded = json.loads(resolved.read_text())
        assert "FirstName" in loaded and "LastName" in loaded

    def test_packaged_fallback_loadable_by_dictionary(self, tmp_path):
        """The packaged dictionary must round-trip through the Dictionary
        class so a fresh install can normalise common column headers."""
        from ucbfaculty import get_spelling_dictionary

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = self._write_config(tmp_path, data_directory=str(data_dir))

        path = get_spelling_dictionary(target=cfg)
        d = Dictionary(path)
        # Header normalisation: 'First Name' is a known variant of 'FirstName'.
        assert d.spell("First Name") == "FirstName"

    def test_explicit_config_path_is_respected_even_if_missing(self, tmp_path):
        """A user-supplied path wins over the fallback, so typos surface
        clearly rather than silently using the packaged default."""
        from ucbfaculty import get_spelling_dictionary

        explicit = tmp_path / "nowhere" / "spelling_dictionary.json"
        cfg = self._write_config(
            tmp_path,
            data_directory=str(tmp_path),
            spelling_dictionary=str(explicit),
        )

        resolved = get_spelling_dictionary(target=cfg)
        assert resolved == explicit
        assert not resolved.exists()

    def test_no_fallback_for_sensitive_keys(self, tmp_path):
        """faculty_spellings and email_dictionary contain real names /
        emails and must not silently fall back to a packaged default."""
        from ucbfaculty import get_spelling_dictionary

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = self._write_config(tmp_path, data_directory=str(data_dir))

        for key in ("faculty_spellings", "email_dictionary"):
            resolved = get_spelling_dictionary(key=key, target=cfg)
            assert resolved == data_dir / f"{key}.json"
            assert not resolved.exists()
