"""Tests for reindex_to_email() and its helpers."""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from ucbfaculty.merge_features import (
    _is_email_column,
    _is_name_column,
    _detect_name_format,
    _match_emails,
    _match_names,
    reindex_to_email,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def canonical_mapping():
    """Mock canonical FullName mapping: Email -> FullName."""
    return pd.Series(
        {
            'alice@berkeley.edu': 'Alice Smith',
            'bob@berkeley.edu': 'Bob Jones',
            'carol@berkeley.edu': 'Carol Williams',
            'dave@berkeley.edu': 'Dave Brown',
        },
        name='FullName',
    )


@pytest.fixture
def canonical_emails(canonical_mapping):
    return set(canonical_mapping.index)


def _make_mock_dict():
    """Create a mock spelling Dictionary whose inverse_d returns key if missing."""
    d = MagicMock()
    # Mimic StrippedDefaultDict: returns the key itself when not found
    inverse = {}

    class _FakeInverse(dict):
        def __getitem__(self, key):
            return self.get(key, key)

    d.inverse_d = _FakeInverse()
    return d


@pytest.fixture
def mock_email_dict():
    return _make_mock_dict()


@pytest.fixture
def mock_name_dict():
    return _make_mock_dict()


# ===========================================================================
# Column-detection helpers
# ===========================================================================

class TestIsEmailColumn:
    def test_positive(self):
        s = pd.Series(['a@b.com', 'c@d.edu', 'e@f.org', None])
        assert _is_email_column(s)

    def test_negative_names(self):
        s = pd.Series(['Alice Smith', 'Bob Jones', 'Carol Williams'])
        assert not _is_email_column(s)

    def test_negative_numbers(self):
        s = pd.Series([1, 2, 3])
        assert not _is_email_column(s)

    def test_empty(self):
        s = pd.Series([], dtype='object')
        assert not _is_email_column(s)

    def test_all_null(self):
        s = pd.Series([None, None, None])
        assert not _is_email_column(s)


class TestIsNameColumn:
    def test_positive(self):
        s = pd.Series(['Alice Smith', 'Bob Jones', 'Carol Williams', 'Dave Brown'])
        assert _is_name_column(s)

    def test_positive_last_first(self):
        s = pd.Series(['Smith, Alice', 'Jones, Bob', 'Williams, Carol'])
        assert _is_name_column(s)

    def test_negative_emails(self):
        s = pd.Series(['a@b.com', 'c@d.edu', 'e@f.org'])
        assert not _is_name_column(s)

    def test_negative_single_word(self):
        s = pd.Series(['Economics', 'Physics', 'Chemistry', 'Biology'])
        assert not _is_name_column(s)

    def test_negative_numbers(self):
        s = pd.Series(['12345', '67890', '11111'])
        assert not _is_name_column(s)

    def test_empty(self):
        s = pd.Series([], dtype='object')
        assert not _is_name_column(s)


class TestDetectNameFormat:
    def test_last_first(self):
        s = pd.Series(['Smith, Alice', 'Jones, Bob', 'Williams, Carol'])
        assert _detect_name_format(s) == 'last_first'

    def test_first_last(self):
        s = pd.Series(['Alice Smith', 'Bob Jones', 'Carol Williams'])
        assert _detect_name_format(s) == 'first_last'

    def test_empty(self):
        s = pd.Series([], dtype='object')
        assert _detect_name_format(s) == 'first_last'

    def test_mixed_defaults_to_majority(self):
        # 2 out of 3 have commas
        s = pd.Series(['Smith, Alice', 'Jones, Bob', 'Carol Williams'])
        assert _detect_name_format(s) == 'last_first'


# ===========================================================================
# Matching helpers
# ===========================================================================

class TestMatchEmails:
    def test_exact_match(self, canonical_emails, mock_email_dict):
        s = pd.Series(['alice@berkeley.edu', 'unknown@gmail.com'])
        result = _match_emails(s, canonical_emails, mock_email_dict)
        assert result.iloc[0] == 'alice@berkeley.edu'
        assert pd.isna(result.iloc[1])

    def test_local_part_match(self, canonical_emails, mock_email_dict):
        s = pd.Series(['alice@econ.berkeley.edu', 'bob@gmail.com'])
        result = _match_emails(s, canonical_emails, mock_email_dict)
        assert result.iloc[0] == 'alice@berkeley.edu'
        assert result.iloc[1] == 'bob@berkeley.edu'

    def test_case_insensitive(self, canonical_emails, mock_email_dict):
        s = pd.Series(['Alice@Berkeley.edu'])
        result = _match_emails(s, canonical_emails, mock_email_dict)
        assert result.iloc[0] == 'alice@berkeley.edu'

    def test_null_handling(self, canonical_emails, mock_email_dict):
        s = pd.Series([None, pd.NA, 'alice@berkeley.edu'])
        result = _match_emails(s, canonical_emails, mock_email_dict)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 'alice@berkeley.edu'

    def test_no_match(self, canonical_emails, mock_email_dict):
        s = pd.Series(['nobody@nowhere.com', 'stranger@gmail.com'])
        result = _match_emails(s, canonical_emails, mock_email_dict)
        assert result.isna().all()


class TestMatchNames:
    def test_exact_match(self, canonical_mapping, mock_name_dict):
        s = pd.Series(['Alice Smith', 'Bob Jones', 'Unknown Person'])
        result = _match_names(s, canonical_mapping, mock_name_dict)
        assert result.iloc[0] == 'alice@berkeley.edu'
        assert result.iloc[1] == 'bob@berkeley.edu'
        assert pd.isna(result.iloc[2])

    def test_null_handling(self, canonical_mapping, mock_name_dict):
        s = pd.Series([None, '', 'Alice Smith'])
        result = _match_names(s, canonical_mapping, mock_name_dict)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 'alice@berkeley.edu'

    def test_no_match(self, canonical_mapping, mock_name_dict):
        s = pd.Series(['Nobody Here', 'Stranger Danger'])
        result = _match_names(s, canonical_mapping, mock_name_dict)
        assert result.isna().all()


# ===========================================================================
# Main function: reindex_to_email
# ===========================================================================

def _patch_canonical(canonical_mapping):
    """Return context managers patching _fullname_email_mapping + dictionaries.

    The function imports these lazily, so we patch at the source modules:
    - _fullname_email_mapping  lives in ucbfaculty.new_feature
    - FACULTY_NAME_DICTIONARY  lives in ucbfaculty
    - EMAIL_DICTIONARY         lives in ucbfaculty
    """
    name_dict = _make_mock_dict()
    email_dict = _make_mock_dict()

    return (
        patch(
            'ucbfaculty.new_feature._fullname_email_mapping',
            return_value=canonical_mapping,
        ),
        patch(
            'ucbfaculty.FACULTY_NAME_DICTIONARY',
            name_dict,
        ),
        patch(
            'ucbfaculty.EMAIL_DICTIONARY',
            email_dict,
        ),
    )


class TestReindexToEmail:
    def test_explicit_name_col(self, canonical_mapping):
        wild = pd.DataFrame({
            'Name': ['Alice Smith', 'Bob Jones', 'Stranger Person'],
            'Score': [95, 88, 72],
        })
        patches = _patch_canonical(canonical_mapping)
        with patches[0], patches[1], patches[2]:
            result = reindex_to_email(wild, name_col='Name')
        assert result.index.name == 'Email'
        assert 'alice@berkeley.edu' in result.index
        assert 'bob@berkeley.edu' in result.index
        assert len(result) == 2
        assert 'Score' in result.columns

    def test_explicit_email_col(self, canonical_mapping):
        wild = pd.DataFrame({
            'Contact': ['alice@berkeley.edu', 'nobody@gmail.com', 'bob@berkeley.edu'],
            'Role': ['Chair', 'Member', 'Secretary'],
        })
        patches = _patch_canonical(canonical_mapping)
        with patches[0], patches[1], patches[2]:
            result = reindex_to_email(wild, email_col='Contact')
        assert 'alice@berkeley.edu' in result.index
        assert 'bob@berkeley.edu' in result.index
        assert len(result) == 2

    def test_local_part_email_matching(self, canonical_mapping):
        wild = pd.DataFrame({
            'Email': ['alice@econ.berkeley.edu', 'bob@gmail.com', 'unknown@yahoo.com'],
            'Data': [1, 2, 3],
        })
        patches = _patch_canonical(canonical_mapping)
        with patches[0], patches[1], patches[2]:
            result = reindex_to_email(wild, email_col='Email')
        assert 'alice@berkeley.edu' in result.index
        assert 'bob@berkeley.edu' in result.index
        assert len(result) == 2

    def test_split_name_cols(self, canonical_mapping):
        wild = pd.DataFrame({
            'First': ['Alice', 'Bob', 'Stranger'],
            'Last': ['Smith', 'Jones', 'Person'],
            'Rank': [1, 2, 3],
        })
        patches = _patch_canonical(canonical_mapping)
        with patches[0], patches[1], patches[2]:
            result = reindex_to_email(wild, name_col=('First', 'Last'))
        assert 'alice@berkeley.edu' in result.index
        assert 'bob@berkeley.edu' in result.index
        assert len(result) == 2

    def test_last_first_format(self, canonical_mapping):
        wild = pd.DataFrame({
            'Member': ['Smith, Alice', 'Jones, Bob', 'Person, Stranger'],
            'Year': [2024, 2024, 2024],
        })
        patches = _patch_canonical(canonical_mapping)
        with patches[0], patches[1], patches[2]:
            result = reindex_to_email(wild, name_col='Member')
        assert 'alice@berkeley.edu' in result.index
        assert 'bob@berkeley.edu' in result.index
        assert len(result) == 2

    def test_auto_detect_email(self, canonical_mapping):
        wild = pd.DataFrame({
            'contact_email': ['alice@berkeley.edu', 'nobody@gmail.com'],
            'notes': ['active', 'inactive'],
        })
        patches = _patch_canonical(canonical_mapping)
        with patches[0], patches[1], patches[2]:
            result = reindex_to_email(wild)
        assert 'alice@berkeley.edu' in result.index
        assert len(result) == 1

    def test_auto_detect_firstname_lastname(self, canonical_mapping):
        wild = pd.DataFrame({
            'FirstName': ['Alice', 'Stranger'],
            'LastName': ['Smith', 'Person'],
            'Dept': ['CS', 'Econ'],
        })
        patches = _patch_canonical(canonical_mapping)
        with patches[0], patches[1], patches[2]:
            result = reindex_to_email(wild)
        assert 'alice@berkeley.edu' in result.index
        assert len(result) == 1

    def test_email_precedence_over_name(self, canonical_mapping):
        """When both email and name match, email takes precedence."""
        wild = pd.DataFrame({
            'Email': ['alice@berkeley.edu'],
            'Name': ['Alice Smith'],
        })
        patches = _patch_canonical(canonical_mapping)
        with patches[0], patches[1], patches[2]:
            result = reindex_to_email(wild, email_col='Email', name_col='Name')
        assert result.index[0] == 'alice@berkeley.edu'
        assert len(result) == 1

    def test_no_matches_empty(self, canonical_mapping):
        wild = pd.DataFrame({
            'Name': ['Stranger Person', 'Nobody Nowhere'],
            'Score': [1, 2],
        })
        patches = _patch_canonical(canonical_mapping)
        with patches[0], patches[1], patches[2]:
            result = reindex_to_email(wild, name_col='Name')
        assert len(result) == 0
        assert result.index.name == 'Email'

    def test_scholarly_society_minority(self, canonical_mapping):
        """User story: scholarly society, minority are Berkeley faculty."""
        names = ['Alice Smith', 'Bob Jones'] + [f'Person {i}' for i in range(20)]
        wild = pd.DataFrame({
            'Member': names,
            'MemberID': list(range(len(names))),
        })
        patches = _patch_canonical(canonical_mapping)
        with patches[0], patches[1], patches[2]:
            result = reindex_to_email(wild, name_col='Member')
        assert len(result) == 2
        assert 'alice@berkeley.edu' in result.index
        assert 'bob@berkeley.edu' in result.index

    def test_advisor_column_not_matched(self, canonical_mapping):
        """User story: faculty + advisor columns, only match specified one."""
        wild = pd.DataFrame({
            'Faculty': ['Alice Smith', 'Stranger Person'],
            'Advisor': ['Bob Jones', 'Carol Williams'],
        })
        patches = _patch_canonical(canonical_mapping)
        with patches[0], patches[1], patches[2]:
            result = reindex_to_email(wild, name_col='Faculty')
        # Only Alice should match (from Faculty column)
        assert len(result) == 1
        assert 'alice@berkeley.edu' in result.index

    def test_invalid_name_col_raises(self, canonical_mapping):
        wild = pd.DataFrame({'A': [1]})
        patches = _patch_canonical(canonical_mapping)
        with patches[0], patches[1], patches[2]:
            with pytest.raises(ValueError, match="name_col='Missing'"):
                reindex_to_email(wild, name_col='Missing')

    def test_invalid_email_col_raises(self, canonical_mapping):
        wild = pd.DataFrame({'A': [1]})
        patches = _patch_canonical(canonical_mapping)
        with patches[0], patches[1], patches[2]:
            with pytest.raises(ValueError, match="email_col='Missing'"):
                reindex_to_email(wild, email_col='Missing')

    def test_preserves_columns(self, canonical_mapping):
        """All original columns should be preserved in output."""
        wild = pd.DataFrame({
            'Name': ['Alice Smith'],
            'Score': [95],
            'Year': [2024],
            'Notes': ['good'],
        })
        patches = _patch_canonical(canonical_mapping)
        with patches[0], patches[1], patches[2]:
            result = reindex_to_email(wild, name_col='Name')
        assert set(result.columns) == {'Name', 'Score', 'Year', 'Notes'}
