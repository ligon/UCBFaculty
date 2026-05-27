import importlib
import logging
import subprocess
import warnings
from io import BytesIO
from functools import lru_cache
from typing import Optional
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

from ligonlibrary import dataframes
from ucbfaculty import datadir
from ucbfaculty.security import get_effective_passphrase


class DecryptionError(RuntimeError):
    """Raised when an encrypted file cannot be decrypted."""

from . import spelling

def _fallback_get_dataframe(path: Path, **kwargs):
    """Basic dataframe loader used if ligonlibrary lacks get_dataframe."""
    sheet = kwargs.pop('sheet', None)
    suffix = path.suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(path, **kwargs)
    elif suffix in {'.xlsx', '.xlsm', '.xls'}:
        if sheet is not None:
            kwargs.setdefault('sheet_name', sheet)
        return pd.read_excel(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type for {path}")


def _parse_gpg_error(stderr: str) -> tuple[Optional[str], Optional[str]]:
    """Extract (status, key_id) from GPG status output if available."""
    status = None
    key = None
    for line in stderr.splitlines():
        line = line.strip()
        if not line.startswith("[GNUPG:]"):
            continue
        parts = line[len("[GNUPG:]"):].strip().split()
        if not parts:
            continue
        code = parts[0]
        if code == "ENC_TO" and len(parts) > 1:
            key = parts[1]
        elif code in {"BAD_PASSPHRASE", "NO_SECKEY", "ERR"}:
            status = code
            if len(parts) > 1:
                key = parts[1]
    return status, key


@lru_cache(maxsize=32)
def _cached_gpg_decrypt(path: Path) -> bytes:
    """Cached wrapper around _gpg_decrypt to avoid repeated gpg calls."""
    return _gpg_decrypt(path)


def _gpg_decrypt(path: Path) -> bytes:
    """
    Return decrypted bytes for a GPG-encrypted file.

    Uses a layered approach to passphrase resolution:
    1. If passphrase is set in memory, use loopback mode directly
    2. Otherwise try gpg-agent (may invoke pinentry for CLI/interactive use)

    Raises DecryptionError if decryption fails.
    """
    # Layer 1: Check for in-memory passphrase first
    # If we have one, skip gpg-agent and go straight to loopback mode
    # (this avoids pinentry prompts when using the Panel UI)
    passphrase = get_effective_passphrase()

    if not passphrase:
        # Layer 2: No passphrase set - try gpg-agent (may use pinentry for CLI)
        agent_result = _try_gpg_agent_decrypt(path)
        if agent_result is not None:
            return agent_result

    # Layer 3: Use loopback mode with passphrase (if available)
    command = [
        "gpg",
        "--batch",
        "--quiet",
        "--status-fd=2",
        "--decrypt",
    ]
    kwargs = {
        "check": False,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if passphrase:
        command.extend(["--pinentry-mode", "loopback", "--passphrase-fd", "0"])
        kwargs["input"] = passphrase.encode("utf-8")
    command.append(str(path))

    result = subprocess.run(command, **kwargs)
    stderr = result.stderr.decode(errors="ignore").strip()
    if result.returncode != 0:
        status, key = _parse_gpg_error(stderr)
        if status == "BAD_PASSPHRASE" and key:
            msg = f"Incorrect passphrase for key {key}"
        elif status == "NO_SECKEY" and key:
            msg = f"No secret key available for recipient {key}"
        else:
            # Provide helpful guidance
            hint = (
                "\n\nHint: Use unlock_rawdata() to set your passphrase:\n"
                "  from ucbfaculty.security import unlock_rawdata\n"
                "  success, msg = unlock_rawdata('your-passphrase')"
            )
            msg = (stderr or "unknown error") + hint
        raise DecryptionError(f"Failed to decrypt {path}: {msg}")
    return result.stdout


def _try_gpg_agent_decrypt(path: Path) -> Optional[bytes]:
    """
    Attempt decryption using gpg-agent's cached passphrase.

    Returns decrypted bytes on success, None if agent doesn't have passphrase.
    May invoke pinentry if gpg-agent is configured to prompt.
    """
    command = [
        "gpg",
        "--batch",
        "--quiet",
        "--decrypt",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception as exc:
        logger.debug("gpg-agent decryption failed: %s", exc)
        return None


def _load_encrypted_dataframe(path: Path, **kwargs):
    """Decrypt a GPG-encrypted file and load it into a DataFrame."""
    if len(path.suffixes) < 2:
        raise ValueError(f"Encrypted file {path} must include original suffix before .gpg/.pgp")

    # Determine original file extension (before .gpg/.pgp)
    base_suffix = path.suffixes[-2].lower()
    decrypted = _cached_gpg_decrypt(path)
    buffer = BytesIO(decrypted)

    if base_suffix in {".csv", ".txt"}:
        return pd.read_csv(buffer, **kwargs)
    elif base_suffix in {".xlsx", ".xlsm", ".xls"}:
        sheet = kwargs.pop("sheet", None)
        if sheet is not None:
            kwargs.setdefault("sheet_name", sheet)
        return pd.read_excel(buffer, **kwargs)
    else:
        raise ValueError(f"Unsupported encrypted file type for {path}")

_DATAFRAME_CACHE: dict = {}


def clear_dataframe_cache() -> None:
    """Discard all cached DataFrames loaded by :func:`get_dataframe`."""
    _DATAFRAME_CACHE.clear()


def _dataframe_cache_key(source: Path, sheet_value, colspell_path, options: dict):
    """Build a hashable cache key from the call parameters."""
    try:
        items = tuple(sorted(options.items()))
    except TypeError:
        items = tuple(sorted((k, str(v)) for k, v in options.items()))
    return (source.resolve(), sheet_value, colspell_path, items)


def get_dataframe(fn, correct_column_spellings=True, **kwargs):
    """Load a data file into a pandas DataFrame.

    Supports CSV, Excel, and GPG-encrypted variants.  Results are
    cached in-process; call :func:`clear_dataframe_cache` to reset.

    Parameters
    ----------
    fn : str | Path
        File name or path.  Relative paths are resolved against the
        configured data directory.
    correct_column_spellings : bool | str
        When ``True`` (default), normalise column names through the
        project spelling dictionary.  A string value is interpreted
        as a path to an alternate dictionary file.  ``False`` skips
        column-name correction.
    **kwargs
        Forwarded to the underlying loader (e.g. ``sheet`` for Excel
        files).

    Returns
    -------
    pandas.DataFrame | dict[str, pandas.DataFrame]
        A single DataFrame for CSV/single-sheet files, or a dict of
        ``{sheet_name: DataFrame}`` for multi-sheet Excel files when
        no specific sheet is requested.
    """
    if correct_column_spellings:
        if isinstance(correct_column_spellings, str):
            colspell = spelling.Dictionary(correct_column_spellings)
        else:
            colspell = spelling.Dictionary()
        colspell_path = getattr(colspell, "spelling_file", None)
    else:
        colspell = None
        colspell_path = None

    source = Path(fn)
    if not source.is_absolute():
        source = datadir / source

    loader = getattr(dataframes, 'get_dataframe', None)
    if loader is None:
        loader = _fallback_get_dataframe
    kwargs_copy = dict(kwargs)
    sheet_value = kwargs_copy.get("sheet")

    cache_key = _dataframe_cache_key(source, sheet_value, colspell_path, kwargs_copy)
    cached = _DATAFRAME_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    if source.suffix.lower() in {'.gpg', '.pgp'}:
        df = _load_encrypted_dataframe(source, **kwargs_copy)
    else:
        df = loader(source, **kwargs_copy)

    if isinstance(df, dict):  # Different sheets from an xlsx or similar?
        if colspell:
            return {k: v.rename(columns=lambda s: colspell.inverse_d[s.title()]) for k, v in df.items()}
        return df

    renamed = df.rename(columns=lambda s: colspell.inverse_d[s.title()]) if colspell else df
    _DATAFRAME_CACHE[cache_key] = renamed
    return renamed.copy()


def load_coms_tracking(year="2025-26", workbook="COMS Tracking Sheet AY25-26.xlsx.gpg", drop_unmatched=False):
    """Parse the COMS tracking workbook into a normalized dataframe.

    Parameters
    ----------
    year : str
        Academic year represented in the workbook (used in output).
    workbook : str
        Relative path inside RawData for the tracking workbook.
    drop_unmatched : bool
        If True, drop rows that could not be matched to a faculty email.

    Returns
    -------
    pandas.DataFrame
        Normalized dataset with columns:
        Committee, FullName, Department, Title, CurrentCommittee,
        ContinueCurrent, FirstChoice, SecondChoice, ThirdChoice,
        AnyCommittee, CantServe, Email, AcademicYear
    """
    import pandas as pd
    from .faculty import faculty_vitals
    from .spelling import Dictionary

    path = datadir / workbook
    if path.suffix.lower() in {".gpg", ".pgp"}:
        decrypted = _gpg_decrypt(path)
        book = pd.ExcelFile(BytesIO(decrypted))
    else:
        book = pd.ExcelFile(path)

    fac = faculty_vitals().reset_index()
    fac['FullName'] = fac['FirstName'].fillna('') + ' ' + fac['LastName'].fillna('')
    fac_lookup = fac[['FullName', 'Email']]

    speller = Dictionary()

    excluded_sheets = {
        "Any Committee",
        "Emeriti List",
        "Nomination by dean andor dept c",
        "POC Faculty as of 03.2024",
        "Discplines",
        "Systemwide appointments",
    }

    records = []
    for sheet in book.sheet_names:
        if sheet in excluded_sheets:
            continue

        df = book.parse(sheet, header=None)

        header_idx = None
        for idx, row in df.iterrows():
            values = [str(x).strip() if pd.notna(x) else "" for x in row]
            if values[:10] == [
                'FIRST', 'LAST', 'DEPT', 'TITLE', 'CURRENT',
                'CONTINUE', 'FIRST', 'SECOND', 'THIRD', 'ANY'
            ]:
                header_idx = idx
                break

        if header_idx is None:
            continue

        headers = df.iloc[header_idx].fillna('').tolist()
        unique_headers = []
        counts = {}
        for header in headers:
            key = str(header).strip()
            if not key:
                unique_headers.append(None)
                continue
            counts[key] = counts.get(key, 0) + 1
            unique_headers.append(key if counts[key] == 1 else f"{key}_{counts[key]}")

        data = df.iloc[header_idx + 1:].copy()
        data.columns = unique_headers

        required = [
            'FIRST', 'LAST', 'DEPT', 'TITLE', 'CURRENT', 'CONTINUE',
            'FIRST_2', 'SECOND', 'THIRD', 'ANY', "CAN'T SERVE"
        ]
        for col in required:
            if col not in data.columns:
                data[col] = None

        data = data.dropna(subset=['FIRST', 'LAST'], how='all')
        if data.empty:
            continue

        data['FullName'] = (
            data['FIRST'].fillna('').astype(str).str.strip() + ' ' +
            data['LAST'].fillna('').astype(str).str.strip()
        ).str.strip()
        data = data[data['FullName'] != '']
        if data.empty:
            continue

        data['FullName'] = data['FullName'].map(speller.spell)
        data['Committee'] = sheet
        data = data.rename(
            columns={
                'DEPT': 'Department',
                'TITLE': 'Title',
                'CURRENT': 'CurrentCommittee',
                'CONTINUE': 'ContinueCurrent',
                'FIRST_2': 'FirstChoice',
                'SECOND': 'SecondChoice',
                'THIRD': 'ThirdChoice',
                'ANY': 'AnyCommitteePreference',
                "CAN'T SERVE": 'CantServe'
            }
        )
        data = data[
            [
                'Committee', 'FullName', 'Department', 'Title',
                'CurrentCommittee', 'ContinueCurrent', 'FirstChoice',
                'SecondChoice', 'ThirdChoice', 'AnyCommitteePreference',
                'CantServe'
            ]
        ]

        records.append(data)

    if not records:
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)
    result = result.merge(fac_lookup, how='left', on='FullName')
    if drop_unmatched:
        result = result[result['Email'].notna()].copy()
    result['AcademicYear'] = year
    return result

import os
from functools import cached_property
from git import Repo
from ucbfaculty import datadir

class RawData:
    """
    A class to handle raw data files with support for property access, indexing, and data transformations.

    This class manages loading data files, setting indexes, and providing access to data columns
    with optional transformations.
    """
    __slots__ = ('Provides',
                 '_FileName','_AccessionDate','_IndexVars','_data')
    __properties__ = ('FileName', 'AccessionDate','IndexVars','data')

    def __init__(self, **kwargs):
        """
        Initialize a RawData object with optional attributes.

        Args:
            **kwargs: Keyword arguments corresponding to class properties.
                      Can include FileName, AccessionDate, IndexVars, and data.
        """
        for attr in self.__slots__:
            setattr(self,attr,kwargs.get(attr,None))

        for attr in self.__properties__:
            setattr(self,'_'+attr,kwargs.get(attr,None))

        self.Provides = {}

    @property
    def AccessionDate(self):
        """
        Get the accession date of the file.

        Returns:
            str: The accession date in 'YYYY-MM-DD' format if available, None otherwise.
                 If not explicitly set, attempts to retrieve from git commit history.
        """
        if self._AccessionDate is not None:
            return self._AccessionDate

        try:
            repo = Repo(search_parent_directories=True)
            commits = list(repo.iter_commits(paths=self.FileName))
            if commits:
                # The first commit in the list is the latest modification date
                return commits[0].committed_datetime.strftime('%Y-%m-%d')
        except Exception as exc:
            logger.debug("Failed to resolve git modification date for %s: %s", self.FileName, exc)

        return None

    @AccessionDate.setter
    def AccessionDate(self, value):
        """
        Set the accession date.

        Args:
            value (str): Date string to parse and store as the accession date.

        Raises:
            ValueError: If the provided date string cannot be parsed.
        """
        try:
            from dateutil import parser
            date_obj = parser.parse(value)
            self._AccessionDate = date_obj.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            raise ValueError("AccessionDate must be a valid date string.")

    @property
    def FileName(self):
        """
        Get the filename of the data source.

        Returns:
            str or tuple: The filename or (filename, sheet) tuple for Excel files.
        """
        return self._FileName

    @FileName.setter
    def FileName(self, value):
        """
        Set the filename of the data source.

        Args:
            value (str or tuple): Filename as string or (filename, sheet) tuple for Excel files.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if isinstance(value,(list,tuple)):
            fn,sheet = value
        else:
            fn = value
        if not os.path.exists(fn):
            raise FileNotFoundError(f"The file '{fn}' does not exist.")
        self._FileName = value

    @property
    def data(self):
        """
        Get the data as a pandas DataFrame, loading it if not already loaded.

        Returns:
            pandas.DataFrame: The loaded data with proper indexing.
        """
        if self._data is None:
            if isinstance(self.FileName, (list, tuple)):
                fn, sheet = self.FileName
                self._data = get_dataframe(fn, sheet=sheet)
            else:
                df = get_dataframe(self.FileName)
                if isinstance(df,dict):
                    raise ValueError(f"{self.FileName} has data in {','.join(df.keys())}.  Specify one.")
                else:
                    self._data = df

            self.Provides.update({col: col for col
                             in self._data.columns.difference(list(self.Provides.keys()))})

        self.set_index()

        return self._data

    @property
    def IndexVars(self):
        """
        Get the variables used for indexing the data.

        Returns:
            list or str: Column names to use as index.
        """
        return self._IndexVars

    @IndexVars.setter
    def IndexVars(self, value):
        """
        Set the variables to use for indexing the data.

        Setting this property stores the value and re-applies the index
        if data has already been loaded.

        Args:
            value (list or str): Column names to use as index.
        """
        self._IndexVars = value
        if hasattr(self, '_data') and self._data is not None:
            self.set_index()

    def set_index(self, indices=None):
        """
        Set the index of the data DataFrame.

        Args:
            indices (list, str, optional): Column names to use as index.
                                          If None, uses self.IndexVars.
        """
        if not hasattr(self, '_data') or self._data is None:
            self.data  # Initialize self._data

        # Normalize indices to a list, falling back to self.IndexVars
        if isinstance(indices, str):
            indices = [indices]
        if indices is not None:
            # Store without triggering the setter's recursive set_index call
            self._IndexVars = indices
        else:
            indices = self.IndexVars
            if indices is None:
                return

        # Ensure any provided index columns are computed
        if self._data is not None:
            for i in set(indices).intersection(
                set(self.Provides.keys()).difference(self._data.columns)
            ):
                self.add_item(i)

        # Apply the index
        if isinstance(self._data.index, pd.RangeIndex):
            self._data.set_index(indices, inplace=True)
        else:
            try:
                self._data.reset_index(inplace=True)
            except ValueError:
                self._data.reset_index(inplace=True, drop=True)
            self._data.set_index(indices, inplace=True)

    def __getitem__(self, key, add=False):
        """
        Allow dictionary-like access to data columns, with translation via Provides.

        Args:
            key (str): The column name or alias to access.
            add (bool, optional): If True, adds the result to the data DataFrame.
                                 Defaults to False.

        Returns:
            pandas.Series: The requested data column or transformed data.

        Raises:
            ValueError: If the requested key cannot be found in the data.
        """
        def get_ins(val):
            if val in self._data.columns:
                return self._data[val]
            elif val in self._data.index.names:
            # Convert index to DataFrame for better usability
                if len(self._data.index.names) > 1:  # MultiIndex
                    return self._data.index.to_frame()[val]
                else:  # Single index
                    return self._data.index.to_frame()[val]
            else:
                raise ValueError(f"'{val}' not found in data columns.")

        try:
            val = self.Provides[key]
        except KeyError:
            val = key

        if isinstance(val,str): # We'll just lookup df[val]
             out = get_ins(val)
        elif isinstance(val,(list,tuple)): # A transformation requested on val
            val,fn = val
            out = get_ins(val).apply(fn)
        elif callable(val): # Map the whole dataframe into a series
            out = self._data.reset_index().apply(val,axis=1)
            out.index = self._data.index
            out.name = key

        if add and key not in self._data.columns:
            self._data[key] = out

        return out

    def add_item(self, item):
        """
        Add a new column to the data DataFrame.

        Args:
            item (str): The name of the column to add. The column will be created
                      by processing the data according to the rules in __getitem__.
        """
        self.__getitem__(item, add=True)

    #print(edr.data.head())
