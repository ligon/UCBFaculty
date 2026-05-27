"""
Committee metadata and convenience helpers.
"""
from __future__ import annotations

import logging
from functools import cache
from pathlib import Path
from typing import Optional, List

import pandas as pd

from . import datadir, spelling

logger = logging.getLogger(__name__)


@cache
def committee_metadata() -> dict:
    """Load committee metadata from ``committee_metadata.yaml``.

    Returns
    -------
    dict
        Mapping of committee acronyms to metadata dicts.  Returns an
        empty dict if the file is missing or cannot be parsed.
    """
    path = Path(datadir) / "committee_metadata.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("Failed to load committee metadata from %s: %s", path, exc)
        return {}


def committees_with_divco_seat() -> list[str]:
    """Return sorted list of committee acronyms that have a DIVCO seat.

    Returns
    -------
    list[str]
        Committee acronyms where ``has_divco_seat`` is truthy in
        the metadata.
    """
    meta = committee_metadata()
    return sorted([key for key, payload in meta.items() if payload.get("has_divco_seat")])


def committees_from_metadata() -> list[str]:
    """Return sorted list of committee acronyms present in committee_metadata.yaml."""
    return sorted(committee_metadata().keys())


class Committee:
    """Simple representation of a Senate committee by acronym.

    Static attributes pulled from metadata when available.
    """

    def __init__(self, acronym: str):
        """Create a Committee by its acronym.

        Parameters
        ----------
        acronym : str
            Committee acronym (e.g. ``'CAPBD'``).  Normalised via
            the project spelling dictionary on construction.
        """
        self.acronym = spelling.spell(acronym)
        self._metadata = committee_metadata().get(self.acronym, {})

    @property
    def name(self) -> str:
        return self._metadata.get("name") or self.acronym

    @property
    def has_divco_seat(self) -> bool:
        return bool(self._metadata.get("has_divco_seat", False))

    def __repr__(self) -> str:
        return f"Committee({self.acronym})"

    def membership(self, year: Optional[object] = None, as_df: bool = False):
        """Return membership for this committee.

        Parameters
        ----------
        year : str | int | tuple | None
            Year selector:

            - ``None`` — current academic year.
            - ``'all'`` — every year on record.
            - An academic-year string/int — that single year.
            - A ``(start, end)`` tuple — year range (inclusive).
        as_df : bool
            When ``True``, return a :class:`pandas.DataFrame`;
            otherwise return a list of
            :class:`~ucbfaculty.faculty.FacultyMember` objects.

        Returns
        -------
        pandas.DataFrame | list[FacultyMember]
            Membership records for this committee.
        """
        from . import faculty as fac
        from . import utils

        start_year = end_year = None
        if year is None:
            year = fac.get_current_year()
        if isinstance(year, str) and year.lower() == "all":
            start_year = end_year = None
        elif isinstance(year, (list, tuple)) and len(year) == 2:
            start_year, _ = fac.normalize_year_input(year[0])
            end_year, _ = fac.normalize_year_input(year[1])
        else:
            start_year, end_year = fac.normalize_year_input(year)

        df = fac.committee_membership(
            committee=self.acronym, start_year=start_year, end_year=end_year
        )
        if as_df:
            return df
        return utils.df_to_faculties(df)

    def volunteers(self, year: Optional[object] = None, as_df: bool = False, data_fn: Optional[str] = None):
        """
        Return volunteers for this committee.

        year: None -> current service year (uses volunteers from prior year),
              'all' -> stack all available volunteer years,
              or an explicit academic year (int/str).
        as_df: when True, return a DataFrame (indexed by Email when available);
               otherwise return a list of FacultyMember objects.
        """
        from . import faculty as fac
        from . import utils
        if data_fn is None:
            data_fn = fac.VOLUNTEER_WORKBOOK

        def _service_to_volunteer_year(service_year):
            start, _ = fac.normalize_year_input(service_year)
            if start is None:
                return None
            vol_start = start - 1
            return f"{vol_start}-{(vol_start + 1) % 100:02d}"

        service_year = fac.get_current_year() if year is None else year
        vol_years: List[str] = []
        service_map = {}

        if isinstance(service_year, str) and service_year.lower() == "all":
            # Load all sheets in the volunteer workbook (supports encrypted files).
            for sheet in fac._volunteer_years(data_fn):
                vol_years.append(sheet)
                svc_start, _ = fac.normalize_year_input(sheet)
                if svc_start is not None:
                    svc_year = f"{svc_start + 1}-{(svc_start + 2) % 100:02d}"
                    service_map[sheet] = svc_year
        else:
            vol_year = _service_to_volunteer_year(service_year)
            if vol_year:
                vol_years = [vol_year]
                service_map[vol_year] = fac.normalize_year_input(service_year)[1]

        frames = []
        for vy in vol_years:
            try:
                df = fac.volunteers(year=vy, data_fn=data_fn)
            except Exception as exc:
                logger.debug("Failed to load volunteers for year %s: %s", vy, exc)
                continue
            mask = False
            for col in ("Committee", "Choice1", "Choice2", "Choice3"):
                if col in df.columns:
                    mask = mask | (df[col] == self.acronym)
            if not isinstance(mask, pd.Series):
                continue
            subset = df[mask].copy()
            if subset.empty:
                continue
            subset["VolunteerYear"] = vy
            subset["ServiceYear"] = service_map.get(vy)
            frames.append(subset)

        if not frames:
            return pd.DataFrame() if as_df else []

        # fac.volunteers() returns Email-indexed frames (no Email column).
        # Promote Email to a column on each frame so ignore_index=True does
        # not silently discard identity, then re-establish Email as the
        # index while keeping the column for downstream joins.
        frames = [
            f.reset_index() if f.index.name == "Email" else f
            for f in frames
        ]
        out_df = pd.concat(frames, ignore_index=True, sort=False)
        if "Email" in out_df.columns:
            out_df = out_df.set_index("Email", drop=False)
        if as_df:
            return out_df
        return utils.df_to_faculties(out_df.reset_index(drop=True))

    def chairs(self, year: Optional[object] = None, as_df: bool = False):
        """
        Return chair(s) for this committee. Default is current year.

        year: None (current year), 'all', academic-year string/int, or (start, end) tuple.
        as_df: when True, return a DataFrame; otherwise return a list of FacultyMember.
        """
        from . import faculty as fac
        from . import utils
        df = self.membership(year=year, as_df=True)
        if df.empty:
            return df if as_df else []
        role_cols = [c for c in df.columns if c.lower() in {"role", "appointment"}]
        if role_cols:
            mask = False
            for col in role_cols:
                mask = mask | df[col].astype(str).str.lower().str.contains('chair', na=False)
            df = df[mask]
        if as_df:
            return df
        return utils.df_to_faculties(df.reset_index(drop=True))


def clear_committee_caches() -> None:
    """Clear cached committee metadata."""
    committee_metadata.cache_clear()


__all__ = ["Committee", "committee_metadata", "committees_with_divco_seat", "clear_committee_caches"]
