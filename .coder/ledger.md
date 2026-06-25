# Prior-Art Ledger — UCBFaculty (`ucbfaculty`)

> Standing repo ledger (per the `prior-art-ledger` skill): the machinery,
> definitions, and invariants already in force, so a task neither reinvents what
> exists nor contradicts a local definition. Living + git-tracked — edit in place;
> the commit history is the journal. `§N` are the citations used in code comments
> and verification ("OK, anchored on §N"). This DISTILLS the proto-ledger
> `AGENTS.org` (esp. its "Operating Principles" + "Tacit Knowledge") — read that
> for the full narrative before non-trivial faculty-data work.

**Search tier used:** ripgrep over `ucbfaculty/*.py` + the proto-ledger `AGENTS.org`; symbol→test mapped against `tests/`.

## §1 Repo, restated
`ucbfaculty` is a Python package (Poetry) for Berkeley Academic Senate faculty
data: roster **vitals**, **committee** membership/service, **volunteer** rosters,
and YAML-spec'd **Features**. The data itself lives in a **separate `RawData` git
repo** (encrypted `.gpg` CSV/XLSX) pointed at by `data_directory` in
`~/.ucbfaculty/config.yaml` — the package bundles **no** data and decrypts via a
layered GPG model. `ucbfaculty/*.py` is the **authoritative, git-tracked**
implementation (NOT a tangled artifact); `notes.org`/`spelling.org` are design
notes only. Three surfaces: the Python API, a Panel app (`apps/lookup.py`), and a
DuckDB-backed MCP `query` server.

## §2 Existing machinery — REUSE these, don't reinvent
**The headline: identity/name→email matching is SOLVED here. Do not hand-roll it.**
Email (`@berkeley.edu`) is the unique key; a `FacultyMember` is trustworthy only
when `.verified` (exactly one vitals match).

| symbol | verified path:line | what it does | tested? |
|--------|--------------------|--------------|---------|
| `merge_features.reindex_to_email(df,...)` | merge_features.py:509 | **THE canonical matcher** — reindex a *wild* df to canonical Email (email strategies → name dict/fuzzy, default thresh 93) | `tests/test_reindex_to_email.py` |
| `merge_features.ingest_dataframe(path,...)` | merge_features.py:446 | load CSV/XLSX from disk (via `get_dataframe`) → `reindex_to_email` | indirect (reindex tests) |
| `merge_features.fullname_to_email_index(...)` | merge_features.py:187 | name series → canonical Email index (+ `unmatched` list) | `test_merge_features` |
| `merge_features._match_emails` · `_match_names` | merge_features.py:74 · 125 | email 3-strategy (exact / `inverse_d` / local-part) + name dict+fuzzy primitives | via reindex tests |
| `FACULTY_NAME_DICTIONARY` · `EMAIL_DICTIONARY` | `__init__.py`:209 · 211 | module-global `spelling.Dictionary` (defaultdict + `.inverse_d`) variant→canonical maps | `test_spelling` |
| `spelling.Dictionary.spell(word)` · `.persist()` | spelling.py:167 · 153 | resolve variant name → canonical; write dict additions back to JSON | `test_spelling` |
| `faculty.FacultyMember(**kwargs)` | faculty.py:514 (`.verified` set :545/:560) | identity object; `verified=True` **iff** `faculty_vitals` returns exactly ONE match (`match_info.method` ∈ direct/ambiguous/none) | indirect (`test_utils`,`test_queries`) |
| `faculty.faculty_vitals(index_by='Email', **filters)` | faculty.py:183 | the roster loader, Email-indexed, filter by column kwargs | `test_queries` |
| `faculty.volunteers` · `any_committee` · `unexploited_volunteers` | faculty.py:284 · 410 · 436 | volunteer rosters (Email-**indexed**) | `test_faculty_volunteers` |
| `faculty.committee_membership(...)` · `FacultyMember.service_history` | faculty.py:1045 · 881 | committee history by committee / by person | `test_committee_membership`,`test_committee_service` |
| `committee.Committee(acronym)` | committee.py:60 | per-committee API: `.membership()`, `.volunteers()` (:132), `.chairs()` (:210) | `test_committee_*` |
| `new_feature.Feature(name)[year]` | new_feature.py:82 (`__getitem__` :163) | YAML-spec'd single feature → Series; year ∈ `'latest'`/`'history'`/`'2023-24'`/int; reindexed to Email via `EMAIL_DICTIONARY` (:593) | `test_new_feature*`,`test_feature_builder` |
| `new_feature.Features(...)` | new_feature.py:319 | multi-feature DataFrame assembler | `test_new_feature` |
| `rawdata.get_dataframe(fn, correct_column_spellings=True)` | rawdata.py:263 | **THE reader** — CSV/XLSX/GPG, in-process cache, delegates to `ligonlibrary.dataframes` | `test_rawdata_gpg` |
| `rawdata.validate_holdings(...)` | rawdata.py:153 | holdings "doctor": declared (spec) vs present data files | `test_holdings` |
| `security.unlock_rawdata` · `is_unlocked` · `lock_rawdata` | security.py:231 · 296 · 310 | GPG-agent passphrase session mgmt | `test_security` |
| `queries.execute(sql, register=)` · `describe()` · `con()` | queries.py:197 · 256 · 190 | DuckDB zero-copy views (`vitals`, `committees`, per-feature) over the loaders | `test_queries` |
| `get_data_dir()` · `get_spelling_dictionary(key)` | `__init__.py`:81 · 116 | resolve config paths — never hardcode RawData / dictionary locations | (config) |
| `clear_faculty_caches()` · `committee.clear_committee_caches()` | faculty.py:1167 · committee.py:233 | invalidate in-process caches after spec/dict edits | — |

**DO NOT** hand-roll casefold loops, fuzzy thresholds, or name→email fallbacks: the
April-2026 staffing-audit multi-fallback strategy (AGENTS.org "Name resolution") is
already encoded in `reindex_to_email` + the two module dictionaries.

## §3 Definitions & conventions in force
- **Email is the canonical key.** Everything indexes by `@berkeley.edu` Email.
  **UID is sparse/inconsistent — do NOT reindex to UID** (AGENTS.org principle 7).
- **`.verified` semantics:** trust a `FacultyMember` only when `verified=True` (exactly
  one `faculty_vitals` match). The MCP path gates on `fm.verified` (mcp_server.py:406).
- **Data lives OUTSIDE the package** in a separate `RawData` git repo at
  `config["data_directory"]`, encrypted `.gpg`. **Never read RawData CSV/XLS directly** —
  go through `Feature` / `get_dataframe`; never bake in config paths (use §2 helpers).
  The Panel Settings tab syncs RawData via `git pull --ff-only`.
- **Feature year keys:** `'latest'`, `'history'`, academic-year string `'2023-24'`, or int start year.
- **`ucbfaculty/*.py` is authoritative & git-tracked** (unlike CFE/LSMS tangled `.py`);
  `notes.org`/`spelling.org` are design notes. Don't re-tangle to overwrite `.py` unless asked.
- **Config** from `~/.ucbfaculty/config.yaml` (or `$UCBFACULTY_CONFIG`): `data_directory`,
  `feature_specs_yaml`, `sources_yaml`, `spelling_dictionary`, `coms_sheet_key`.
- **Spelling normalization lives in dictionaries/transforms**, not ad-hoc casing fixes downstream.
- Default branch **`main`**; default git author **`Ethan Ligon & Sue Coder`**. GitNexus-indexed
  (CLAUDE.md mandates `gitnexus_impact`/`detect_changes` — tooling, not a code invariant).

## §4 Invariants & landmines
- **DO NOT reinvent name/identity matching.** Use the canonical machinery
  (`reindex_to_email` / `fullname_to_email_index` / `FACULTY_NAME_DICTIONARY` /
  `EMAIL_DICTIONARY`). **Email is the unique key; trust only when `.verified`** (one and
  only one vitals match). Hand-rolled matchers drift from the canonical fallbacks and
  re-introduce identity collisions.
- **Email-indexed, NOT Email-columned.** `faculty.volunteers()` (both single-year and
  `year='all'`) returns Email as the **index**. Never `pd.concat(..., ignore_index=True)`
  without `reset_index()` first — it discards per-frame Email → collisions (the bug fixed
  in `committee.py` rev `6e15556`; AGENTS.org "faculty.volunteers() returns Email-indexed").
- **Appointment vs Sheet rosters disagree (~5%).** `unexploited_volunteers()` checks only the
  Appointment source — **union both sources** when computing "appointed" sets.
- **COMS Google Sheet key ROTATES** (delete-and-recopy mints a new key) → keep it in config
  `coms_sheet_key`, never hardcode. Tabs are dual-section (stop parsing at the
  volunteer-choices secondary header); the GC tab needs the column-offset heuristic.
- **Nullable-boolean Features** (`AnyCommittee`, etc.) carry `pd.NA` → `series.fillna(False).astype(bool)`
  before any `==`/`if` (else `TypeError: boolean value of NA is ambiguous`).
- **Never overwrite/normalize files in `RawData/`** unless a maintainer directs it; encrypted
  assets need GPG (prime `gpg-agent` by decrypting one file first).
- **Caches are in-process** — call `clear_faculty_caches()` / `clear_committee_caches()` after
  spec or dictionary edits, or stale rows leak.

## §5 Reuse decision (standing guidance)
- Match a wild df to faculty → `reindex_to_email` / `ingest_dataframe`; names→email →
  `fullname_to_email_index`. **Never hand-roll.**
- Resolve a name variant → `FACULTY_NAME_DICTIONARY.spell()`; persist additions → `.persist()`.
- Load a data file → `rawdata.get_dataframe` (CSV/XLSX/GPG). Roster → `faculty_vitals`.
  A feature column → `Feature(name)[year]`.
- Slice / join / aggregate at scale → `queries.execute` (DuckDB views) or the MCP `query`
  tool — not whole-frame pulls into context.
- Identity object → `FacultyMember`; honor `.verified` before trusting it.

## §6 Open questions / notes
- AGENTS.org still says "no dedicated test suite," but a real `tests/` exists now (~24 pytest
  files; `poetry run pytest`) — prefer running it over the ad-hoc `python -c` smoke checks.
- `ligonlibrary` is an external dep supplying the real `get_dataframe` + fuzzy helpers
  (`find_similar_pairs`, `most_similar`); `thefuzz`/fallbacks kick in when it's absent.
- MCP surface was trimmed 20→13 tools (Apr 2026); `query`+`describe` subsume the old per-frame
  getters. Two gitnexus bugs filed in passing (#1091 method kind, #1092 tool flow attribution).
