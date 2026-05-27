import pandas as pd
import panel as pn
from panel.viewable import Viewable

import ucbfaculty.apps.lookup as lookup


def test_build_committee_view_smoke(monkeypatch):
    """Ensure committee view builds without raising."""
    pn.extension("tabulator", inline=True)

    # Minimal stubs to avoid hitting real data
    monkeypatch.setattr(lookup, "committee_names", lambda: ["COMS"])
    monkeypatch.setattr(lookup, "committee_year_range", lambda: (2024, 2025))
    monkeypatch.setattr(lookup, "_committee_feature_options", lambda: [])

    dummy = pd.DataFrame(
        {"Academic Year": ["2024-25"], "Committee": ["COMS"], "FullName": ["X"], "Role": ["Chair"], "YearStart": [2024], "Email": ["x@example.com"], "DeptName": ["Dept"]}
    )
    monkeypatch.setattr(lookup, "committee_membership", lambda committee=None, start_year=None, end_year=None: dummy)

    view = lookup.build_committee_view()
    assert isinstance(view, Viewable)
