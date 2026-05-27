from ucbfaculty.apps.lookup import _committee_feature_options, _committee_feature_frame


def test_committee_feature_options_produce_mergeable_frames():
    feature_names = _committee_feature_options()
    assert feature_names, "Expected at least one committee feature option"

    for name in feature_names:
        frame = _committee_feature_frame((name,))
        assert frame is not None, f"Feature {name} returned no data frame"
        assert "Email" in frame.columns, f"Feature {name} lacks Email column"
