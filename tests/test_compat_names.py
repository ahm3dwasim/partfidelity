"""Names from before the public rename keep importing as aliases of the new ones."""

def test_former_module_and_constant_names_still_import():
    from app import contracts, d1b, scorer_time
    from app.routes import site

    assert d1b.D1bMatch is scorer_time.TimeMatch and d1b.D1bReport is scorer_time.TimeReport
    assert d1b.load_d1b_law("tolerance_s") == scorer_time.load_scorer_law("tolerance_s")
    assert contracts.D1B_SCORE_LAW is contracts.TIMING_SCORE_LAW
    assert contracts.C1_READBACK_CUTOFF == contracts.READBACK_CONFIDENCE_CUTOFF
    assert site.H1 == site.HEADLINE
