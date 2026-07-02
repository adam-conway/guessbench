from guessbench.acceptance import (
    acceptance_table,
    calibration_artifacts,
    check_t1_anchors,
    check_t2_monotonicity,
    check_t3_stability,
    check_t4_surface_invariance,
    load_anchors,
    load_ladders,
)


def anchor_reports(a1=1.0, a2=1.1, a3=1.8, a4=6.0, a1_k=1, a3_k=2):
    return {
        "A1": {"ei": a1, "k": a1_k},
        "A2": {"ei": a2, "k": 2},
        "A3": {"ei": a3, "k": a3_k},
        "A4": {"ei": a4, "k": 12},
    }


class TestCalibrationData:
    def test_anchors_load(self):
        anchors = load_anchors()
        assert [a["id"] for a in anchors] == ["A1", "A2", "A3", "A4"]
        assert "HELLO" in anchors[0]["prompt"]

    def test_ladders_load_with_structure(self):
        ladders = load_ladders()
        assert len(ladders) == 5
        families = {lad["family"] for lad in ladders}
        assert len(families) == 5
        for ladder in ladders:
            assert ladder["status"] == "PENDING_REVIEW"
            assert [lv["level"] for lv in ladder["levels"]] == [1, 2, 3, 4, 5]
            # Each rung above L1 resolves at least one named dimension (SPEC 5.1).
            for level in ladder["levels"][1:]:
                assert level["dimensions_resolved"], (
                    f"{ladder['family']} L{level['level']} resolves no dimensions"
                )

    def test_calibration_artifact_count(self):
        artifacts = calibration_artifacts()
        assert len(artifacts) == 29  # 4 anchors + 5 ladders x 5 levels
        assert "A1" in artifacts and "code:L5" in artifacts


class TestT1:
    def test_passes_on_expected_shape(self):
        assert check_t1_anchors(anchor_reports()).passed

    def test_a1_must_be_single_cluster(self):
        assert not check_t1_anchors(anchor_reports(a1=1.2, a1_k=2)).passed

    def test_a3_cluster_identity_not_entropy(self):
        # EI anywhere in [1, 2] is fine; k=3 is not.
        assert check_t1_anchors(anchor_reports(a3=1.05)).passed
        assert not check_t1_anchors(anchor_reports(a3_k=3)).passed

    def test_a4_needs_floor_and_margin(self):
        assert not check_t1_anchors(anchor_reports(a4=3.0)).passed  # below floor
        assert not check_t1_anchors(anchor_reports(a2=1.25, a4=2.4)).passed  # thin margin


class TestT2:
    def test_monotone_ladder_passes(self):
        eis = {"code": {1: 8.0, 2: 5.0, 3: 3.0, 4: 2.0, 5: 1.2}}
        assert check_t2_monotonicity(eis).passed

    def test_non_monotone_fails(self):
        eis = {"code": {1: 8.0, 2: 9.0, 3: 3.0, 4: 2.0, 5: 1.2}}
        assert not check_t2_monotonicity(eis).passed

    def test_l5_too_high_fails_even_if_monotone(self):
        eis = {"code": {1: 8.0, 2: 6.0, 3: 4.0, 4: 3.0, 5: 2.0}}
        assert not check_t2_monotonicity(eis).passed

    def test_all_ladders_must_pass(self):
        eis = {
            "good": {1: 8.0, 2: 5.0, 3: 3.0, 4: 2.0, 5: 1.2},
            "bad": {1: 2.0, 2: 5.0, 3: 3.0, 4: 2.0, 5: 1.2},
        }
        assert not check_t2_monotonicity(eis).passed


class TestT3:
    def run(self, ei_a, ei_b, ci_a, ci_b):
        return check_t3_stability(
            {"X": ({"ei": ei_a, "ei_ci_90": ci_a}, {"ei": ei_b, "ei_ci_90": ci_b})}
        )

    def test_close_runs_pass(self):
        assert self.run(2.0, 2.3, [1.5, 2.6], [1.8, 2.9]).passed

    def test_non_overlapping_cis_fail(self):
        assert not self.run(2.0, 2.4, [1.8, 2.1], [2.2, 2.7]).passed

    def test_large_delta_fails_despite_overlap(self):
        assert not self.run(2.0, 3.0, [1.0, 4.0], [1.0, 4.0]).passed

    def test_relative_tolerance_for_large_ei(self):
        # dEI = 1.0 but 15% of 10 = 1.5, so allowed.
        assert self.run(9.0, 10.0, [8.0, 11.0], [8.5, 11.5]).passed


class TestT4:
    def test_boundary(self):
        assert check_t4_surface_invariance({"ei": 1.3}).passed
        assert not check_t4_surface_invariance({"ei": 1.31}).passed


class TestTable:
    def test_render(self):
        results = {
            "llm_judge": [
                check_t1_anchors(anchor_reports()),
                check_t4_surface_invariance({"ei": 1.5}),
            ]
        }
        table = acceptance_table(results)
        assert "llm_judge" in table
        assert "PASS" in table and "FAIL" in table
