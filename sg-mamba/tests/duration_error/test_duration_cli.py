from experiments.duration_error.cli import parse_thresholds


def test_parse_thresholds_accepts_comma_and_space_separated_values():
    assert parse_thresholds(["0.3,0.5", "0.7"]) == (0.3, 0.5, 0.7)