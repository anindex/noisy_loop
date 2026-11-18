import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "study,options",
    [
        (
            "fit",
            [
                "--counts",
                "32",
                "--targets",
                "ramp",
                "smooth",
                "--resolution",
                "both",
                "--shrink",
                "--validation",
                "16",
            ],
        ),
        ("tiling", ["--widths", "2", "4"]),
        ("queries", []),
    ],
)
def test_experiment_outputs_and_overwrite_protection(tmp_path, study, options):
    output = tmp_path / "results.json"
    script = Path(__file__).resolve().parents[1] / "experiments.py"
    command = [
        sys.executable,
        str(script),
        study,
        "--test",
        "16",
        "--out",
        str(output),
        *options,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    original = output.read_bytes()
    rows = json.loads(original)["rows"]
    assert rows
    if study == "fit":
        assert {row["target"] for row in rows} == {"ramp", "smooth"}
        assert {row["resolution"] for row in rows} == {"fixed", "growing"}
        for row in rows:
            assert row["conditional_kl"]["forest"]["mean"] == pytest.approx(
                row["joint_kl"]["mean"]
            )
            assert row["block_ise"] >= -1e-12
            assert row["projection_gap"] <= 1e-9
            assert row["shrinkage"] in [0, 0.25, 0.5, 0.75, 1]
    elif study == "tiling":
        unchanged = next(row for row in rows if row["tile_width"] == 4)
        assert unchanged["deleted_edges"] == 0
        assert unchanged["joint_kl"]["mean"] == 0
        assert unchanged["conditional_kl"]["mean"] == pytest.approx(0, abs=1e-14)
        assert unchanged["tv"]["mean"] == 0
        assert all(0 <= row["tv"]["mean"] <= 1 for row in rows)
    else:
        assert {row["ordering"] for row in rows} == {"colamd", "nested_dissection"}
        assert max(row["transfer_discrepancy"] for row in rows) < 1e-10
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "Output already exists" in repeated.stderr
    assert output.read_bytes() == original
