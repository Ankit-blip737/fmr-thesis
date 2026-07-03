"""Smoke test for the Stage 6 benchmark orchestration (identity-correction path).

Runs run_fmr_full against a tiny synthetic config to exercise: scoring both
splits, the guarded correction fallback, incremental-fusion validation, and the
post-correction abstention gate. Guards the whole Stage-6 wiring in one test.
"""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _tiny_config_dir(tmp_path):
    import yaml
    d = tmp_path / "cfg"
    d.mkdir()
    (d / "data.yaml").write_text(yaml.safe_dump({
        "dataset": "synthetic",
        "synthetic": {"name": "synthetic", "n": 400, "grounded_fraction": 0.5, "grid": 4, "seed": 7},
        "split": {"fractions": [0.5, 0.25, 0.25], "holdout_modality": None, "seed": 13},
    }))
    (d / "models.yaml").write_text(yaml.safe_dump({
        "model": "mock_reasoner",
        "mock_reasoner": {"backend": "mock", "name": "mock-reasoner", "is_reasoning": True,
                          "n_steps": 4, "drift": 0.35, "seed": 0},
        "mock_reasoner_b": {"backend": "mock", "name": "mock-reasoner-b", "is_reasoning": True,
                            "n_steps": 5, "drift": 0.5, "grounded_peak": 2.8, "prior_peak": 1.8, "seed": 101},
    }))
    (d / "experiment.yaml").write_text(yaml.safe_dump({
        "signals": {"weights": {"a": 0.4, "b": 0.3, "c": 0.3},
                    "consistency": {"n_samples": 3, "temperature": 0.7}},
        "abstention": {"alpha": 0.1, "delta": 0.1},
    }))
    return str(d)


def test_full_benchmark_runs(tmp_path):
    import run_fmr_full
    cfg_dir = _tiny_config_dir(tmp_path)
    out = str(tmp_path / "out")
    res = run_fmr_full.run(["mock_reasoner", "mock_reasoner_b"], alpha=0.1, delta=0.1,
                           out_dir=out, config_dir=cfg_dir)
    # Correction is now merged in, so the guarded import resolves.
    assert res["correction_present"] is True
    for key in ("mock_reasoner", "mock_reasoner_b"):
        m = res["models"][key]
        v = m["validation"]
        # Incremental fusion produces three AUROCs in a sane band.
        for s in ("auroc_fs_A", "auroc_fs_AB", "auroc_fs_ABC"):
            assert 0.5 < v[s] <= 1.0
        # Correction ran (selective): it raises mean faithfulness, and accuracy
        # does not collapse. A small accuracy dip is the documented
        # faithfulness/accuracy trade-off (right-by-luck ungrounded answers may
        # flip) — bounded, not zero.
        ce = m["correction_effect"]
        assert ce["n_applied"] > 0
        assert ce["mean_fs_after"] >= ce["mean_fs_before"] - 1e-9
        assert ce["acc_after"] >= ce["acc_before"] - 0.05
        # Deployed gate: if feasible, the empirical guarantee must hold on test.
        gate = m["abstention"]["fs_post_correction"]
        assert gate["guarantee_holds"]
    assert (tmp_path / "out" / "full_benchmark.json").exists()
