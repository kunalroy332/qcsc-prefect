"""FlowParameters strictness and the sampler options."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from sbd.flow_params import CircuitParameters, DEParameters, FlowParameters

_REPO = Path(__file__).resolve().parents[3]
_PARAM_JSONS = sorted(
    [
        *(_REPO / "algorithms" / "sbd" / "default_params").glob("*.json"),
        *(_REPO / "docker" / "config").glob("*.json"),
    ]
)


@pytest.mark.parametrize("path", _PARAM_JSONS, ids=lambda p: p.name)
def test_existing_param_jsons_still_validate(path):
    params = FlowParameters.model_validate_json(path.read_text())
    assert params.fcidump


def test_param_jsons_found():
    # guard against the glob silently matching nothing (the parametrized test would then vanish)
    assert any(p.parent.name == "default_params" for p in _PARAM_JSONS)


@pytest.mark.parametrize(
    "payload, loc",
    [
        ({"fcidump": "x", "fcidum": "y"}, ("fcidum",)),
        ({"fcidump": "x", "ao_labels": ["0_Fe_3dxy"]}, ("ao_labels",)),
        ({"fcidump": "x", "spin_pattern": [1, -1]}, ("spin_pattern",)),
        ({"fcidump": "x", "circ_params": {"n_lucj_layer": 3}}, ("circ_params", "n_lucj_layer")),
        ({"fcidump": "x", "de_params": {"num_walker": 2}}, ("de_params", "num_walker")),
    ],
)
def test_unknown_keys_rejected_at_every_level(payload, loc):
    with pytest.raises(ValidationError) as exc:
        FlowParameters.model_validate(payload)
    errors = exc.value.errors()
    assert [e["type"] for e in errors] == ["extra_forbidden"]
    assert tuple(errors[0]["loc"]) == loc


def test_unknown_keys_rejected_from_json_and_constructors():
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        FlowParameters.model_validate_json(json.dumps({"fcidump": "x", "sqd_dimm": 10}))
    with pytest.raises(ValidationError):
        CircuitParameters(n_lucj_layers=2, bogus=1)
    with pytest.raises(ValidationError):
        DEParameters(num_walkers=4, bogus=1)


def test_known_keys_still_accepted():
    p = FlowParameters(
        fcidump="x",
        sqd_dim=10,
        circ_params=CircuitParameters(n_lucj_layers=1),
        de_params=DEParameters(num_walkers=1, iterations=1),
    )
    assert p.circ_params.n_lucj_layers == 1 and p.de_params.num_walkers == 1


def test_fcidump_required():
    with pytest.raises(ValidationError):
        FlowParameters()


def test_round_trip_json():
    p = FlowParameters(fcidump="/data/x.fcidump", sqd_dim=5)
    q = FlowParameters.model_validate_json(p.model_dump_json())
    assert q == p


# ---------------------------------------------------------------------------------------------
# Sampler options are flow parameters (no Prefect Variable)
# ---------------------------------------------------------------------------------------------


def test_sampler_defaults_and_runtime_options():
    from sbd.flow_params import SamplerParameters

    p = FlowParameters(fcidump="x")
    assert p.sampler.shots == 10_000 and p.sampler.n_shot_batches == 1
    assert p.sampler.saved_samples == []
    assert p.sampler.runtime_options() == {"params": {"shots": 10_000}}

    sp = SamplerParameters(
        shots=5000, dynamical_decoupling=True, dd_sequence="XY4", measure_twirling=True
    )
    assert sp.runtime_options(shots=1000) == {
        "params": {
            "shots": 1000,
            "options": {
                "dynamical_decoupling": {"enable": True, "sequence_type": "XY4"},
                "twirling": {"enable_gates": False, "enable_measure": True},
            },
        }
    }


@pytest.mark.parametrize(
    "sampler, loc",
    [
        ({"shot": 10}, ("sampler", "shot")),
        ({"shots": 0}, ("sampler", "shots")),
        ({"dd_sequence": "XYZ"}, ("sampler", "dd_sequence")),
    ],
)
def test_sampler_validation(sampler, loc):
    with pytest.raises(ValidationError) as exc:
        FlowParameters.model_validate({"fcidump": "x", "sampler": sampler})
    assert tuple(exc.value.errors()[0]["loc"]) == loc


def test_flow_passes_sampler_to_walkers():
    import inspect

    import sbd.main as main
    import sbd.sqd as sqd

    assert "sampler_params=parameters.sampler" in inspect.getsource(main)
    src = inspect.getsource(sqd.walker_sqd.fn)
    assert "Variable.get" not in src and not hasattr(sqd, "Variable")


@pytest.mark.parametrize(
    "argv, env, key",
    [
        (["--shots", "5000"], {}, "shots"),
        (["--measure-twirling"], {}, "measure_twirling"),
        (["--saved-samples", "file:///p.npz"], {}, "saved_samples"),
        (["--options-variable-name", "x"], {}, "options_variable_name"),
        ([], {"SBD_SHOTS": "100"}, "shots"),
    ],
)
def test_create_blocks_rejects_removed_sampler_options(monkeypatch, argv, env, key):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "create_blocks_under_test", _REPO / "algorithms" / "sbd" / "create_blocks.py"
    )
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    for var in list(env) + ["SBD_SHOTS"]:
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sys, "argv", ["create_blocks.py", *argv])
    with pytest.raises(SystemExit, match=rf"{key} .*FlowParameters\.sampler|{key} .*no Prefect"):
        cb._reject_removed_sampler_options(cb._parse_args(), {}, cb._env_values())
    # a config file key is rejected the same way
    monkeypatch.setattr(sys, "argv", ["create_blocks.py"])
    for k in env:
        monkeypatch.delenv(k)
    with pytest.raises(SystemExit, match="FlowParameters"):
        cb._reject_removed_sampler_options(cb._parse_args(), {"shots": 1}, cb._env_values())
    cb._reject_removed_sampler_options(cb._parse_args(), {}, cb._env_values())  # clean: no exit
