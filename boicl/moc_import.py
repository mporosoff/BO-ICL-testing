"""Explicit MoC workbook/CSV import. Reading or validating never calls a model."""
from copy import deepcopy
from functools import lru_cache
import csv
import gzip
import hashlib
from io import StringIO, BytesIO
import json
import math
from pathlib import Path

DATA = Path(__file__).parent / "data" / "moc"
SOURCE_COMMIT = "810c3f7d7ecc632e4cfd422977b10b471e526f82"
NUMERIC = {
    "temperature_C",
    "ramp_C_per_min",
    "flow_sccm",
    "hold_h",
    "sucrose_to_AMT_mass_ratio",
    "AMT_to_sucrose_mass_ratio",
    "AMT_mass_g",
    "sucrose_mass_g",
    "AMT_water_mL",
    "sucrose_water_mL",
    "moc_wt_pct",
    "moc_wt_pct_sigma",
    "gof",
    "closure_gap_wt_pct",
    "phase_accounting_residual_wt_pct",
    "mo_wt_pct",
    "mo_wt_pct_sigma",
    "mo2c_wt_pct",
    "mo2c_wt_pct_sigma",
    "moo2_wt_pct",
    "moo2_wt_pct_sigma",
}
GRID = {
    "temperature_C": list(range(550, 901, 10)),
    "ramp_C_per_min": [5, 10, 15],
    "flow_sccm": [30, 60, 100],
    "hold_h": [0.5, 2, 5, 10],
    "sucrose_to_AMT_mass_ratio": [0.5, 1, 2],
    "gas": ["H2", "N2"],
}


def digest(value):
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def canonical_id(row):
    text = (
        f"moc-t1-v1|T={int(row['temperature_C'])}|ramp={int(row['ramp_C_per_min'])}"
        f"|hold={float(row['hold_h']):g}|gas={row['gas']}|flow={int(row['flow_sccm'])}"
        f"|sucrose_amt={float(row['sucrose_to_AMT_mass_ratio']):g}"
    )
    return "moc-" + digest(text)[:16]


def _rows(text):
    rows = list(csv.DictReader(StringIO(text.lstrip("\ufeff"))))
    return [_clean(row) for row in rows]


def _clean(row):
    result = {}
    for key, value in row.items():
        if (
            value is None
            or value == ""
            or (isinstance(value, float) and math.isnan(value))
        ):
            result[key] = None
        elif key in NUMERIC:
            result[key] = float(value)
        elif key in {"training_included", "eligible_for_new_quality_repeat"}:
            if str(value).lower() not in {"true", "false"}:
                raise ValueError(f"{key} must be true or false")
            result[key] = str(value).lower() == "true"
        else:
            result[key] = value
    return result


@lru_cache(maxsize=1)
def _source_candidate_hashes():
    """Trusted committed fixture, independent of hashes claimed by an import."""
    text = gzip.decompress((DATA / "design_space.csv.gz").read_bytes()).decode(
        "utf-8-sig"
    )
    return {
        row["candidate_id"]: row["procedure_sha256"]
        for row in csv.DictReader(StringIO(text))
    }


def validate_candidates(rows, require_full_grid=True):
    if require_full_grid and len(rows) != 7776:
        raise ValueError("MoC import requires exactly 7,776 unique candidates")
    seen = set()
    source_hashes = _source_candidate_hashes()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each candidate must be an object")
        for key, values in GRID.items():
            if row.get(key) not in values:
                raise ValueError(f"Off-grid or wrong-unit {key}: {row.get(key)!r}")
        cid = canonical_id(row)
        if row.get("candidate_id") != cid or cid in seen:
            raise ValueError(
                f"Invalid or duplicate canonical candidate ID: {row.get('candidate_id')}"
            )
        seen.add(cid)
        ratio = row["sucrose_to_AMT_mass_ratio"]
        if row.get("AMT_to_sucrose_mass_ratio") != 1 / ratio:
            raise ValueError("AMT:sucrose ratio is the reciprocal of sucrose:AMT")
        water = {0.5: (6.0, 1.5), 1.0: (5.5, 2.0), 2.0: (4.0, 3.5)}[ratio]
        if (
            row.get("AMT_mass_g"),
            row.get("sucrose_mass_g"),
            row.get("AMT_water_mL"),
            row.get("sucrose_water_mL"),
        ) != (1.0, ratio, *water):
            raise ValueError("Inconsistent linked recipe quantities")
        if row.get("protocol_version") != "moc-t1-v1":
            raise ValueError("Unsupported fixed-protocol version")
        if not isinstance(row.get("procedure"), str) or digest(
            row["procedure"]
        ) != row.get("procedure_sha256"):
            raise ValueError("Procedure text hash mismatch; preserve exact source text")
        if row["procedure_sha256"] != source_hashes.get(cid):
            raise ValueError(
                "MoC procedure differs from the pinned original recipe body"
            )
    return rows


def _validate_seeds(seeds, candidates):
    from .measurement_quality import quality_metadata

    expected = {"M7": 72.1, "M12": 83.8, "M13": 23.4}
    if (
        len(seeds) != 3
        or {r.get("run_id"): r.get("moc_wt_pct") for r in seeds} != expected
    ):
        raise ValueError("Confirmed seed snapshot must be M7=72.1, M12=83.8, M13=23.4")
    lookup = {r["candidate_id"]: r for r in candidates}
    original = _rows((DATA / "seeds.csv").read_text(encoding="utf-8-sig"))
    reference = {r["run_id"]: r for r in original}
    for row in seeds:
        if quality_metadata(row)["quantification_method"] != "historical_unspecified":
            raise ValueError(
                "Confirmed source seeds have unspecified quantification; record a new method through a documented refinement, not an initialization relabel"
            )
        if (
            row["candidate_id"] not in lookup
            or row["candidate_id"] != reference[row["run_id"]]["candidate_id"]
        ):
            raise ValueError("Seed identity does not join to the canonical pool")
        for key in [
            "moc_wt_pct_sigma",
            "gof",
            "closure_gap_wt_pct",
            "closure_gap_origin",
            "objective_authority",
            "refinement_version",
            "phase_accounting_residual_wt_pct",
        ]:
            if row.get(key) != reference[row["run_id"]][key]:
                raise ValueError(f"Confirmed seed quality metadata mismatch: {key}")
        if row.get("procedure") != lookup[row["candidate_id"]]["procedure"]:
            raise ValueError("Seed procedure does not match candidate")
        for key in (*GRID, "AMT_to_sucrose_mass_ratio"):
            if row.get(key) != lookup[row["candidate_id"]][key]:
                raise ValueError(f"Seed recipe field does not match candidate: {key}")
        row.update(
            training_included=True,
            record_status="measured",
            is_seed=True,
            physical_measurement_id=row["run_id"],
            measured_at=None,
            chronology="source order; experimental dates unknown",
        )
    return seeds


def _validate_archive(archive, candidates):
    original = _rows((DATA / "archive.csv").read_text(encoding="utf-8-sig"))
    reference = {r["observation_id"]: r for r in original}
    if len(archive) != 5 or {r.get("observation_id") for r in archive} != set(
        reference
    ):
        raise ValueError(
            "Historical archive must preserve the five excluded source measurements"
        )
    lookup = {r["candidate_id"]: r for r in candidates}
    for row in archive:
        expected = reference[row["observation_id"]]
        for key in (
            "candidate_id",
            "moc_wt_pct",
            "training_included",
            "record_status",
            "raw_result_metadata_json",
        ):
            if row.get(key) != expected[key]:
                raise ValueError(
                    f"Historical archive differs from the pinned source: {key}"
                )
        if row.get("procedure") != lookup[row["candidate_id"]]["procedure"]:
            raise ValueError("Archived recipe text does not match the candidate")


def load_moc_package(path=None, workbook_bytes=None):
    """Return independent canonical candidate, observation and provenance records."""
    schema = json.loads((DATA / "import_schema.json").read_text(encoding="utf-8"))
    hashes = {}
    if workbook_bytes is not None or (
        path is not None and Path(path).suffix.lower() == ".xlsx"
    ):
        import pandas as pd

        raw = workbook_bytes if workbook_bytes is not None else Path(path).read_bytes()
        sheets = pd.read_excel(
            BytesIO(raw), sheet_name=["Design_space", "Seed_observations"]
        )
        candidates = [_clean(r) for r in sheets["Design_space"].to_dict("records")]
        seeds = [_clean(r) for r in sheets["Seed_observations"].to_dict("records")]
        hashes["workbook"] = digest(raw)
        archive_bytes = (DATA / "archive.csv").read_bytes()
        archive = _rows(archive_bytes.decode("utf-8-sig"))
        hashes["bundled_archive"] = digest(archive_bytes)
    else:
        base = Path(path) if path else DATA
        if (base / "data").is_dir():
            base = base / "data"
        if path:
            loaded_schema = json.loads(
                (base / "import_schema.json").read_text(encoding="utf-8-sig")
            )
            if loaded_schema != schema:
                raise ValueError("Import schema differs from moc_handoff_v1 contract")
            files = {
                "candidates": "MoC_design_space_7776.csv",
                "seeds": "MoC_seed_observations.csv",
                "archive": "MoC_historical_excluded_observations.csv",
            }
            content = {key: (base / name).read_bytes() for key, name in files.items()}
        else:
            content = {
                "candidates": gzip.decompress(
                    (base / "design_space.csv.gz").read_bytes()
                ),
                "seeds": (base / "seeds.csv").read_bytes(),
                "archive": (base / "archive.csv").read_bytes(),
            }
        hashes = {key: digest(raw) for key, raw in content.items()}
        candidates, seeds, archive = (
            _rows(content[key].decode("utf-8-sig"))
            for key in ["candidates", "seeds", "archive"]
        )
    validate_candidates(candidates)
    _validate_seeds(seeds, candidates)
    _validate_archive(archive, candidates)
    lookup = {r["candidate_id"]: r for r in candidates}
    for row in archive:
        if row["candidate_id"] not in lookup:
            raise ValueError("Archived measurement is outside candidate pool")
        raw = json.loads(row.get("raw_result_metadata_json") or "{}")
        row.update(
            is_seed=True,
            measured_at=None,
            chronology="source order; experimental dates unknown",
            physical_measurement_id=row["observation_id"],
            refinement_version="historical-workbook-v1",
            moc_wt_pct_sigma=raw.get("sigma.2"),
            gof=raw.get("GOF"),
            closure_gap_wt_pct=None,
            closure_gap_origin="unknown; phase list not established complete",
            phase_accounting_complete=False,
            phase_accounting_residual_wt_pct=100
            - sum(
                float(raw.get(k, 0))
                for k in [
                    "Mo (Im3m) weight fraction ",
                    "Mo2C(pbcn) weight fraction",
                    "MoC(Fm3m) weight fraction",
                    "MoO2(P21/c) weight fraction",
                ]
            ),
        )
    fingerprint = digest(
        json.dumps(
            [(r["candidate_id"], r["procedure_sha256"]) for r in candidates],
            separators=(",", ":"),
        )
    )
    return dict(
        candidates=candidates,
        observations=seeds,
        archive=archive,
        pool_fingerprint=fingerprint,
        provenance=dict(
            schema=schema,
            source_commit=SOURCE_COMMIT,
            source_file_hashes=hashes,
            m12_resolution=schema["m12_resolution"],
            saved_gp_proposal="reference only; not reserved or measured",
            source_order="ascending first source Excel row; duplicates mapped in source_rows",
        ),
    )
