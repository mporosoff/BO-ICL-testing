from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import pandas as pd


OCM_COLUMN_NAMES = {
    "Name": "name",
    "Support ": "sup",
    "M1": "m1",
    "M1_mol": "m1_mol",
    "M2": "m2",
    "M2_mol": "m2_mol",
    "M3": "m3",
    "M3_mol": "m3_mol",
    "Temp": "react_temp",
    "Total_flow": "flow_vol",
    "Ar_flow": "ar_vol",
    "CH4_flow": "ch4_vol",
    "O2_flow": "o2_vol",
    "CT": "contact",
}

OCM_PROMPT_TEMPLATE = (
    "To synthesize {name}, {sup} (1.0 g) was impregnated with 4.5 mL "
    "of an aqueous solution consisting of {m1} ({m1_mol} mol), "
    "{m2} ({m2_mol} mol), {m3} ({m3_mol} mol), at 50 degrees C for 6 h. "
    "Once activated the reaction is ran at {react_temp} degrees C. "
    "The total flow rate was {flow_vol} mL/min (Ar: {ar_vol} mL/min, "
    "CH4: {ch4_vol} mL/min, O2: {o2_vol} mL/min), leading to a contact "
    "time of {contact} s."
)


def calculate_ocm_m1_mol(row: pd.Series) -> float:
    return round(
        (row["M1_mol%"] / 100)
        * (row["M2_mol"] + row["M3_mol"])
        / (1 - (row["M1_mol%"] / 100)),
        3,
    )


def build_ocm_dataset(
    raw_csv: Optional[Union[str, Path]] = None,
    samples_per_catalyst: int = 216,
) -> pd.DataFrame:
    data_dir = Path(__file__).resolve().parents[1] / "paper" / "dataset"
    raw_csv = (
        Path(raw_csv)
        if raw_csv is not None
        else data_dir / "oxidative_methane_coupling.csv"
    )

    raw_data = pd.read_csv(raw_csv)
    raw_data["M1_mol"] = raw_data.apply(calculate_ocm_m1_mol, axis=1)
    raw_data = raw_data.rename(columns=OCM_COLUMN_NAMES)

    filtered_data = pd.concat(
        [
            raw_data[raw_data["name"] == catalyst].iloc[:samples_per_catalyst]
            for catalyst in raw_data["name"].unique()
        ],
        ignore_index=True,
    )

    rows = []
    for _, row in filtered_data.iterrows():
        props = OrderedDict(
            (key, value)
            for key, value in row.items()
            if key in OCM_COLUMN_NAMES.values()
        )
        rows.append(
            {
                "prompt": OCM_PROMPT_TEMPLATE.format(**props),
                "completion": row["C2y"],
            }
        )

    return pd.DataFrame(rows)


def write_ocm_dataset(
    output_csv: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Path:
    data_dir = Path(__file__).resolve().parents[1] / "paper" / "dataset"
    output_csv = (
        Path(output_csv)
        if output_csv is not None
        else data_dir / "data" / "12708_ocm_dataset.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    build_ocm_dataset(**kwargs).to_csv(output_csv, sep=";", index=False)
    return output_csv
