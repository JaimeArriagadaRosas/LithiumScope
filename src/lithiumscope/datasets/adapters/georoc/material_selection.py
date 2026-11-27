"""Select GEOROC material without changing its provenance."""
from .material import canonical_material_label, parse_georoc_material


def select_material(frame, allowed_material_types):
    if "Sample_type" not in frame.columns and allowed_material_types:
        return frame.iloc[:0].copy()
    if "Sample_type" in frame.columns:
        frame["Sample_type_raw"] = (
            frame["Sample_type"]
            .astype("string")
            .fillna("")
            .str.strip()
        )
        parsed_material = frame[
            "Sample_type_raw"
        ].map(
            parse_georoc_material
        )
        frame["Sample_material_code"] = (
            parsed_material.map(
                lambda item: item.code
            )
        )
        frame["Sample_material_batch_id"] = (
            parsed_material.map(
                lambda item: item.batch_id
            )
        )
        frame["Sample_type"] = (
            frame["Sample_material_code"].map(
                canonical_material_label
            )
        )

        if allowed_material_types:
            accepted_codes = {
                parse_georoc_material(
                    value
                ).code
                for value in allowed_material_types
            }
            if None in accepted_codes:
                raise ValueError(
                    "allowed_material_types contiene "
                    "un material GEOROC no reconocido."
                )
            frame = frame[
                frame["Sample_material_code"].isin(
                    accepted_codes
                )
            ].copy()

    return frame
