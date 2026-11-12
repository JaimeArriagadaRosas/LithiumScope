import pandas as pd
from lithiumscope.experiments.scenarios.extended import experiment_locations
from lithiumscope.experiments.validation.ground_truth import validate_geochemistry, REQUIRED


def test_shifted_coordinates_distinct():
    locations = experiment_locations()
    assert len(locations) == 8
    assert locations["atacama_east"] != locations["atacama_north"]


def test_geochemical_evidence_rejects_brine_and_nonindependent():
    common = {key: None for key in REQUIRED}
    common.update(sample_id="R1", material="rock", latitude=-23, longitude=-68,
                  Li_value=25, Li_unit="ppm", analytical_method="ICP-MS",
                  source="laboratory", sample_date="2025-01-01", independent_of_training=True)
    brine = dict(common, sample_id="B1", material="brine")
    reused = dict(common, sample_id="R2", independent_of_training=False)
    eligible, excluded = validate_geochemistry(pd.DataFrame([common,brine,reused]))
    assert eligible.sample_id.tolist() == ["R1"]
    assert set(excluded.sample_id) == {"B1","R2"}
