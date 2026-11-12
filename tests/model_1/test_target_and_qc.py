import pandas as pd
from lithiumscope.model_1.steps.step_04_quality_control import apply_quality_control
from lithiumscope.model_1.steps.step_05_target_filtering import filter_target

def test_target_then_quality_control():
    frame=pd.DataFrame({"Li_icpms":list(range(1,101)),"SUM (no water)":[98.0]*99+[120.0]});filtered,target=filter_target(frame,["Li_icpms"],0.025,0.975);assert target=="Li_icpms";qc=apply_quality_control(filtered,["SUM (no water)"],94.0,102.0);assert qc["SUM (no water)"].between(94,102).all()
