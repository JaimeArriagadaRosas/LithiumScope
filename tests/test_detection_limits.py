import pandas as pd
from lithiumscope.model_1.steps.step_02_detection_limits import clean_detection_limits

def test_detection_limits_follow_source_rules():
    frame=pd.DataFrame({"x":["<10",">100","7.5",None]});result=clean_detection_limits(frame);assert result.loc[0,"x"]==5.0;assert result.loc[1,"x"]==100.0;assert result.loc[2,"x"]=="7.5"
