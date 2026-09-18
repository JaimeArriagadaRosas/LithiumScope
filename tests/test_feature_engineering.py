import pandas as pd
import pytest
from lithiumscope.model_1.steps.step_07_feature_engineering import add_geochemical_features

def test_geochemical_features():
    frame=pd.DataFrame({"Na2O":[3.0],"K2O":[4.0],"MgO":[2.0],"Fe2O3":[6.0],"Al2O3":[14.0],"CaO":[1.0]});result=add_geochemical_features(frame);assert result.loc[0,"Alkali_Sum"]==pytest.approx(7.0);assert result.loc[0,"Mg_Number"]==pytest.approx(2.0/(8.0+1e-6));assert result.loc[0,"A_CNK_proxy"]==pytest.approx(14.0/(8.0+1e-6));assert result.loc[0,"K_Mg_ratio"]==pytest.approx(4.0/(2.0+1e-6))
