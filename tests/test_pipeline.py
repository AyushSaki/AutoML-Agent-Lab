import pandas as pd
from ml.pipeline import analyze_dataset

def test_profile(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"age":[20,30,40], "target":[0,1,0]}).to_csv(path, index=False)
    result = analyze_dataset(path)
    assert result["rows"] == 3
    assert result["columns"] == 2
