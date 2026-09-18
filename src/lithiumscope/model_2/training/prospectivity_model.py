from sklearn.ensemble import RandomForestClassifier
def create_model(random_seed:int=42,n_jobs:int=-1,**params):
    defaults={"n_estimators":400,"random_state":random_seed,"n_jobs":n_jobs,"class_weight":"balanced"};defaults.update(params);return RandomForestClassifier(**defaults)
