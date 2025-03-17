from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import time

X, y = make_regression(n_samples=100_000, n_features=30, random_state=0)

rf = RandomForestRegressor(
    n_estimators=200,
    max_features="sqrt",
    n_jobs=-1,
    random_state=2025
)

start = time.time()
rf.fit(X, y)
print("train time:", time.time() - start, "seconds")
