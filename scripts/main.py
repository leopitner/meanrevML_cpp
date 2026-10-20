import sys
sys.path.insert(0, "build/Release")
import meanrevML_py as mr

# Load real data
bars = mr.load_stooq_txt("data/spy.us.txt")
print(f"{len(bars)} bars loaded, close of last = {bars[-1].close}")

# Train a forest
cfg = mr.RFConfig()
cfg.n_trees, cfg.max_depth, cfg.min_samples_leaf = 150, 6, 15
rf = mr.ShallowRF(cfg)
rf.fit(X_train, y_train)
proba = rf.predict_proba(X_test)