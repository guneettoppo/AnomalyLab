"""AnomalyLab — a leakage-aware re-implementation of the NAB benchmark.

Package layout
--------------
io          corpus + label loading
features    causal rolling feature matrix
detectors   detector implementations (classical stats, IF, PCA, HBOS, ensemble)
nab_scoring faithful port of NAB's application-based scoring profile
metrics     range-aware / false-alarm-aware evaluation on top of the NAB score
plots       figures for the report
"""
__version__ = "1.0.0"
