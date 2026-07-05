# Preprocessing

This directory contains dataset-specific conversion and validation scripts.

Each dataset should preferably provide:

```text
preprocess_<dataset>.py
validate_<dataset>.py
```

A preprocessing script should:

1. read the original dataset;
2. preserve chronological order;
3. identify feature and label columns;
4. use only normal training data where required by the protocol;
5. generate processed training, test, and test-label files;
6. save metadata describing feature names, shapes, and label statistics.
