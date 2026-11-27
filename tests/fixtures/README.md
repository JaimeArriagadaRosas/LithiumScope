# Test fixtures

Use pytest tmp_path for generated synthetic data. The public integrated demonstration inputs remain under examples/integrated_demo, and real archived data and models remain excluded from Git. No synthetic observations are valid external scientific evidence.

Shared pytest fixtures are registered globally in `tests/conftest.py` so all eight categories can use them. Keep generated data in `tmp_path` and only reusable fixture inputs in this directory.
