# Canonical configuration and execution relocation

Legacy import paths intentionally resolve to the same module object so historical imports and monkeypatch targets remain functional. No duplicated implementations. New canonical locations:

- `lithiumscope.core.config_schema` → `lithiumscope.core.configuration.schema`
- `lithiumscope.core.config` → `lithiumscope.core.configuration.loading`
- `lithiumscope.core.checkpoints` → `lithiumscope.core.execution.checkpoints`
- `lithiumscope.core.run_context` → `lithiumscope.core.execution.context`
- `lithiumscope.core.run_resume` → `lithiumscope.core.execution.resume`
- `lithiumscope.core.experiment_tracker` → `lithiumscope.core.execution.tracker`
- `lithiumscope.core.code_size_audit` → `lithiumscope.tools.inspection.code_size`

Execution and tests pending local validation. No scientific acceptance criterion is closed by relocation alone.
