# Test architecture

Tests are classified by the primary purpose of their assertions, not by the package under test. The eight categories are architecture (boundaries and structure), unit (isolated behavior), integration (component interaction), functional (user-visible contracts), scientific (methodology and evidence), regression (historical behavior), performance (resource and execution behavior), and reliability (persistence, logging and validation).

## Incremental migration

The relocation catalog at `docs/integral-refactor/test-relocation-purpose.json` records the planned destination of each existing test module. During migration, exactly one of the original or canonical paths must exist (identical paths count once). New tests should use canonical destinations. No test is duplicated merely to preserve its former path.

Move categories in separate commits, updating path-sensitive tests and criterion references in the same commit. Keep the default pytest invocation unfiltered. Preserve test assertions and scientific criteria; collection and execution must be validated separately from static checks.

Limits: 300 lines per Python file, at most 20 direct Python files per directory, unique test module basenames. Shared fixtures belong in a common conftest when required by multiple categories. Category markers are registered ahead of migration; automatic assignment and cross-cutting markers are activated only when their collection contract is implemented.
