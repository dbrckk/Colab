# Agent instructions — Colab

## Shared development policy

This repository adopts the [88 validated rules](https://github.com/dbrckk/repo-standards/blob/db2f86657ada74a0561e07189f9942d6b66ebb4a/standards/88-rules.md), [agent skill](https://github.com/dbrckk/repo-standards/blob/db2f86657ada74a0561e07189f9942d6b66ebb4a/skills/repo-excellence-88/SKILL.md) and [educational wiki](https://github.com/dbrckk/repo-standards/blob/db2f86657ada74a0561e07189f9942d6b66ebb4a/docs/WIKI-88.md) as a **policy-only adoption**. The standard is pinned to the verified commit shown in these links.

- Follow relevant foundational rules and activate conditional rules only when pertinent.
- **Do not create new unit tests.** Do not delete existing tests. Prefer real functional/integration checks, lint, build and smoke validation appropriate to the change.
- Before modifying: read the repository README, inspect current branch/head, recent changes, architecture, relevant code and any project-specific instructions.
- Deliver small, coherent, reversible changes. Never claim a check or deploy succeeded unless actually observed.
- Preserve existing workflows, dependencies and security configurations. Any destructive or high-impact action needs appropriate authorization.
- Update the existing project status/wiki when meaningful; save validated reusable procedures as skills.
- Priority: reliability > real functionality > security > architecture > performance > interface > secondary features.

## Repository-specific focus

Notebooks/Kaggle/Colab: verify notebook startup, external credentials handling, GPU resources and reproducibility.

This documentation does **not** configure `repo-standards` automation or install new CI workflows. Treat repository-specific instructions as complementary unless contradictory to the current owner's explicit preferences.
