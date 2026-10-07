# SONIC consolidation implementation plan

> Execute inline using executing-plans, retaining the original controller code.

**Goal:** Ship a runnable repository and offline resource archive from the proven lean pipeline.
**Architecture:** Preserve workspace layout and materialize existing path templates; reuse the existing mobile60 scene snapshot and original augmentation scripts.
**Tech Stack:** Python 3.11, Isaac Sim 5.1, SONIC ONNX, LeRobot v2.1.
**Spec:** ../specs/2026-10-08-consolidation.md

## Global constraints

No changes to PD, contact gates, physics200/control50/camera25 Hz or decoder history. No credentials. Preserve old experiments. Runtime/output paths must avoid upstream robot→meta substitution.

## Review focus

Relocation, corrupt resource files, path traversal, existing outputs, missing environment dependencies. Verify resource materialization without the original workspace and preserve failed evidence.

## Tasks

- [x] Add CLI contract tests: clean subprocess preparation in a path with spaces; path rejection; manifest corruption rejection; existing runtime rejection.
- [x] Copy preserved source/resources and record original hashes; supplement encoder/planner, safe025, walking initial state and Revo2 collision meshes. Generate resource manifest.
- [x] Implement a thin launcher and materialization. Run tests, real offline candidate preparation and dependency doctor.
- [x] Run one full lean generation/replay and export after relocation; record complete wall and capture timing separately.
- [x] Write README, GPT task prompt, setup script, source-change report and capability limits.
- [x] Verify all tracked files and offline archive, review changes, commit the local repository and provide links.

## Execution record

Ruling: The user's follow-up specifies minimal changes and immediate runnability and authorizes implementation of the proposed consolidation; proceed without another design permission round. Use a newly created independent repository, so no existing branch or worktree changes.

Verification: one independent runtime physical pair and real GR00T loader passed; full60 offline candidates checked; preflight archive extracted to another directory and prepared/doctor/candidate/tests passed. Review fixed three launcher-only portability guards; controller unchanged. Clean-machine installation and full60 physical rerun are outside the completed validation scope.
