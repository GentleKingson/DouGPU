# DouGPU source provenance and licenses

DouGPU is derived from the user-provided
`DouTPU_v6e1_Optimized_Final.ipynb`. The original MIT license terms and copyright
notice are retained in `LICENSE`, which also covers this project's original
modifications and additions. The notebook source inventory and original
SHA-256 hashes are in `docs/notebook_source_manifest.json`; those historical
hashes do not describe the current modified files.

The current project and Python package are named DouGPU and `dougpu`.
Original notebook filenames, copyright notices, historical evidence and the
legacy checkpoint/source-lock field `doutpu_sha256` retain their original names.

This distribution additionally includes the pinned DouZero CPU rules files and
optional official opponent model definition, taken from:

- Repository: <https://github.com/kwai/DouZero>
- Commit: `718a5c920bf3361e34178a38f3b80458e176b351`
- Upstream license: Apache License, Version 2.0 (`Apache-2.0`), in `vendor/LICENSE`
- Bundled source: `vendor/douzero/` and `upstream_cache/upstream_source.zip`
- Verified archive and file hashes: `upstream_cache/source_lock.json`

DouZero's license and notices apply to those upstream files. The upstream code
is not re-licensed by this project's MIT license. When redistributing these
files, retain the upstream license and applicable notices; any modified
upstream files must carry prominent notices stating that they were changed.

The distributed source package does not include pretrained weights, user
checkpoints, replay data, credentials, or DanLM/FableDan source. Local experiment
artifacts in ignored directories such as `reports/` and `runs/` are outside
that statement. Python dependencies and NVIDIA runtime libraries are installed
separately under their own licenses. This project's MIT license does not grant
rights to third-party weights or other separately obtained artifacts.
