# FORK_NOTES — socrasteeze/taggui

This is a personal fork of [jhc13/taggui](https://github.com/jhc13/taggui).
This file is the always-current merge map: every place the fork diverges from
upstream, so `git merge upstream/main` stays a short job. Update it in the same
commit as any change that adds a new fork-only file or upstream touchpoint
(same convention as the sibling ai-toolkit and lora-dataset-studio forks).

`Plan.md` is the *roadmap* (captioning alignment to SDXL / Illustrious XL /
FLUX.2 Klein 9B / FLUX.1 Krea, captioner roster, token counters, bucketing —
with Done vs Deferred tracking). This file is the *divergence ledger* — what is
actually landed and where.

## Fork-only files (no upstream counterpart — merges never touch these)

- `Plan.md` — the modernization roadmap.
- `run.bat` — Windows bootstrap: creates/updates a venv, installs
  `requirements.txt`, pins `HF_HOME` to the local SSD cache, launches the app
  (`run.bat update` / `-u` forces a reinstall). Self-repairing: it pins the
  venv to CPython 3.11/3.12 (the only versions `requirements.txt` has Windows
  torch wheels for), records a `venv\.installed-requirements.txt` stamp only
  after a *verified* install so an interrupted one is retried rather than
  silently launching a dependency-less venv, and pauses on every failure path
  so errors stay readable when launched by double-click.
- `create_shortcut.bat` — one-click Desktop `.lnk` creator (points at
  `run.bat`, uses `images/icon.ico`).
- `.gitattributes` — forces CRLF for `.bat` files.
- `taggui/utils/bucketing.py` — aspect-ratio bucketing math, compatible with
  kohya `make_bucket_resolutions` (step 64, target area, min/max resolution,
  `--bucket_no_upscale` equivalent).
- `taggui/dialogs/bucket_calculator_dialog.py` — the Tools ▸ Aspect Ratio
  Bucket Calculator dialog: distribution table with upscale / heavy-crop /
  sparse-bucket warnings, plus the optional **Process Images into Buckets**
  action (moves originals to `original_images/`, writes resized+cropped PNGs
  in place, carries `.txt` captions along).
- `taggui/utils/dimension_cache.py` — disk cache of image dimensions.
- `taggui/utils/tag_writer.py` — background queue for `.txt` sidecar writes.
- `taggui/utils/caption_profiles.py` — SDXL / Illustrious / FLUX caption profiles.
- `taggui/utils/tag_vocab.py` — a1111-format CSV vocab loader + merged completer.
- `taggui/utils/create_shortcut.py` — Windows `.lnk` helper (WScript.Shell).
- `taggui/dialogs/caption_stats_dialog.py` — caption / token / trigger stats.
- `taggui/dialogs/trigger_token_dialog.py` — insert trigger token tooling.
- `taggui/dialogs/create_shortcut_dialog.py` — Tools ▸ Create Desktop Shortcut.
- `taggui/auto_captioning/models/qwen3_vl.py` — Qwen3-VL captioner.
- `FORK_NOTES.md` — this file.

## Upstream files with fork edits (merge conflicts concentrate here)

- `taggui/widgets/main_window.py` — Tools menu (bucket calculator, caption
  stats, update tag lists, create desktop shortcut, caption profile submenu);
  File export actions; Edit trigger / Illustrious reorder; View list/grid;
  async directory load progress; debounced filter; lazy tokenizer load;
  vocab wiring.
- `taggui/models/image_list_model.py` — async directory load worker, dimension
  cache, background thumbnails, sparse undo diffs, async tag writes, trigger
  insert, Illustrious reorder, JSONL / Kohya metadata export;
  `original_images/` exclusion.
- `taggui/models/proxy_image_list_model.py` — cached `tokens:` filter counts;
  optional lazy tokenizer.
- `taggui/models/tag_counter_model.py` — incremental tag counting; `tags`
  property for vocab completer.
- `taggui/widgets/image_tags_editor.py` — profile token limits; merged
  dataset+CSV autocomplete.
- `taggui/widgets/image_viewer.py` — decoded-image cache on resize.
- `taggui/widgets/image_list.py` — list/grid view mode.
- `taggui/widgets/auto_captioner.py` — JoyCaption tag-grounding toggle.
- `taggui/dialogs/settings_dialog.py` — caption profile + autocomplete mode.
- `taggui/dialogs/find_and_replace_dialog.py` — debounced match counts.
- `taggui/dialogs/batch_reorder_tags_dialog.py` — move tags to back;
  Illustrious order button.
- `taggui/auto_captioning/captioning_thread.py` — prefetch + WD batching.
- `taggui/auto_captioning/models/wd_tagger.py` — ONNX GPU providers + batch
  generate.
- `taggui/auto_captioning/models/joycaption.py` — tag-grounded prompts.
- `taggui/auto_captioning/models_list.py` — Qwen3-VL models; roster reorder.
- `taggui/utils/settings.py` — new defaults (profile, autocomplete mode, etc.).
- `taggui/utils/image.py` — `token_count` cache field.
- `requirements.txt` — note on optional `onnxruntime-gpu` / DirectML.

## Behavioural notes for merges

- Upstream has no `Tools` menu; if upstream ever adds one, fold the calculator
  action into theirs rather than keeping two.
- The `original_images/` exclusion in `image_list_model.py` and the bucket
  processor in the dialog are a **pair** — dropping one side orphans the other.
- The bucket processor **rewrites images in place** (PNG) and is meant for
  pre-training prep only; trainers that bucket at load time (ai-toolkit) don't
  need it — treat the dialog as a calculator first (see the stack-wide
  integration plan in lora-dataset-studio's `PLAN.md`, Phase 3).
- Directory load is async: callers must wait for `load_finished` before
  selecting rows (main_window handles this).
- Undo history items are sparse `{index: previous_tags}` dicts, not full
  dataset snapshots.

## Merge routine

```
git remote add upstream https://github.com/jhc13/taggui   # once
git fetch upstream && git merge upstream/main
# expected conflict surface: the edited upstream files above.
# Re-run the app afterwards:
#   run.bat            (Windows)
#   python taggui/run_gui.py
# and sanity-check: directory load (progress dialog), tag editing, WD tagger
# on GPU, Tools ▸ Aspect Ratio Bucket Calculator, caption profile, CSV
# type-ahead (Tools ▸ Update Tag Lists), export JSONL.
```
