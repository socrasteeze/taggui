# Handoff: SOTA anime/booru tagging + duplicate taglist (2026-10-25)

Prepared for socrasteeze. Status: **spec only, no code changed.**
Reviewed 2026-10-25 against the code and the model cards; corrections from that review are applied below.
Base: `main` @ `b18ea91` (`v1.34.0-17-gb18ea91`), fork `socrasteeze/taggui`, upstream `jhc13/taggui`
(upstream latest release is still v1.34.0, 2025-10-11; this fork is 0 commits behind upstream; not re-checked in review).
Transformers pin: 4.57.6. Test suite: headless, no torch (`pytest`). The app's `venv` has no pytest, so install
`requirements-dev.txt` into an environment before running the suite.

House rules for whoever builds this:
- Never create `.bak` files. Never write outside this repository.
- Every item adds tests under `tests/` and a line in `README.md` and `FORK_CHANGELOG.md`.
- Effort: **S** = under half a day, **M** = 1-2 days, **L** = 3 or more days. Priority: **P0** (do first) to **P3** (optional).
- Benchmark figures below are the model authors' own numbers unless stated otherwise. **PixAI's numbers are self-reported**
  and not independently confirmed (https://isitsota.com/m/pixai-tagger-1-0, "pending benchmarks").

---

## 1. Review summary (why this work)

Current roster (`taggui/auto_captioning/models_list.py`): JoyCaption Beta One, Qwen3-VL 2B/4B/8B/30B-A3B,
Gemma 4 (listed, but not runnable on the 4.57.6 pin), Florence-2 + PromptGen v2.0, `deepghs/pixai-tagger-v0.9-onnx`,
WD v3 (EVA02-L / ViT-L / SwinV2 / ConvNeXt / ViT), Phi-3-Vision, plus legacy models.
Caption profiles (`taggui/utils/caption_profiles.py`): SDXL, Illustrious XL, FLUX.2 Klein 9B, FLUX.1 Krea.

This was SOTA around Aug 2026. The gaps now:
- PixAI Tagger **v1.0** (release date not stated on the model card): 30,877 tags, 1008px input, May 2026 cutoff,
  style/meta/rating categories.
  Self-reported general micro-F1 0.6660 and character micro-F1 0.9242, vs v0.9 0.5980 / 0.8198.
- AnimeTIMM dbv4 taggers beat WD v3 in the same (PixAI-run) shared-tag benchmark.
- ToriiGate-0.5 anime captioner is not supported.
- No **Anima** caption profile.
- JoyCaption Beta One is still the latest JoyCaption, so no change is needed there.

Existing duplicate-related features (from upstream):
- Image list **Copy Tags / Paste Tags** (Ctrl+C / Ctrl+V), in `taggui/widgets/image_list.py` around lines 117-216.
  Paste goes to `ImageListModel.add_tags`, which uses `image.tags.extend(tags)`. Problems confirmed by reading the code
  (not yet reproduced in the running app):
  - Pasting creates duplicate tags, because tags already on an image aren't skipped (`image_list_model.py:676`).
  - A multi-image copy joins captions with `\n` (`image_list.py:191`), while paste splits only on the tag separator
    (`image_list.py:214`). That produces tags with embedded newlines.
  - Pasting from an empty clipboard adds an empty tag (`''.split(separator)` returns `['']`).
  - There's no replace or prepend mode.
  - Not a bug: a multi-image paste asks for confirmation once. `add_to_undo_stack` only stores a flag on the history item;
    that flag prompts on undo (`image_list_model.py:365`), the same as every other multi-image action.
- **Edit > Remove Duplicate Tags** (Ctrl+D), `ImageListModel.remove_duplicate_tags`: exact string match only.

---

## 2. Items

### Item 1: PixAI Tagger v1.0 (P0, M)
- **Decide first: loading path.** Every tagger today (WD, pixai v0.9) runs on ONNX Runtime without torch, which keeps the tagger
  tests headless. Pick one, and record the choice here:
  - **Official torch pipeline:** `pipeline(..., trust_remote_code=True)` on `pixai-labs/pixai-tagger-v1.0` (weights:
    `model.safetensors`, 1.95 GB; no ONNX file). Puts torch into the tagger layer, so tests must mock the pipeline.
  - **Community ONNX export** (several exist, none from deepghs). The one checked, `noaione/pixai-tagger-v1.0-onnx`, ships
    `model.onnx` + `model.onnx.data` + `tags.json`, with no `preprocess.json` or `thresholds.csv`. It doesn't fit
    `PixaiTaggerModel` as is: it needs a tags/thresholds adapter and the 1008px resize-and-pad preprocessing re-implemented.
    Unofficial, so pin a revision.
- **What:** add a tagger class for v1.0. Thresholds live in the model's `config.json`: `category_best_threshold` per category and
  `tags_best_threshold` per tag (per-tag wins when set; fallback 0.2). The model card's per-category values are
  general 0.17, character 0.27, style 0.15, copyright 0.24, meta 0.17, rating 0.41. Reuse the existing
  "Use the model's thresholds" checkbox (`tagger_use_model_thresholds`, `widgets/auto_captioner.py:148`) rather than adding
  new threshold controls; add include toggles for style/meta/rating only.
- **Files:** new `taggui/auto_captioning/models/pixai_tagger_v1.py`; `models_list.py` (route `pixai-tagger-v1` **before**
  the v0.9 rule, which matches any id containing `pixai` and `tagger`; add the class to `TAGGER_CLASS_NAMES`);
  `widgets/auto_captioner.py` (style/meta/rating include toggles); `utils/settings.py`; `requirements.txt` (pin
  `torchvision` if the torch path is chosen: `tagger_pipeline.py` imports it, and it is only present in the venv as a side effect).
- **Acceptance:** the routing test resolves v1.0 and v0.9 to different classes. Mocked scores are filtered correctly by per-tag and
  per-category thresholds and the category toggles. Tag names: the pipeline emits `config.tags` unchanged, so check
  whether they use underscores; if they do, convert with `tag_utils.normalize_tag` (keeps kaomoji), matching the existing WD behavior.
- **Risks:** low transformers risk. `tagger_pipeline.py` defines its own ViTDet backbone and imports only APIs already in 4.57
  (`AutoModel`, `Pipeline`, `PreTrainedModel`, `BaseImageProcessor`, `PIPELINE_REGISTRY`, ...), with a fallback for older timm.
  So this item doesn't depend on Item 5; confirm on hardware. 1008px input means heavy VRAM use, so expose the batch size setting.
  Model-card link: https://huggingface.co/pixai-labs/pixai-tagger-v1.0

### Item 2: Anima caption profile (P0, S-M)
- **What:** add `CaptionProfile.ANIMA`. Tag order per the model card: quality/meta/year/safety, then count (1girl...), character,
  series, `@artist`, general. Lowercase, spaces instead of underscores (`score_N` is the only exception). The card says the
  `@` artist prefix is required, so turn it on by default in this profile (user can turn it off). Optional output format
  `tags. natural-language prose` on **one line**, shared with Item 11: whichever item lands first writes the split/join helpers
  in `models/image_list_model.py`, and the other reuses them. Newline guard (warn or strip):
  Anima reserves "dataset tag + newline" as the first line for its non-anime datasets, and kohya sd-scripts keeps only the
  first line of a caption.
- **Category sources:** count tags by the existing regex; character, series and artist from
  `TagVocab.get_names_in_categories` (`utils/tag_vocab.py:118`: 1 artist, 3 copyright/series, 4 character), so this item does
  **not** wait for Item 8. Safety tags (`safe`, `sensitive`, `nsfw`, `explicit`) have no source today: WD reads its rating
  tags (category 9, `general`/`sensitive`/`questionable`/`explicit`) but drops them from the output
  (`wd_tagger.py:44-58`), and pixai v0.9 emits no rating tags either (`pixai_tagger_model.py:142`). Options: an opt-in "emit rating" setting on WD
  (and on v1.0, which has a rating category) that maps to Anima's names, or manual tagging. Confirm the mapping against the card.
- **Files:** `utils/caption_profiles.py`; the reorder logic in `models/image_list_model.py` (extend the Illustrious reorder with
  safety and artist categories); `dialogs/batch_reorder_tags_dialog.py`; the newline guard goes where tags are joined into
  caption text, `ImageListModel.write_image_tags_to_disk` (`models/image_list_model.py:352`), not `utils/tag_writer.py`,
  which only queues file writes.
- **Acceptance:** reorder tests for each category; `@` prefix applied to artist-category tags (on by default for Anima);
  rating-to-safety mapping test (if the emit-rating option is built); newline guard test.
- **Risks:** the text encoder is Qwen3 0.6B (`qwen_3_06b_base.safetensors` per the card), so use the existing
  `TokenEncoder.QWEN` (`utils/caption_profiles.py:18`). The token limit isn't stated on the card; show the token count as
  approximate until it is confirmed.
  Links: https://huggingface.co/circlestone-labs/Anima |
  https://huggingface.co/circlestone-labs/Anima/discussions/205

### Item 3: AnimeTIMM taggers (P1, S)
- **What:** add `animetimm/caformer_b36.dbv4-full` (fast, 384px), `animetimm/eva02_large_patch14_448.dbv4-full`, and the
  giant variants (SigLIP Giant / EVA Giant) as accurate options. Run them on ONNX Runtime, not timm: the repos ship
  `model.onnx`, `preprocess.json`, `selected_tags.csv` and `thresholds.csv` (checked on `caformer_b36.dbv4-full`), the same
  four file names `PixaiTaggerModel` (`auto_captioning/models/pixai_tagger_model.py`) and `utils/onnx_preprocess.py` already read.
  The work is mostly routing plus fixtures. The file contents weren't opened (gated), and the giant variants weren't checked.
- **Files:** `models_list.py` (routing, `TAGGER_CLASS_NAMES`); a thin class or a generalised `PixaiTagger` only if the
  file formats differ from the pixai v0.9 export (CSV columns, the `stages` key in `preprocess.json`, ONNX output names).
- **Acceptance:** preprocess and threshold parsing tested on fixture JSON/CSV files copied from an AnimeTIMM repo; routing test.
- **Risks:** the repos are **gated**: the user must accept the conditions on Hugging Face and be logged in before the first
  download, so show a clear error that says so instead of a raw 401. The giant models need lots of VRAM.
  Link: https://huggingface.co/animetimm

### Item 4: Duplicate taglist (P0, M)
- **What:**
  - `ImageListModel.apply_tags(tags, indices, mode)` with modes `append_skip_existing` (default), `replace`, `prepend`. One undo entry.
  - Image-list action **Apply Current Image's Tags to Selection...** (no clipboard round-trip).
  - New dialog for mode, tag preview, and per-tag exclude checkboxes.
  - Fix paste: split on newlines as well as the separator, strip whitespace, drop empty tags (including an empty clipboard),
    route through `apply_tags`. Keep the existing single paste confirmation.
  - **Normalized Remove Duplicate Tags:** case, underscore vs space, repeated whitespace, and alias variants (aliases from
    `utils/tag_vocab.py`). Keeps the first occurrence. Alias matching only works once a tag CSV is loaded:
    `TagVocab.resolve` (`utils/tag_vocab.py:115`) knows no aliases before `load_csv`.
  - **Find Images with Identical Tag Lists** report: groups of images, with a click to select each group.
- **Files:** `models/image_list_model.py`, `widgets/image_list.py`, `widgets/main_window.py`, new
  `dialogs/apply_tags_dialog.py`, `utils/tag_vocab.py`, `tests/test_apply_tags.py`.
- **Acceptance:** each mode is tested, including undo restoring every image; pasting a multi-line clipboard produces no tags containing `\n`;
  pasting an empty clipboard changes nothing; normalized dedup treats `long_hair`, `Long Hair` and `long  hair` as one tag
  with no vocab loaded, and merges aliases only with a vocab loaded; identical-list grouping is tested.

### Item 5: transformers 5.x bump, then ToriiGate-0.5 (P1, L)
- **What:** raise the pin to `>=5.5` (the fork's `transformers_compat.py` already supports 5.x). This unblocks Gemma 4. Then add
  `Minthy/ToriiGate-0.5` (`Qwen3_5ForConditionalGeneration`) with grounding from existing tags and character names, using
  `{tags}` in the prompt.
- **Files:** `requirements.txt`, `auto_captioning/transformers_compat.py`, new `auto_captioning/models/toriigate.py`,
  `models_list.py`.
- **Acceptance:** import and roster tests pass on 5.x; per-model minimum-version checks report clearly.
- **Risks:** the `trust_remote_code` captioners named in `requirements.txt` (Florence-2, Phi-3-Vision, Moondream 1,
  InstructBLIP, BLIP-2, Kosmos-2) are only tested on 4.x and may break. Mark broken ones legacy and "needs 4.x" rather than
  silently failing. The bump may also force a torch upgrade from 2.8.0, which means new torch and flash-attn wheel URLs in
  `requirements.txt` (the flash-attn wheels are built per torch version). Keep this item a **separate, clearly labelled change**
  so it's easy to revert.
  Link: https://huggingface.co/Minthy/ToriiGate-0.5

### Item 6: Tagger ensemble (P1, M)
- **What:** run several taggers per image. Merge modes: union, intersection, majority vote (at least k of n), mean or max confidence
  against a threshold. Normalize tag names first (rating formats, underscores).
- **Files:** new `auto_captioning/ensemble.py`, `auto_captioning/captioning_thread.py`, `widgets/auto_captioner.py`.
- **Acceptance:** merge tests with synthetic score dictionaries for every mode.
- **Risks:** VRAM use when several models are loaded at once; offer to load them one after another. Reference: BooruDatasetTagManager combine
  modes, https://github.com/starik222/BooruDatasetTagManager

### Item 7: OpenAI-compatible endpoint captioner (P1, S-M)
- **What:** settings for base URL, model, API key (optional), max megapixels, temperature, concurrency, and system prompt.
  Images are sent as base64. Supports `{tags}` in prompts and strips `<think>` blocks. Works with LM Studio, vLLM, and ToriiGate served through vLLM,
  with no change to the transformers pin.
- **Files:** new `auto_captioning/models/openai_endpoint.py`, `models_list.py`, `widgets/auto_captioner.py`,
  `utils/settings.py`.
- **Acceptance:** tests against a mocked HTTP endpoint: request shape, think-block stripping, timeout handling.
- **Risks:** never log the API key; save it in settings only if the user opts in. References: BooruDatasetTagManager v2.6.0
  (https://github.com/starik222/BooruDatasetTagManager/releases); ToriiGate `scripts/caption_distributed.py`.

### Item 8: Tag categories (P1, M)
- **What:** assign categories from the downloaded Danbooru/e621 CSVs, with per-tag user overrides saved in a dataset sidecar JSON.
  Color-code tags in the Image Tags and All Tags panes, add a category filter term, and add a category-order mode to Batch Reorder.
  Item 2 can switch to it once it exists, adding user overrides; neither Item 2 nor Item 6 waits for it.
- **Files:** `utils/tag_vocab.py`, `models/image_tag_list_model.py`, `models/tag_counter_model.py`,
  `widgets/image_tags_editor.py`, `widgets/all_tags_editor.py`, `dialogs/batch_reorder_tags_dialog.py`.
- **Acceptance:** category lookup, override persistence and filter-term tests.

### Item 9: Weighted tags (P3, M)
- **What:** parse `(tag:1.2)`, `((tag))` and `[tag]` into a weight, show a weight control, and write the weights back. Off by default.
- **Risks:** most trainers ignore weights; this only matters for prompt-style captions.

### Item 10: Tag translation (P3, S-M)
- **What:** an optional column showing translated tag names, with a pluggable translation service and a local cache. Display only; never written to captions.

### Item 11: Natural-language caption mode (P1, M)
- **What:** toggle the Image Tags pane between the tag list and a free-text editor for the prose part. Saved as
  `tags. prose` on one line. The token counter covers both parts. Fits the Anima and FLUX profiles. Shares the split/join
  helpers with Item 2 (see there).
- **Files:** `widgets/image_tags_editor.py`, `models/image_list_model.py` (split and join helpers, and the caption join in
  `write_image_tags_to_disk`; `utils/tag_writer.py` only queues file writes).
- **Acceptance:** splitting and joining round-trip without loss; no newlines written.
- Reference: https://github.com/Overtime-Artist/taggui-plus

### Item 12: Near-duplicate image finder (P2, M)
- **What:** Tools > Find Duplicates using perceptual hashing (`imagehash`) with a strictness slider, defaulting to 0 (exact).
  Groups shown side by side. For each group: keep one, then send the rest to the Recycle Bin, move them to a folder, or tag them `duplicate`.
- **Files:** new `dialogs/find_duplicates_dialog.py`, a hash cache next to `utils/dimension_cache.py`, `requirements.txt`.
- **Risks:** variant sets (edits, recolors) get grouped at loose settings, so default to strict. Deleting must go to the Recycle Bin,
  never be permanent. Reference: taggui-plus.

### Item 13: Mark images complete (P2, S)
- **What:** Ctrl+K marks complete, Ctrl+Shift+K marks incomplete. Saved in a dataset sidecar. Adds a `complete:` filter term and a
  "Complete" label next to the token count. Reference: taggui-plus.

### Item 14: Camie Tagger v2 (P3, S-M)
- **What:** an optional ONNX tagger with 70,527 tags and micro/macro threshold profiles. Low priority: it ranked last in PixAI's
  shared-tag benchmark (self-reported by PixAI). Link: https://huggingface.co/Camais03/camie-tagger-v2

### Item 15: Tagger ban list and keep list (P1, S)
- **Already exists in part:** the tagger settings have "Tags to exclude" (`wd_tagger_tags_to_exclude`,
  `widgets/auto_captioner.py:162`, parsed in `auto_captioning/tag_utils.py`), applied per tagger. Extend it rather than adding
  a second ban list.
- **What:** add wildcards to the existing exclude list (for example character names for a style LoRA, or `watermark` /
  `signature`), add an optional keep list (whitelist), and apply both after all taggers and the ensemble.
- **Acceptance:** filter tests, including wildcards.

### Item 16: Threshold presets (P2, S)
- **What:** precision, balanced and recall presets for each tagger, taken from each model's published micro- and macro-F1 thresholds.

### Item 17: Dataset doctor report (P2, S-M)
- **What:** report trigger-word coverage, captions containing newlines, very rare tags, captions over the token limit, and images with identical
  tag lists. Extends `dialogs/caption_stats_dialog.py`.

### Item 18: Tag wiki lookup (P3, S)
- **What:** a right-click action on a tag that opens its Danbooru wiki page. Reference: taggui-plus.

---

## 3. Build order
1. Items 4, 2, 1, 15 (P0 core plus the cheap ban list)
2. Items 7, 6, 3, 8, 11
3. Item 5 (transformers 5.x, a separate labelled change because it's the riskiest)
4. Items 12, 13, 16, 17
5. Items 9, 10, 14, 18 (P3)

## 4. Test plan
- Unit tests (headless, no torch) for every item, as listed under each item's Acceptance. Run them from an environment with
  `requirements-dev.txt` installed (the app's `venv` has no pytest).
- Model routing tests in `tests/` for every new model id.
- Must be checked on hardware (GPU plus weights), not in CI:
  - PixAI v1.0 load (torch pipeline or ONNX export, per the Item 1 decision) and VRAM use at batch 1/4/8
  - AnimeTIMM giant models
  - ToriiGate on 5.x
  - Regression check after the bump of every `trust_remote_code` captioner: Florence-2, Phi-3-V, Moondream 1,
    InstructBLIP, BLIP-2 and Kosmos-2
- Manual test of item 4 on a copy of a real dataset folder (never the original): append, replace and prepend; undo; multi-line paste.

## 5. Out of scope
- Training or fine-tuning taggers. Hosted or paid APIs as defaults. Video tagging. Background removal and cropping tools.
- Pushing to GitHub or opening upstream PRs (need the maintainer's separate approval).
- Re-benchmarking models independently; numbers here are cited, not measured.

## 6. Sources
- PixAI v1.0: https://huggingface.co/pixai-labs/pixai-tagger-v1.0 (self-reported) | https://isitsota.com/m/pixai-tagger-1-0
- PixAI v1.0 community ONNX export (example checked): https://huggingface.co/noaione/pixai-tagger-v1.0-onnx
- PixAI v0.9: https://huggingface.co/pixai-labs/pixai-tagger-v0.9
- AnimeTIMM: https://huggingface.co/animetimm
- Camie v2: https://huggingface.co/Camais03/camie-tagger-v2
- ToriiGate-0.5: https://huggingface.co/Minthy/ToriiGate-0.5
- JoyCaption Beta One (still latest): https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava
- Anima: https://huggingface.co/circlestone-labs/Anima
- Upstream taggui releases: https://github.com/jhc13/taggui/releases
- BooruDatasetTagManager: https://github.com/starik222/BooruDatasetTagManager
- taggui-plus fork: https://github.com/Overtime-Artist/taggui-plus
