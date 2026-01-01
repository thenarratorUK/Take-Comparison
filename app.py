import re
import io
import json
import hashlib
import random
from pathlib import Path
from itertools import combinations
from zipfile import ZipFile, BadZipFile
from datetime import datetime, timezone

import streamlit as st


APP_RESULTS_VERSION = 2  # persistence format version

AUDIO_MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}

ALLOWED_EXTS = set(AUDIO_MIME_BY_EXT.keys())
FILENAME_RE = re.compile(r"^(?P<line>.+)\.(?P<take>\d+)$")

# ZIP guardrails
MAX_ZIP_UNCOMPRESSED_BYTES = 1_200_000_000  # 1.2 GB extracted
MAX_ZIP_FILE_COUNT = 2_000

PERSIST_DIR = Path(".take_picker_persist")
PERSIST_DIR.mkdir(exist_ok=True)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_key_hash(user_key: str) -> str:
    return hashlib.sha256(user_key.encode("utf-8")).hexdigest()


def persist_path_for_key(user_key: str) -> Path:
    return PERSIST_DIR / f"{safe_key_hash(user_key)}.json"


def parse_uploaded_filename(name: str):
    """Expect: <lineKey>.<takeNumber>.<extension>.

    <lineKey> may contain dots. We interpret the last dot-separated stem component
    as the integer take number.
    """
    p = Path(name)
    stem = p.stem
    m = FILENAME_RE.match(stem)
    if not m:
        return None
    line_key = m.group("line").strip()
    take_num = int(m.group("take"))
    ext = p.suffix.lower()
    return line_key, take_num, ext


def stable_seed_for_line(line_key: str, take_ids_sorted: list[str]) -> int:
    payload = line_key + "|" + "|".join(take_ids_sorted)
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit seed


def init_state():
    st.session_state.setdefault("step", 1)  # 1=user key, 2=app
    st.session_state.setdefault("user_key", "")
    st.session_state.setdefault("library", {})      # take_id -> metadata + bytes
    st.session_state.setdefault("line_runs", {})    # line_key -> run state
    st.session_state.setdefault("selected_line", None)

    # persistence status
    st.session_state.setdefault("persist_available", None)  # None=unknown, bool after first write
    st.session_state.setdefault("persist_last_error", None)
    st.session_state.setdefault("persist_loaded_msg", None)
    st.session_state.setdefault("persist_autoload_attempted", False)


def build_library_from_name_bytes(file_items):
    lib = {}
    errors = []
    for name, b in file_items:
        parsed = parse_uploaded_filename(name)
        if not parsed:
            errors.append(f"Unrecognised filename format: {name}")
            continue

        line_key, take_num, ext = parsed
        if ext not in ALLOWED_EXTS:
            errors.append(f"Unsupported audio extension: {name}")
            continue

        take_id = f"{line_key}.{take_num}"
        mime = AUDIO_MIME_BY_EXT.get(ext, "audio/wav")

        lib[take_id] = {
            "take_id": take_id,
            "line": line_key,
            "take_number": take_num,
            "filename": name,
            "bytes": b,
            "mime": mime,
        }
    return lib, errors


def natural_sort_key(s: str):
    """
    Natural sort key: splits digits so 'Line2' < 'Line10'.
    """
    parts = re.split(r"(\d+)", s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.casefold())
    return key


def takes_by_line(library: dict):
    by_line = {}
    for take_id, meta in library.items():
        by_line.setdefault(meta["line"], []).append(take_id)

    for line_key in by_line:
        by_line[line_key].sort(key=lambda tid: (library[tid]["take_number"], tid))
    return by_line


def ensure_line_run(line_key: str, take_ids: list[str]):
    """Ensure a run exists for this line.

    If a run exists but its take_ids don't match current uploads, rebuild it.
    """
    runs = st.session_state.line_runs
    take_ids_sorted = sorted(take_ids)

    if line_key in runs:
        existing_ids = sorted(runs[line_key].get("scores", {}).keys())
        if existing_ids == take_ids_sorted:
            return
        runs.pop(line_key, None)

    seed = stable_seed_for_line(line_key, take_ids_sorted)
    rng = random.Random(seed)

    unordered_pairs = list(combinations(take_ids_sorted, 2))
    tests = []
    for a, b in unordered_pairs:
        tests.append((a, b) if rng.random() < 0.5 else (b, a))
    rng.shuffle(tests)

    runs[line_key] = {
        "seed": seed,
        "tests": tests,
        "idx": 0,
        "scores": {tid: 0 for tid in take_ids_sorted},
        "history": [],  # stack of {"idx": int, "winner": take_id}
        "results": [None for _ in range(len(tests))],  # per-test outcome
    }


def record_vote(line_key: str, winner_take_id: str):
    run = st.session_state.line_runs[line_key]
    idx = run["idx"]
    a, b = run["tests"][idx]
    if winner_take_id not in (a, b):
        raise ValueError("Winner must be one of the current pair.")

    run["scores"][winner_take_id] += 1
    run["history"].append({"idx": idx, "winner": winner_take_id})
    run["results"][idx] = {"a": a, "b": b, "winner": winner_take_id}
    run["idx"] += 1


def undo_last_vote(line_key: str):
    run = st.session_state.line_runs[line_key]
    if not run["history"]:
        return

    last = run["history"].pop()
    idx = last["idx"]
    winner = last["winner"]

    run["scores"][winner] -= 1
    run["results"][idx] = None
    run["idx"] = idx


def export_all_results_obj():
    payload = {
        "version": APP_RESULTS_VERSION,
        "created_utc": now_utc_iso(),
        "user_key_hash": safe_key_hash(st.session_state.user_key) if st.session_state.user_key else None,
        "line_runs": {},
    }
    for line_key, run in st.session_state.line_runs.items():
        payload["line_runs"][line_key] = {
            "seed": run["seed"],
            "tests": run["tests"],
            "scores": run["scores"],
            "results": run["results"],
            "idx": run["idx"],
            "history": run["history"],
        }
    return payload


def export_all_results_json():
    return json.dumps(export_all_results_obj(), indent=2)


def export_line_json(line_key: str):
    run = st.session_state.line_runs[line_key]
    payload = {
        "version": APP_RESULTS_VERSION,
        "created_utc": now_utc_iso(),
        "user_key_hash": safe_key_hash(st.session_state.user_key) if st.session_state.user_key else None,
        "line": line_key,
        "seed": run["seed"],
        "tests": run["tests"],
        "scores": run["scores"],
        "results": run["results"],
        "idx": run["idx"],
        "history": run["history"],
    }
    return json.dumps(payload, indent=2)


def _normalise_imported_run(line_key: str, run_obj: dict):
    required = ["seed", "tests", "scores", "results", "idx", "history"]
    for k in required:
        if k not in run_obj:
            raise ValueError(f"Line '{line_key}' missing key '{k}' in imported JSON.")

    tests = []
    for t in run_obj["tests"]:
        if not isinstance(t, (list, tuple)) or len(t) != 2:
            raise ValueError(f"Line '{line_key}' has invalid test entry: {t!r}")
        tests.append((t[0], t[1]))

    results = run_obj["results"]
    if not isinstance(results, list):
        raise ValueError(f"Line '{line_key}' results must be a list.")

    scores = run_obj["scores"]
    if not isinstance(scores, dict):
        raise ValueError(f"Line '{line_key}' scores must be a dict.")

    idx = int(run_obj["idx"])
    history = run_obj["history"]
    if not isinstance(history, list):
        raise ValueError(f"Line '{line_key}' history must be a list.")

    if len(results) != len(tests):
        raise ValueError(f"Line '{line_key}' results length does not match tests length.")
    if idx < 0 or idx > len(tests):
        raise ValueError(f"Line '{line_key}' idx out of range.")

    return {
        "seed": int(run_obj["seed"]),
        "tests": tests,
        "scores": {str(k): int(v) for k, v in scores.items()},
        "results": results,
        "idx": idx,
        "history": history,
    }


def import_results_json(json_bytes: bytes):
    obj = json.loads(json_bytes.decode("utf-8"))
    version = obj.get("version", None)
    if version != APP_RESULTS_VERSION:
        raise ValueError(f"Unsupported results JSON version: {version!r}")

    imported_runs = obj.get("line_runs")
    if not isinstance(imported_runs, dict):
        raise ValueError("Expected top-level key 'line_runs' to be a dict.")

    merged = 0
    for line_key, run_obj in imported_runs.items():
        st.session_state.line_runs[str(line_key)] = _normalise_imported_run(str(line_key), run_obj)
        merged += 1
    return merged


def persist_save_best_effort():
    """Attempt to save results on the server by user key.

    Local writes are best-effort. If the app restarts, the file may be lost.
    """
    if not st.session_state.user_key:
        return
    try:
        p = persist_path_for_key(st.session_state.user_key)
        p.write_text(export_all_results_json(), encoding="utf-8")
        st.session_state.persist_available = True
        st.session_state.persist_last_error = None
    except Exception as e:
        st.session_state.persist_available = False
        st.session_state.persist_last_error = str(e)


def persist_load_best_effort():
    if not st.session_state.user_key:
        return 0, False
    p = persist_path_for_key(st.session_state.user_key)
    if not p.exists():
        return 0, False
    try:
        merged = import_results_json(p.read_bytes())
        st.session_state.persist_available = True
        st.session_state.persist_last_error = None
        return merged, True
    except Exception as e:
        st.session_state.persist_available = False
        st.session_state.persist_last_error = str(e)
        return 0, False


def vote(side: str):
    line_key = st.session_state.selected_line
    run = st.session_state.line_runs[line_key]
    idx = run["idx"]
    a_id, b_id = run["tests"][idx]
    winner = a_id if side == "A" else b_id
    record_vote(line_key, winner)
    persist_save_best_effort()


def back():
    line_key = st.session_state.selected_line
    undo_last_vote(line_key)
    persist_save_best_effort()


def load_zip_to_items(zip_bytes: bytes):
    errors = []
    items = []

    try:
        zf = ZipFile(io.BytesIO(zip_bytes))
    except BadZipFile:
        return [], ["Uploaded file is not a valid ZIP archive."]

    infos = zf.infolist()

    if len(infos) > MAX_ZIP_FILE_COUNT:
        return [], [f"ZIP contains {len(infos)} entries; limit is {MAX_ZIP_FILE_COUNT}."]

    total_uncompressed = sum(i.file_size for i in infos)
    if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
        mb = total_uncompressed / (1024 * 1024)
        limit_mb = MAX_ZIP_UNCOMPRESSED_BYTES / (1024 * 1024)
        return [], [f"ZIP uncompressed size is ~{mb:.1f} MB; limit is {limit_mb:.1f} MB."]

    for info in infos:
        if info.is_dir():
            continue

        base_name = Path(info.filename).name

        if base_name in (".DS_Store",) or info.filename.startswith("__MACOSX/"):
            continue

        ext = Path(base_name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue

        try:
            items.append((base_name, zf.read(info)))
        except Exception as e:
            errors.append(f"Failed to read {info.filename}: {e}")

    if not items:
        errors.append("No supported audio files found inside the ZIP.")
    return items, errors


def build_head_to_head_map(line_key: str):
    run = st.session_state.line_runs[line_key]
    h2h = {}
    for r in run["results"]:
        if r is None:
            continue
        a = r["a"]
        b = r["b"]
        key = tuple(sorted((a, b)))
        h2h[key] = r["winner"]
    return h2h


def head_to_head_winner(h2h: dict, x: str, y: str):
    key = tuple(sorted((x, y)))
    return h2h.get(key)


def mini_league_points(h2h: dict, group: list[str]):
    pts = {tid: 0 for tid in group}
    for a, b in combinations(group, 2):
        w = head_to_head_winner(h2h, a, b)
        if w is None:
            continue
        pts[w] += 1
    return pts


def stable_fallback_order(seed: int, ids: list[str]) -> list[str]:
    rng = random.Random(seed ^ 0xA5A5A5A5)
    ids2 = list(ids)
    rng.shuffle(ids2)
    return ids2


def order_with_tiebreaks(line_key: str, take_ids: list[str]):
    run = st.session_state.line_runs[line_key]
    seed = run["seed"]
    scores = run["scores"]
    h2h = build_head_to_head_map(line_key)

    groups = {}
    for tid in take_ids:
        groups.setdefault(scores[tid], []).append(tid)

    ordered_points = sorted(groups.keys(), reverse=True)
    final_order = []
    tie_rank = {}

    for pts in ordered_points:
        members = groups[pts]
        if len(members) == 1:
            tid = members[0]
            final_order.append(tid)
            tie_rank[tid] = 1
            continue

        mini_pts = mini_league_points(h2h, members)

        buckets = {}
        for tid in members:
            buckets.setdefault(mini_pts[tid], []).append(tid)

        mini_points_desc = sorted(buckets.keys(), reverse=True)
        rank_counter = 1

        for mp in mini_points_desc:
            sub = buckets[mp]
            if len(sub) == 1:
                tid = sub[0]
                final_order.append(tid)
                tie_rank[tid] = rank_counter
                rank_counter += 1
            elif len(sub) == 2:
                x, y = sub[0], sub[1]
                w = head_to_head_winner(h2h, x, y)
                if w is None:
                    ordered = sorted(sub)
                else:
                    ordered = [w, y if w == x else x]
                for tid in ordered:
                    final_order.append(tid)
                    tie_rank[tid] = rank_counter
                    rank_counter += 1
            else:
                ordered = stable_fallback_order(seed, sorted(sub))
                for tid in ordered:
                    final_order.append(tid)
                    tie_rank[tid] = rank_counter
                    rank_counter += 1

    return final_order, tie_rank


# ---------------- UI ----------------

st.set_page_config(page_title="Take Picker", layout="wide")
init_state()

st.title("Take Picker (Blind A/B Comparisons)")

if st.session_state.step == 1:
    st.subheader("Step 1: User key")
    st.write("Type any key you want. This key is used to auto-save and (when possible) auto-load your progress.")
    st.text_input("User key", key="user_key_input", placeholder="e.g. david-2026-01-01-reels")

    proceed_disabled = not st.session_state.get("user_key_input", "").strip()
    if st.button("Continue to Step 2", use_container_width=True, disabled=proceed_disabled):
        st.session_state.user_key = st.session_state.user_key_input.strip()
        merged, ok = persist_load_best_effort()
        if ok and merged > 0:
            st.session_state.persist_loaded_msg = f"Loaded {merged} line(s) from saved progress for this key."
        else:
            st.session_state.persist_loaded_msg = "No saved progress found for this key (starting fresh)."
        st.session_state.step = 2
        st.rerun()

    st.stop()

# Step 2: everything else
st.subheader("Step 2: Upload and compare")

if st.session_state.persist_loaded_msg:
    st.info(st.session_state.persist_loaded_msg)
    st.session_state.persist_loaded_msg = None

top_row_a, top_row_b = st.columns([1, 1])
with top_row_a:
    st.button("Change user key", use_container_width=True, on_click=lambda: st.session_state.update({"step": 1}))
with top_row_b:
    st.caption(f"User key: {st.session_state.user_key}")

with st.expander("Import JSON backup (restore progress)", expanded=True):
    uploaded_results = st.file_uploader(
        "Import JSON backup (restores/overwrites line progress)",
        type=["json"],
        accept_multiple_files=False,
        key="results_json_uploader_main",
    )
    if uploaded_results is not None:
        try:
            merged = import_results_json(uploaded_results.getvalue())
            persist_save_best_effort()
            st.success(f"Imported {merged} line(s) from JSON backup.")
        except Exception as e:
            st.error(f"Could not import results JSON: {e}")

    if st.session_state.persist_available is False and st.session_state.persist_last_error:
        st.warning(
            "Server-side auto-save is not available in this deployment (write failed). "
            "Use the full-progress JSON download button at the bottom of the page frequently.\n\n"
            f"Error: {st.session_state.persist_last_error}"
        )

st.divider()

st.subheader("Upload audio")

upload_mode = st.radio(
    "Upload method",
    ["Multiple audio files", "ZIP archive of audio files"],
    horizontal=True,
)

if upload_mode == "Multiple audio files":
    uploaded_files = st.file_uploader(
        "Upload audio takes (format: LineKey.takeNumber.ext, e.g. Line1.6.mp3)",
        accept_multiple_files=True,
        type=[ext.lstrip(".") for ext in sorted(ALLOWED_EXTS)],
    )
    if uploaded_files:
        items = [(uf.name, uf.getvalue()) for uf in uploaded_files]
        new_lib, errors = build_library_from_name_bytes(items)
        if errors:
            st.error("Some files were ignored:\n\n" + "\n".join(f"- {e}" for e in errors))
        st.session_state.library.update(new_lib)

else:
    uploaded_zip = st.file_uploader(
        "Upload a ZIP containing audio takes (filenames inside must be LineKey.takeNumber.ext)",
        accept_multiple_files=False,
        type=["zip"],
    )
    if uploaded_zip is not None:
        items, zip_errors = load_zip_to_items(uploaded_zip.getvalue())
        if zip_errors:
            st.error("ZIP issues:\n\n" + "\n".join(f"- {e}" for e in zip_errors))
        if items:
            new_lib, errors = build_library_from_name_bytes(items)
            if errors:
                st.error("Some files were ignored:\n\n" + "\n".join(f"- {e}" for e in errors))
            st.session_state.library.update(new_lib)

if not st.session_state.library:
    st.info("Upload audio files to begin comparisons.")
    st.divider()
    st.subheader("Download all progress")
    st.download_button(
        "Download all progress (JSON)",
        data=export_all_results_json(),
        file_name="take_picker_results_all_lines.json",
        mime="application/json",
        use_container_width=True,
        disabled=(len(st.session_state.line_runs) == 0),
    )
    st.stop()

st.divider()
st.subheader("Compare")

by_line = takes_by_line(st.session_state.library)
available_lines = sorted(by_line.keys())

if not available_lines:
    st.error("No valid lines were found in the uploaded files. Check filename format.")
    st.stop()

if st.session_state.selected_line not in available_lines:
    st.session_state.selected_line = available_lines[0]

st.selectbox(
    "Select line to compare",
    available_lines,
    key="selected_line",
)

selected_line = st.session_state.selected_line
take_ids_for_line = by_line.get(selected_line, [])
if len(take_ids_for_line) < 2:
    st.warning("This line has fewer than 2 takes, so there are no comparisons to run.")
    st.stop()

ensure_line_run(selected_line, take_ids_for_line)

run = st.session_state.line_runs[selected_line]
total_tests = len(run["tests"])
idx = run["idx"]

st.write(f"Progress: Test {min(idx + 1, total_tests)} of {total_tests}")

st.caption(
    "If iOS Safari drops the connection after backgrounding, refresh the page, re-enter your user key, "
    "re-upload the audio, then continue."
)

completed_line = idx >= total_tests

if completed_line:
    st.subheader(f"{selected_line} Ranking (with tie-break ranks)")

    ordered, tie_rank = order_with_tiebreaks(selected_line, take_ids_for_line)

    scores = run["scores"]
    rows = []
    # Only show tie-break ranks when there is an actual tie on points.
    point_group_sizes = {}
    for tid in take_ids_for_line:
        p = scores[tid]
        point_group_sizes[p] = point_group_sizes.get(p, 0) + 1

    for tid in ordered:
        p = scores[tid]
        if point_group_sizes.get(p, 0) > 1:
            points_display = f"{p}({tie_rank.get(tid, 1)})"
        else:
            points_display = str(p)

        rows.append({
            "Take": tid,
            "Points": points_display,
            "Filename": st.session_state.library[tid]["filename"],
        })

    st.table(rows)

    # Next line navigation (natural sort order)
    try:
        current_idx = available_lines.index(selected_line)
    except ValueError:
        current_idx = -1

    has_next = (current_idx >= 0 and current_idx < len(available_lines) - 1)
    if has_next:
        if st.button("Next line", use_container_width=True):
            st.session_state.selected_line = available_lines[current_idx + 1]
            st.rerun()
    else:
        st.caption("No next line (this is the last line in the list).")

else:
    a_id, b_id = run["tests"][idx]
if not completed_line:
    lib = st.session_state.library

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("A")
        st.audio(lib[a_id]["bytes"], format=lib[a_id]["mime"])
    with col_b:
        st.subheader("B")
        st.audio(lib[b_id]["bytes"], format=lib[b_id]["mime"])

    st.write("Which do you prefer?")

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col1:
        st.button("Prefer A", use_container_width=True, on_click=vote, args=("A",))
    with btn_col2:
        st.button("Prefer B", use_container_width=True, on_click=vote, args=("B",))
    with btn_col3:
        st.button(
            "Back",
            use_container_width=True,
            disabled=(len(run["history"]) == 0),
            on_click=back,
        )

st.divider()
st.subheader("Download all progress")

st.download_button(
    "Download all progress (JSON)",
    data=export_all_results_json(),
    file_name="take_picker_results_all_lines.json",
    mime="application/json",
    use_container_width=True,
    disabled=(len(st.session_state.line_runs) == 0),
)

