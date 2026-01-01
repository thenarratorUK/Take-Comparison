import re
import io
import json
import hashlib
import random
from pathlib import Path
from itertools import combinations
from zipfile import ZipFile, BadZipFile

import streamlit as st


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

# Safety guardrails for ZIP uploads.
MAX_ZIP_UNCOMPRESSED_BYTES = 1_200_000_000  # 1.2 GB total extracted size
MAX_ZIP_FILE_COUNT = 2_000


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
    st.session_state.setdefault("library", {})      # take_id -> metadata + bytes
    st.session_state.setdefault("line_runs", {})    # line_key -> run state
    st.session_state.setdefault("selected_line", None)


def build_library_from_name_bytes(file_items):
    """Build/extend library from iterable of (name, bytes)."""
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


def takes_by_line(library: dict):
    by_line = {}
    for take_id, meta in library.items():
        by_line.setdefault(meta["line"], []).append(take_id)

    for line_key in by_line:
        by_line[line_key].sort(key=lambda tid: (library[tid]["take_number"], tid))
    return by_line


def ensure_line_run(line_key: str, take_ids: list[str]):
    runs = st.session_state.line_runs
    if line_key in runs:
        return

    take_ids_sorted = sorted(take_ids)
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


def export_line_json(line_key: str):
    run = st.session_state.line_runs[line_key]
    payload = {
        "line": line_key,
        "seed": run["seed"],
        "tests": run["tests"],
        "scores": run["scores"],
        "results": run["results"],
    }
    return json.dumps(payload, indent=2)


def vote(side: str):
    line_key = st.session_state.selected_line
    run = st.session_state.line_runs[line_key]
    idx = run["idx"]
    a_id, b_id = run["tests"][idx]
    winner = a_id if side == "A" else b_id
    record_vote(line_key, winner)


def back():
    line_key = st.session_state.selected_line
    undo_last_vote(line_key)


def load_zip_to_items(zip_bytes: bytes):
    """Return (items, errors) where items is list[(basename, bytes)]."""
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
    """Return dict[(low, high)] -> winner_take_id for completed line."""
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
    """Return points within group, 1 point per head-to-head win."""
    pts = {tid: 0 for tid in group}
    for a, b in combinations(group, 2):
        w = head_to_head_winner(h2h, a, b)
        if w is None:
            continue
        pts[w] += 1
    return pts


def stable_fallback_order(seed: int, ids: list[str]) -> list[str]:
    """Deterministic order used only if mini-league cannot separate 3+ items."""
    rng = random.Random(seed ^ 0xA5A5A5A5)
    ids2 = list(ids)
    rng.shuffle(ids2)
    return ids2


def order_with_tiebreaks(line_key: str, take_ids: list[str]):
    """Return:
      - ordered list of take_ids for full line (points desc, then tiebreak within ties)
      - tie_rank: dict[take_id] -> rank within its main-points group (1-based)

    Tie-break logic for each points group:
      - If 2 takes are tied: use head-to-head winner.
      - If 3+ are tied: compute mini-league points within that tied group (1 point per win),
        then resolve any remaining 2-way ties by head-to-head, and any remaining 3+ ties
        by deterministic fallback order.
    """
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


st.set_page_config(page_title="Take Picker", layout="wide")
init_state()

st.title("Take Picker (Blind A/B Comparisons)")

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
    st.info("Upload some audio files to begin.")
    st.stop()

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

col_r1, col_r2 = st.columns([1, 5])
with col_r1:
    st.button("Rerun", use_container_width=True, on_click=st.rerun)
with col_r2:
    st.caption("If Safari gets stuck CONNECTING after returning from another app, refresh usually recovers. Mobile browsers may drop WebSocket connections while backgrounded.")

if idx >= total_tests:
    st.subheader(f"{selected_line} Ranking (with tie-break ranks)")

    ordered, tie_rank = order_with_tiebreaks(selected_line, take_ids_for_line)

    scores = run["scores"]
    rows = []
    for tid in ordered:
        rows.append({
            "Take": tid,
            "Points": f"{scores[tid]}({tie_rank.get(tid, 1)})",
            "Filename": st.session_state.library[tid]["filename"],
        })

    st.table(rows)

    st.download_button(
        label="Download JSON Results for This Line",
        data=export_line_json(selected_line),
        file_name=f"{selected_line}_results.json",
        mime="application/json",
        use_container_width=True,
    )
    st.stop()

a_id, b_id = run["tests"][idx]
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
