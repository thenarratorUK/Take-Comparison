import re
import json
import hashlib
import random
from pathlib import Path
from itertools import combinations

import streamlit as st


AUDIO_MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}

FILENAME_RE = re.compile(r"^(?P<line>.+)\.(?P<take>\d+)$")


def parse_uploaded_filename(name: str):
    """Expect: <lineKey>.<takeNumber>.<extension>

    <lineKey> may contain dots. We interpret the last dot-separated stem component
    as the integer take number.
    """
    p = Path(name)
    stem = p.stem  # removes final extension
    m = FILENAME_RE.match(stem)
    if not m:
        return None
    line_key = m.group("line").strip()
    take_num = int(m.group("take"))
    ext = p.suffix.lower()
    return line_key, take_num, ext


def stable_seed_for_line(line_key: str, take_ids_sorted: list[str]) -> int:
    """Deterministic seed derived from line key + sorted take IDs."""
    payload = line_key + "|" + "|".join(take_ids_sorted)
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit seed


def init_state():
    if "library" not in st.session_state:
        st.session_state.library = {}  # take_id -> metadata + bytes
    if "line_runs" not in st.session_state:
        st.session_state.line_runs = {}  # line_key -> run state


def build_library_from_uploads(uploaded_files):
    """Populate st.session_state.library from UploadedFile objects."""
    lib = {}
    errors = []
    for uf in uploaded_files:
        parsed = parse_uploaded_filename(uf.name)
        if not parsed:
            errors.append(f"Unrecognised filename format: {uf.name}")
            continue

        line_key, take_num, ext = parsed
        take_id = f"{line_key}.{take_num}"

        audio_bytes = uf.getvalue()
        mime = AUDIO_MIME_BY_EXT.get(ext, "audio/wav")

        lib[take_id] = {
            "take_id": take_id,
            "line": line_key,
            "take_number": take_num,
            "filename": uf.name,
            "bytes": audio_bytes,
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
    """Create the test list once per line and keep it stable."""
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


def ranking_table(line_key: str):
    lib = st.session_state.library
    run = st.session_state.line_runs[line_key]
    rows = []
    for tid, pts in run["scores"].items():
        rows.append({"Take": tid, "Points": pts, "Filename": lib[tid]["filename"]})
    rows.sort(key=lambda r: (-r["Points"], r["Take"]))
    return rows


def head_to_head_results(line_key: str):
    run = st.session_state.line_runs[line_key]
    out = []
    for i, r in enumerate(run["results"]):
        if r is None:
            continue
        out.append({"Test": i + 1, "A": r["a"], "B": r["b"], "Winner": r["winner"]})
    return out


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


st.set_page_config(page_title="Take Picker", layout="wide")
init_state()

st.title("Take Picker (A/B Comparisons)")

uploaded_files = st.file_uploader(
    "Upload audio takes (format: LineKey.takeNumber.ext, e.g. Line1.6.wav)",
    accept_multiple_files=True,
    type=["wav", "mp3", "m4a", "aac", "ogg", "flac"],
)

if uploaded_files:
    new_lib, errors = build_library_from_uploads(uploaded_files)
    if errors:
        st.error("Some files were ignored:\n\n" + "\n".join(f"- {e}" for e in errors))
    st.session_state.library.update(new_lib)

if not st.session_state.library:
    st.info("Upload some audio files to begin.")
    st.stop()

by_line = takes_by_line(st.session_state.library)
available_lines = sorted(by_line.keys())

with st.sidebar:
    st.header("Line Selection")
    selected_line = st.selectbox("Choose a line", available_lines, index=0)

take_ids_for_line = by_line[selected_line]
if len(take_ids_for_line) < 2:
    st.warning("This line has fewer than 2 takes, so there are no comparisons to run.")
    st.stop()

ensure_line_run(selected_line, take_ids_for_line)
run = st.session_state.line_runs[selected_line]
total_tests = len(run["tests"])
idx = run["idx"]

st.write(f"Selected line: {selected_line}")
st.write(f"Progress: Test {min(idx + 1, total_tests)} of {total_tests}")

if idx >= total_tests:
    st.subheader(f"{selected_line} Ranking (Points)")
    st.table(ranking_table(selected_line))

    st.subheader("Head-to-Head Outcomes (Completed Tests)")
    h2h = head_to_head_results(selected_line)
    st.table(h2h if h2h else [{"Info": "No recorded outcomes."}])

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
    st.write(lib[a_id]["filename"])
    st.audio(lib[a_id]["bytes"], format=lib[a_id]["mime"])

with col_b:
    st.subheader("B")
    st.write(lib[b_id]["filename"])
    st.audio(lib[b_id]["bytes"], format=lib[b_id]["mime"])

st.write("Which do you prefer?")

btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

with btn_col1:
    if st.button("Prefer A", use_container_width=True):
        record_vote(selected_line, a_id)

with btn_col2:
    if st.button("Prefer B", use_container_width=True):
        record_vote(selected_line, b_id)

with btn_col3:
    if st.button("Back", disabled=(len(run["history"]) == 0), use_container_width=True):
        undo_last_vote(selected_line)

st.divider()
st.subheader("Current Points (This Line)")
st.table(ranking_table(selected_line))
