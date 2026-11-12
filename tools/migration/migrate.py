"""One-off migration of the old OneDrive project folder into a local archive.

Stages, run in this order (each can be re-run safely):

    scan     Record every file's size and attributes WITHOUT reading file contents.
             The cloud-only flag has to be captured before anything hydrates the file.
    hash     SHA-256 every file in OLD (this downloads cloud-only placeholders) and
             write ARCHIVE/manifest.csv.
    copy     robocopy OLD -> ARCHIVE, skipping venvs, caches, Rust target dirs and keys.
    verify   Re-hash every archived file and compare it with the manifest.

Nothing here deletes or modifies anything in OLD.

Usage:  py -3.12 tools/migration/migrate.py <stage> --old <old project folder> --archive <archive folder>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# Set from the command line by `configure`.
OLD = ARCHIVE = AUX = SCAN_CSV = HASH_JOURNAL = MANIFEST = ROBOCOPY_LOG = VERIFY_REPORT = ""


def configure(old: str, archive: str) -> None:
    """Point the stages at the old project folder and the archive folder."""
    global OLD, ARCHIVE, AUX, SCAN_CSV, HASH_JOURNAL, MANIFEST, ROBOCOPY_LOG, VERIFY_REPORT
    OLD, ARCHIVE = os.path.abspath(old), os.path.abspath(archive)
    AUX = os.path.join(ARCHIVE, "_migration")
    SCAN_CSV = os.path.join(AUX, "scan.csv")
    HASH_JOURNAL = os.path.join(AUX, "hash_journal.tsv")
    MANIFEST = os.path.join(ARCHIVE, "manifest.csv")
    ROBOCOPY_LOG = os.path.join(AUX, "robocopy.log")
    VERIFY_REPORT = os.path.join(AUX, "verify_report.txt")

# File History style suffix, e.g. "dqn_model (2025_09_13 23_08_50 UTC).py".
# Dotfiles lost their stem entirely: "(2025_04_27 13_54_43 UTC).gitignore".
SNAPSHOT_RE = re.compile(
    r"^(?P<stem>.*?) ?\((?P<ts>\d{4}_\d{2}_\d{2} \d{2}_\d{2}_\d{2}) UTC\)(?P<ext>\.[^.]*)?$"
)

FILE_ATTRIBUTE_OFFLINE = 0x1000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x40000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000
CLOUD_ONLY_MASK = (
    FILE_ATTRIBUTE_OFFLINE | FILE_ATTRIBUTE_RECALL_ON_OPEN | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
)

ALWAYS_EXCLUDED_DIRS = {
    ".venv": "venv",
    "venv_scrabble_jupyter_notebook": "venv",
    "__pycache__": "pycache",
    ".ipynb_checkpoints": "ipynb_checkpoints",
}
EXCLUDED_FILE_EXTS = {".pem": "private_key"}


def long_path(path: str) -> str:
    """Opt out of MAX_PATH; the venv trees plus snapshot suffixes get long."""
    path = os.path.abspath(path)
    return path if path.startswith("\\\\?\\") else "\\\\?\\" + path


def split_snapshot(name: str) -> tuple[str, str]:
    """Return (base name with the snapshot suffix stripped, snapshot timestamp or '')."""
    m = SNAPSHOT_RE.match(name)
    if not m:
        return name, ""
    return m.group("stem") + (m.group("ext") or ""), m.group("ts")


@dataclass
class Entry:
    rel: str
    size: int
    attrs: int
    mtime: float
    excluded: str  # '' if the file goes to the archive, else the reason

    @property
    def cloud_only(self) -> bool:
        return bool(self.attrs & CLOUD_ONLY_MASK)


def walk(root: str):
    """Yield (relative dir, [dir entries], [file entries]) using only directory metadata."""
    stack = [""]
    while stack:
        rel_dir = stack.pop()
        dirs, files = [], []
        with os.scandir(long_path(os.path.join(root, rel_dir))) as it:
            for entry in it:
                (dirs if entry.is_dir(follow_symlinks=False) else files).append(entry)
        yield rel_dir, dirs, files
        stack.extend(os.path.join(rel_dir, d.name) for d in dirs)


def scan() -> list[Entry]:
    entries: list[Entry] = []
    excluded_roots: dict[str, str] = {}  # relative dir -> reason
    listing = list(walk(OLD))

    for rel_dir, dirs, files in listing:
        base_names = {split_snapshot(f.name)[0] for f in files}
        name = os.path.basename(rel_dir)
        if name in ALWAYS_EXCLUDED_DIRS:
            excluded_roots[rel_dir] = ALWAYS_EXCLUDED_DIRS[name]
        elif "pyvenv.cfg" in base_names:
            excluded_roots[rel_dir] = "venv"
        if "Cargo.toml" in base_names:
            for d in dirs:
                if d.name == "target":
                    excluded_roots[os.path.join(rel_dir, d.name)] = "rust_target"

    def exclusion_for(rel: str) -> str:
        parts = rel.split(os.sep)
        for i in range(1, len(parts)):
            reason = excluded_roots.get(os.sep.join(parts[:i]))
            if reason:
                return reason
        return EXCLUDED_FILE_EXTS.get(os.path.splitext(rel)[1].lower(), "")

    for rel_dir, _dirs, files in listing:
        for f in files:
            st = f.stat(follow_symlinks=False)  # cached find data: does not open the file
            rel = os.path.join(rel_dir, f.name)
            entries.append(Entry(rel, st.st_size, st.st_file_attributes, st.st_mtime, exclusion_for(rel)))

    entries.sort(key=lambda e: e.rel.lower())
    os.makedirs(AUX, exist_ok=True)
    with open(SCAN_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["relative_path", "size_bytes", "attributes_hex", "cloud_only", "mtime_epoch", "excluded"])
        for e in entries:
            w.writerow([e.rel, e.size, f"0x{e.attrs:08X}", int(e.cloud_only), f"{e.mtime:.3f}", e.excluded])

    total = sum(e.size for e in entries)
    cloud = [e for e in entries if e.cloud_only]
    print(f"files: {len(entries)}  bytes: {total:,}")
    print(f"cloud-only: {len(cloud)} files, {sum(e.size for e in cloud):,} bytes")
    by_reason: dict[str, list[Entry]] = {}
    for e in entries:
        by_reason.setdefault(e.excluded or "(archived)", []).append(e)
    for reason, group in sorted(by_reason.items()):
        print(f"  {reason:20s} {len(group):6d} files  {sum(e.size for e in group):>14,} bytes")
    print("excluded directory roots:")
    top_level = sorted(
        r for r in excluded_roots
        if not any(r.startswith(other + os.sep) for other in excluded_roots)
    )
    for r in top_level:
        print(f"  [{excluded_roots[r]}] {r}")
    print("excluded files outside those roots:")
    for e in entries:
        if e.excluded == "private_key":
            print(f"  [{e.excluded}] {e.rel}")
    return entries


def load_scan() -> list[Entry]:
    with open(SCAN_CSV, newline="", encoding="utf-8") as fh:
        return [
            Entry(r["relative_path"], int(r["size_bytes"]), int(r["attributes_hex"], 16),
                  float(r["mtime_epoch"]), r["excluded"])
            for r in csv.DictReader(fh)
        ]


def sha256_file(path: str, attempts: int = 8) -> tuple[str, int]:
    """Hash a file, retrying because OneDrive hydration can fail transiently."""
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            digest, size = hashlib.sha256(), 0
            with open(long_path(path), "rb") as fh:
                while chunk := fh.read(1 << 20):
                    digest.update(chunk)
                    size += len(chunk)
            return digest.hexdigest(), size
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise AssertionError("unreachable")


def hash_all() -> None:
    entries = load_scan()
    done: dict[str, tuple[str, int]] = {}
    if os.path.exists(HASH_JOURNAL):
        with open(HASH_JOURNAL, encoding="utf-8") as fh:
            for line in fh:
                rel, sha, size = line.rstrip("\n").split("\t")
                done[rel] = (sha, int(size))
    todo = [e for e in entries if e.rel not in done]
    print(f"hashing {len(todo)} files ({len(done)} already journalled)")

    failures: list[tuple[str, str]] = []
    started = time.time()
    with open(HASH_JOURNAL, "a", encoding="utf-8") as journal, ThreadPoolExecutor(8) as pool:
        def work(e: Entry):
            try:
                return e, sha256_file(os.path.join(OLD, e.rel)), None
            except OSError as exc:
                return e, None, exc
        for i, (e, result, exc) in enumerate(pool.map(work, todo), 1):
            if exc is not None:
                failures.append((e.rel, repr(exc)))
            else:
                done[e.rel] = result
                journal.write(f"{e.rel}\t{result[0]}\t{result[1]}\n")
                journal.flush()
            if i % 500 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {time.time() - started:.0f}s", flush=True)

    size_mismatches = [e.rel for e in entries if e.rel in done and done[e.rel][1] != e.size]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["relative_path", "base_name", "snapshot_utc", "size_bytes", "sha256",
                    "cloud_only", "archived", "exclusion_reason"])
        for e in entries:
            base, ts = split_snapshot(os.path.basename(e.rel))
            sha = done.get(e.rel, ("HASH_FAILED", 0))[0]
            w.writerow([e.rel, base, ts, e.size, sha, int(e.cloud_only),
                        int(not e.excluded), e.excluded])
    print(f"manifest rows: {len(entries)} -> {MANIFEST}")
    print(f"hash failures: {len(failures)}; size mismatches vs scan: {len(size_mismatches)}")
    for rel, err in failures:
        print(f"  FAILED {rel}: {err}")
    for rel in size_mismatches:
        print(f"  SIZE MISMATCH {rel}")
    if failures or size_mismatches:
        sys.exit(1)


def copy() -> None:
    entries = load_scan()
    roots: dict[str, str] = {}
    for e in entries:
        if e.excluded and e.excluded != "private_key":
            parts = e.rel.split(os.sep)
            for i in range(1, len(parts)):
                prefix = os.sep.join(parts[:i])
                name = parts[i - 1]
                if name in ALWAYS_EXCLUDED_DIRS or name == "target" or prefix in roots:
                    roots.setdefault(prefix, e.excluded)
                    break
            else:
                raise SystemExit(f"cannot derive an excluded root for {e.rel}")
    # A venv found only through pyvenv.cfg has an arbitrary name: add it explicitly.
    for e in entries:
        if e.excluded == "venv" and split_snapshot(os.path.basename(e.rel))[0] == "pyvenv.cfg":
            roots.setdefault(os.path.dirname(e.rel), "venv")
    top = sorted(r for r in roots if not any(r.startswith(o + os.sep) for o in roots))

    cmd = ["robocopy", OLD, ARCHIVE, "/E", "/COPY:DAT", "/DCOPY:DAT", "/R:5", "/W:5", "/MT:8",
           "/NP", "/NFL", "/NDL", f"/LOG:{ROBOCOPY_LOG}", "/XF", "*.pem",
           "/XD", *[os.path.join(OLD, r) for r in top]]
    print("excluding", len(top), "directories and *.pem")
    rc = subprocess.run(cmd).returncode
    print(f"robocopy exit code {rc} (0-7 = success, 8+ = at least one failure); log: {ROBOCOPY_LOG}")
    if rc >= 8:
        sys.exit(1)


def verify() -> None:
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    expected = {r["relative_path"]: r for r in rows if r["archived"] == "1"}
    problems: list[str] = []

    def check(rel: str):
        path = os.path.join(ARCHIVE, rel)
        if not os.path.exists(long_path(path)):
            return rel, "MISSING"
        sha, _ = sha256_file(path, attempts=2)
        return rel, "" if sha == expected[rel]["sha256"] else "HASH MISMATCH"

    with ThreadPoolExecutor(8) as pool:
        for rel, problem in pool.map(check, sorted(expected)):
            if problem:
                problems.append(f"{problem}: {rel}")

    present = set()
    for rel_dir, _dirs, files in walk(ARCHIVE):
        if rel_dir == "_migration" or rel_dir.startswith("_migration" + os.sep):
            continue
        for f in files:
            rel = os.path.join(rel_dir, f.name)
            if rel != "manifest.csv":
                present.add(rel)
    extras = sorted(present - set(expected))
    problems += [f"UNEXPECTED FILE IN ARCHIVE: {rel}" for rel in extras]
    problems += [f"KEY FILE IN ARCHIVE: {rel}" for rel in present if rel.lower().endswith(".pem")]

    lines = [
        f"manifest rows (all files in OLD): {len(rows)}",
        f"expected in archive: {len(expected)} files, "
        f"{sum(int(r['size_bytes']) for r in expected.values()):,} bytes",
        f"present in archive: {len(present)} files",
        f"excluded from archive: {len(rows) - len(expected)} files",
        f"problems: {len(problems)}",
        *problems,
        "RESULT: " + ("FAILED" if problems else "OK - every archived file matches its manifest SHA-256"),
    ]
    with open(VERIFY_REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["scan", "hash", "copy", "verify"])
    parser.add_argument("--old", required=True, help="the old project folder (read only)")
    parser.add_argument("--archive", required=True, help="where the verified copy and manifest.csv go")
    arguments = parser.parse_args()
    configure(arguments.old, arguments.archive)
    {"scan": scan, "hash": hash_all, "copy": copy, "verify": verify}[arguments.stage]()
