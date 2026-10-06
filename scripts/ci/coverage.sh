#!/usr/bin/env bash
# Measure the test coverage of geodex on Linux with the full planning stack (OMPL fork, VAMP
# and the built-in robots).
#
#   1. C++ line and function coverage of include/geodex and src from ctest.
#   2. Python line coverage of python/geodex from pytest.
#   3. C++ line coverage reached through the Python tests, with the extension built with the
#      same coverage flags.
#
# The builds use --coverage -Og by default. pip installs pinned versions of gcovr and
# coverage.py into $COV_TOOLS, outside the pixi environment.
#
# Usage
#   scripts/ci/coverage.sh
#
# Environment
#   COV_JOBS        parallel build jobs (default all logical cores)
#   COV_TEST_JOBS   parallel ctest jobs (default all logical cores)
#   COV_TIMEOUT     ctest timeout per test in seconds (default 1800)
#   COV_BUILD_TYPE  CMake build type of the coverage builds (default Debug)
#   COV_CXX_FLAGS   CMAKE_CXX_FLAGS of the coverage builds (default --coverage -Og)
#   COV_OUT         report directory (default build/coverage)
#   COV_TOOLS       directory for gcovr and coverage.py (default build/coverage-tools)
#   COV_PYTHON_CPP  0 builds the extension without coverage flags and skips step 3 (default 1)
#   COV_SKIP_CTEST  1 reuses the ctest counters in build/cov and the ctest logs in $COV_OUT
#   COV_CLEAN       1 removes build/cov, build/cov-py and build/cov-py-site first
#   COV_FAIL_UNDER  minimum C++ line coverage from ctest in percent (default 0)
#
# Reports in $COV_OUT
#   summary.md      per-directory tables, totals, least-covered files and failing tests
#   cpp/            ctest coverage with html/index.html, cobertura.xml and coverage.json
#   python/         coverage.py reports with cobertura.xml and coverage.json
#   python_cpp/     C++ coverage reached through the Python tests
#   union/          union of the two C++ reports
#
# The script exits with 1 when a test fails or the coverage is below COV_FAIL_UNDER. It
# writes the reports first.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cores="$(getconf _NPROCESSORS_ONLN)"
jobs="${COV_JOBS:-${cores}}"
test_jobs="${COV_TEST_JOBS:-${cores}}"
timeout="${COV_TIMEOUT:-1800}"
build_type="${COV_BUILD_TYPE:-Debug}"
cxx_flags="${COV_CXX_FLAGS:---coverage -Og}"
out="${COV_OUT:-${repo}/build/coverage}"
tools="${COV_TOOLS:-${repo}/build/coverage-tools}"
python_cpp="${COV_PYTHON_CPP:-1}"
skip_ctest="${COV_SKIP_CTEST:-0}"
fail_under="${COV_FAIL_UNDER:-0}"

gcovr_version="8.4"
coverage_version="7.10.7"

cpp_build="${repo}/build/cov"
py_build="${repo}/build/cov-py"
site="${repo}/build/cov-py-site"

cd "${repo}"
if [[ "${COV_CLEAN:-0}" == "1" && "${skip_ctest}" != "1" ]]; then
  rm -rf "${cpp_build}" "${py_build}" "${site}"
fi
mkdir -p "${out}" "${tools}"
out="$(cd "${out}" && pwd)"
tools="$(cd "${tools}" && pwd)"

t0=$(date +%s)
t_last=${t0}
stamp() { echo "[coverage $(date +%H:%M:%S) +$(( $(date +%s) - t0 ))s] $*"; }
# Record the time since the previous call under the given step name.
mark() {
  local now
  now=$(date +%s)
  printf '%s\t%s\n' "$1" "$(( now - t_last ))" >> "${out}/timings.tsv"
  t_last=${now}
}
px() { pixi run --locked "$@"; }

rm -rf "${out}/python" "${out}/python_cpp" "${out}/union" "${out}/cpp/html" \
  "${out}/summary.md" "${out}/timings.tsv"
if [[ "${skip_ctest}" != "1" ]]; then rm -rf "${out}/cpp"; fi
mkdir -p "${out}/cpp/html" "${out}/python" "${out}/python_cpp" "${out}/union"
printf 'build_type\t%s\ncxx_flags\t%s\n' "${build_type}" "${cxx_flags}" > "${out}/settings.tsv"

# ---------------------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------------------
stamp "prerequisites (OMPL fork, VAMP source, gcovr ${gcovr_version}, coverage.py ${coverage_version})"
px check-ompl
px fetch-vamp
if [[ ! -d "${tools}/gcovr-${gcovr_version}/gcovr" ]]; then
  px python -m pip install --quiet --disable-pip-version-check --no-warn-script-location \
    --target "${tools}/gcovr-${gcovr_version}" "gcovr==${gcovr_version}"
fi
if [[ ! -d "${tools}/coverage-${coverage_version}/coverage" ]]; then
  px python -m pip install --quiet --disable-pip-version-check --no-warn-script-location \
    --target "${tools}/coverage-${coverage_version}" "coverage==${coverage_version}"
fi
gcovr() { PYTHONPATH="${tools}/gcovr-${gcovr_version}" px python -m gcovr "$@"; }
pycov() {
  PYTHONPATH="${tools}/coverage-${coverage_version}" px python -m coverage "$@" \
    --rcfile="${out}/python/coveragerc"
}
mark prerequisites

# Count include/geodex and src without the generated robot sources. --merge-lines counts a
# header line once and marks it covered when any template instance or translation unit ran it.
gcovr_args=(--root "${repo}" --filter 'include/geodex/' --filter 'src/'
  --exclude '.*/generated/.*' --gcov-executable gcov -j "${jobs}" --merge-lines)

# ---------------------------------------------------------------------------------------
# 1. C++ coverage from ctest
# ---------------------------------------------------------------------------------------
ctest_rc=0
if [[ "${skip_ctest}" != "1" ]]; then
  stamp "configure and build ${cpp_build} (${build_type}, ${cxx_flags}, -j${jobs})"
  px cmake -S "${repo}" -B "${cpp_build}" -G Ninja -DCMAKE_BUILD_TYPE="${build_type}" \
    "-DCMAKE_CXX_FLAGS=${cxx_flags}" -DVAMP_DIR="${repo}/.pixi/vamp" \
    -DGEODEX_OMPL=ON -DGEODEX_VAMP=ON -DGEODEX_ROBOTS=ON -DBUILD_TESTING=ON \
    -DBUILD_EXAMPLES=OFF -DBUILD_OMPL_EXAMPLES=OFF
  px cmake --build "${cpp_build}" -j "${jobs}"
  mark "C++ build"

  stamp "ctest (-j${test_jobs}, timeout ${timeout} s)"
  find "${cpp_build}" -name '*.gcda' -delete
  px ctest --test-dir "${cpp_build}" -j "${test_jobs}" --timeout "${timeout}" \
    --output-on-failure --output-junit "${out}/cpp/ctest-junit.xml" \
    > "${out}/cpp/ctest.log" 2>&1 || ctest_rc=$?
  tail -n 25 "${out}/cpp/ctest.log"
  stamp "ctest exit code ${ctest_rc}"
  mark ctest
else
  stamp "reusing the ctest counters in ${cpp_build}"
  if grep -q "tests failed" "${out}/cpp/ctest.log"; then ctest_rc=8; fi
fi

stamp "gcovr for ctest"
gcovr "${gcovr_args[@]}" "${cpp_build}" \
  --txt "${out}/cpp/summary.txt" \
  --json "${out}/cpp/coverage.json" \
  --json-summary-pretty --json-summary "${out}/cpp/summary.json" \
  --cobertura "${out}/cpp/cobertura.xml" \
  --html-details "${out}/cpp/html/index.html" --html-title "geodex C++ coverage (ctest)"
mark "gcovr (ctest)"

# ---------------------------------------------------------------------------------------
# 2 and 3. Python line coverage and the C++ coverage reached through the Python tests
# ---------------------------------------------------------------------------------------
py_flags=(-C "build-dir=${py_build}"
  -C cmake.define.GEODEX_BUNDLE_DEPS=OFF
  -C "cmake.define.VAMP_DIR=${repo}/.pixi/vamp")
if [[ "${python_cpp}" == "1" ]]; then
  py_flags+=(-C "cmake.build-type=${build_type}" -C "cmake.define.CMAKE_CXX_FLAGS=${cxx_flags}")
fi
stamp "build the Python module into ${site}"
rm -rf "${site}"
CMAKE_BUILD_PARALLEL_LEVEL="${jobs}" px python -m pip install --quiet \
  --disable-pip-version-check --no-build-isolation --no-deps --target "${site}" "${repo}" \
  "${py_flags[@]}"
mark "Python module build"

# Map the installed package back to python/geodex in the reports.
cat > "${out}/python/coveragerc" <<EOF
[run]
source_pkgs = geodex
parallel = true
data_file = ${out}/python/.coverage

[paths]
source =
    python/geodex
    ${site}/geodex

[report]
include = python/geodex/*
EOF

stamp "pytest under coverage.py"
if [[ -d "${py_build}" ]]; then find "${py_build}" -name '*.gcda' -delete; fi
pytest_rc=0
PYTHONPATH="${site}:${tools}/coverage-${coverage_version}" px python -m coverage run \
  --rcfile="${out}/python/coveragerc" -m pytest python/tests -q -p no:cacheprovider \
  --durations=20 --junitxml="${out}/python/pytest-junit.xml" \
  > "${out}/python/pytest.log" 2>&1 || pytest_rc=$?
tail -n 30 "${out}/python/pytest.log"
stamp "pytest exit code ${pytest_rc}"
mark pytest

pycov combine
pycov report -m | tee "${out}/python/summary.txt"
pycov json -o "${out}/python/coverage.json"
pycov xml -o "${out}/python/cobertura.xml"

if [[ "${python_cpp}" == "1" ]]; then
  stamp "gcovr for the Python tests"
  gcovr "${gcovr_args[@]}" "${py_build}" \
    --txt "${out}/python_cpp/summary.txt" \
    --json "${out}/python_cpp/coverage.json" \
    --json-summary-pretty --json-summary "${out}/python_cpp/summary.json" \
    --cobertura "${out}/python_cpp/cobertura.xml"
  # The nanobind bindings under python/src have a report of their own.
  gcovr --root "${repo}" --filter 'python/src/' --gcov-executable gcov -j "${jobs}" \
    --merge-lines "${py_build}" \
    --txt "${out}/python_cpp/bindings_summary.txt" \
    --json-summary-pretty --json-summary "${out}/python_cpp/bindings_summary.json"

  stamp "gcovr union of ctest and pytest"
  gcovr --root "${repo}" --merge-lines \
    --add-tracefile "${out}/cpp/coverage.json" \
    --add-tracefile "${out}/python_cpp/coverage.json" \
    --txt "${out}/union/summary.txt" \
    --json "${out}/union/coverage.json" \
    --json-summary-pretty --json-summary "${out}/union/summary.json"
fi
mark "reports"

# ---------------------------------------------------------------------------------------
# summary.md
# gcovr counts a function once per template instance. The summary also counts it once per
# source location and marks it covered when any instance ran.
# ---------------------------------------------------------------------------------------
stamp "summary"
px python - "${repo}" "${out}" <<'PY'
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

repo, out = sys.argv[1], sys.argv[2]
DIRS = ["algorithm", "core", "manifold", "metrics", "heuristics", "planning",
        "integration/ompl", "integration/vamp", "integration/pinocchio", "robots",
        "collision", "utils"]


def directory(path):
    if path.startswith("src/"):
        return "src"
    rel = path[len("include/geodex/"):]
    for d in sorted(DIRS, key=len, reverse=True):
        if rel.startswith(d + "/"):
            return d
    return "(top level)"


def pct(c, t):
    return f"{100.0 * c / t:.1f}" if t else "n/a"


def load(*parts):
    path = os.path.join(out, *parts)
    return json.load(open(path)) if os.path.exists(path) else None


def source_functions(name):
    """Covered and total functions per file, one per source line over all instances."""
    raw = load(name, "coverage.json")
    result = {}
    for f in (raw or {}).get("files", []):
        seen = {}
        for fn in f["functions"]:
            seen[fn["lineno"]] = seen.get(fn["lineno"], False) or fn["execution_count"] > 0
        result[f["file"]] = (sum(seen.values()), len(seen))
    return result


def directory_table(summary, functions):
    rows = {}
    for f in summary["files"]:
        c, t = functions.get(f["filename"], (0, 0))
        r = rows.setdefault(directory(f["filename"]), [0, 0, 0, 0, 0])
        for i, v in enumerate((1, f["line_covered"], f["line_total"], c, t)):
            r[i] += v
    lines = ["| Directory | Files | Lines | Line % | Source functions | Function % |",
             "|---|---:|---:|---:|---:|---:|"]
    total = [0, 0, 0, 0, 0]
    for d in DIRS + ["src", "(top level)"]:
        if d in rows:
            n, lc, lt, fc, ft = rows[d]
            total = [a + b for a, b in zip(total, rows[d])]
            lines.append(f"| {d} | {n} | {lc} / {lt} | {pct(lc, lt)} | {fc} / {ft} | {pct(fc, ft)} |")
    n, lc, lt, fc, ft = total
    lines.append(f"| **total** | {n} | {lc} / {lt} | **{pct(lc, lt)}** | {fc} / {ft} | **{pct(fc, ft)}** |")
    return "\n".join(lines)


def least_covered(summary, functions, n=15):
    files = sorted(summary["files"], key=lambda f: (f["line_percent"], -f["line_total"]))
    lines = ["| File | Line % | Lines | Function % |", "|---|---:|---:|---:|"]
    for f in files[:n]:
        lines.append(f"| `{f['filename']}` | {f['line_percent']:.1f} | "
                     f"{f['line_covered']} / {f['line_total']} | "
                     f"{pct(*functions.get(f['filename'], (0, 0)))} |")
    return "\n".join(lines)


def absent(summary):
    seen = {f["filename"] for f in summary["files"]}
    found = set()
    for pattern in ("include/geodex/**/*.hpp", "src/**/*.hpp", "src/**/*.cpp"):
        found |= set(glob.glob(pattern, root_dir=repo, recursive=True))
    return sorted(p for p in found if "/generated/" not in p and p not in seen)


def failures(junit):
    if not os.path.exists(junit):
        return ["(no JUnit report)"]
    bad = set()
    for tc in ET.parse(junit).getroot().iter("testcase"):
        cls, name = tc.get("classname") or "", tc.get("name")
        label = name if cls in ("", name) else f"{cls}::{name}"
        e = tc.find("failure")
        if e is None:
            e = tc.find("error")
        if e is not None:
            msg = (e.get("message") or e.tag).strip().splitlines()
            bad.add(f"{label} ({msg[0][:100] if msg else e.tag})")
        elif tc.get("status") in ("fail", "failed", "timeout"):
            bad.add(f"{label} ({tc.get('status')})")
    return sorted(bad)


def ranges(numbers):
    """Compress sorted line numbers into ranges such as 17-44."""
    parts, start, prev = [], None, None
    for n in numbers:
        if start is not None and n == prev + 1:
            prev = n
            continue
        if start is not None:
            parts.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = n
    if start is not None:
        parts.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ", ".join(parts)


def details(title, body):
    return f"<details><summary>{title}</summary>\n\n{body}\n\n</details>\n"


settings = dict(line.rstrip("\n").split("\t", 1) for line in open(os.path.join(out, "settings.tsv")))
cpp, pyc, uni = load("cpp", "summary.json"), load("python_cpp", "summary.json"), load("union", "summary.json")
cpp_fn = source_functions("cpp")
py = load("python", "coverage.json")
bind = load("python_cpp", "bindings_summary.json")

md = ["## geodex coverage", "",
      f"Build type {settings['build_type']} with `{settings['cxx_flags']}`. C++ counts "
      "`include/geodex/**` and `src/**` without generated files. Python counts `python/geodex/**`.",
      "", "| Measure | Covered / total | Percent |", "|---|---:|---:|"]
if cpp:
    fc = sum(c for c, _ in cpp_fn.values()); ft = sum(t for _, t in cpp_fn.values())
    md.append(f"| C++ lines, ctest | {cpp['line_covered']} / {cpp['line_total']} | **{pct(cpp['line_covered'], cpp['line_total'])}** |")
    md.append(f"| C++ source functions, ctest | {fc} / {ft} | {pct(fc, ft)} |")
if py:
    t = py["totals"]
    md.append(f"| Python lines, pytest | {t['covered_lines']} / {t['num_statements']} | **{t['percent_covered']:.1f}** |")
if pyc:
    md.append(f"| C++ lines, Python tests | {pyc['line_covered']} / {pyc['line_total']} | {pct(pyc['line_covered'], pyc['line_total'])} |")
if uni:
    md.append(f"| C++ lines, union of ctest and Python tests | {uni['line_covered']} / {uni['line_total']} | {pct(uni['line_covered'], uni['line_total'])} |")
if bind:
    md.append(f"| Bindings `python/src/**`, Python tests | {bind['line_covered']} / {bind['line_total']} | {pct(bind['line_covered'], bind['line_total'])} |")
md.append("")

md.append("### Failing or timed-out tests\n")
for label, junit in (("ctest", "cpp/ctest-junit.xml"), ("pytest", "python/pytest-junit.xml")):
    bad = failures(os.path.join(out, junit))
    md.append(f"- {label}, " + ("none" if not bad else f"{len(bad)} tests"))
    md += [f"  - {b}" for b in bad]
md.append("")

if cpp:
    md += ["### C++ coverage from ctest by directory", "", directory_table(cpp, cpp_fn), ""]
    md += ["### 15 least-covered C++ files (ctest)", "", least_covered(cpp, cpp_fn), ""]
if py:
    rows = ["| File | Statements | Missed | Line % | Missing lines |", "|---|---:|---:|---:|---|"]
    for name, f in sorted(py["files"].items()):
        s = f["summary"]
        rows.append(f"| `{name}` | {s['num_statements']} | {s['missing_lines']} | "
                    f"{s['percent_covered']:.1f} | {ranges(f['missing_lines'])} |")
    md += ["### Python line coverage", "", "\n".join(rows), ""]

timings = os.path.join(out, "timings.tsv")
if os.path.exists(timings):
    rows = ["| Step | Seconds |", "|---|---:|"]
    rows += [f"| {s} | {v} |" for s, v in (l.rstrip("\n").split("\t") for l in open(timings))]
    md.append(details("Wall time per step", "\n".join(rows)))
if pyc:
    md.append(details("C++ coverage reached through the Python tests by directory",
                      directory_table(pyc, source_functions("python_cpp"))))
if uni:
    md.append(details("C++ coverage, union of ctest and the Python tests, by directory",
                      directory_table(uni, source_functions("union"))))
if cpp:
    missing = absent(cpp)
    md.append(details(f"Files absent from the ctest report ({len(missing)})",
                      "These files do not have executable lines, or no test compiles them.\n\n"
                      + "\n".join(f"- `{m}`" for m in missing)))

open(os.path.join(out, "summary.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
PY

stamp "done in $(( $(date +%s) - t0 )) s, reports in ${out}"
status=0
if [[ ${ctest_rc} -ne 0 || ${pytest_rc} -ne 0 ]]; then
  echo "coverage: tests failed (ctest exit ${ctest_rc}, pytest exit ${pytest_rc})" >&2
  status=1
fi
if ! px python -c "import json, sys; s = json.load(open('${out}/cpp/summary.json')); \
sys.exit(100.0 * s['line_covered'] / s['line_total'] < float('${fail_under}'))"; then
  echo "coverage: C++ line coverage is below COV_FAIL_UNDER=${fail_under}" >&2
  status=1
fi
exit ${status}
