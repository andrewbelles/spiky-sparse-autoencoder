#!/usr/bin/bash
#
# small.sh  Andrew Belles  April 10th, 2026 
# 
# Downloads the official FMA small archive and normalizes it into
# preprocess/data/fma_small/ for preprocessing.
# 

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="${script_dir}/data"
extract_dir="${data_dir}/fma_small"
legacy_archive_path="${data_dir}/fma_small.zip"
dataset_url="https://os.unil.cloud.switch.ch/fma/fma_small.zip"
expected_sha1="ade154f733639d52e35e32f5593efe5be76c6d70"
python_bin="$(command -v python3 || true)"
tmp_archive=""
tmp_root=""

mkdir -p "${data_dir}"

cleanup() {
  if [ -n "${tmp_root}" ] && [ -d "${tmp_root}" ]; then
    rm -rf "${tmp_root}"
  fi

  if [ -n "${tmp_archive}" ] && [ -f "${tmp_archive}" ]; then
    rm -f "${tmp_archive}"
  fi
}

trap cleanup EXIT

if [ -z "${python_bin}" ]; then
  python_bin="$(command -v python || true)"
fi

if [ -z "${python_bin}" ]; then
  printf 'python3 or python is required to extract %s\n' "${dataset_url}" >&2
  exit 1
fi

tmp_archive="$(mktemp "${data_dir}/.fma_small.zip.XXXXXX")"
tmp_root="$(mktemp -d "${data_dir}/.fma_small.extract.XXXXXX")"

curl -fL --retry 3 --output "${tmp_archive}" "${dataset_url}"

(
  cd "${data_dir}"
  printf '%s  %s\n' "${expected_sha1}" "$(basename "${tmp_archive}")" | sha1sum -c -
)

"${python_bin}" - "${tmp_archive}" "${tmp_root}" <<'PY'
import pathlib
import shutil
import sys
import zipfile

archive = pathlib.Path(sys.argv[1])
tmp_root = pathlib.Path(sys.argv[2])
target = tmp_root / "fma_small"
target.mkdir(parents=True, exist_ok=True)

file_count = 0

with zipfile.ZipFile(archive) as zf:
    for info in zf.infolist():
        raw_path = pathlib.PurePosixPath(info.filename)
        parts = [part for part in raw_path.parts if part not in ("", ".")]

        if any(part == ".." for part in parts):
            raise SystemExit(f"unsafe archive entry: {info.filename}")

        if not parts or info.is_dir():
            continue

        if parts[0] == "fma_small":
            parts = parts[1:]

        if not parts:
            continue

        destination = target.joinpath(*parts)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with zf.open(info, "r") as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

        file_count += 1

if file_count == 0:
    raise SystemExit("archive extracted zero files")

required_path = target / "000" / "000002.mp3"
if not required_path.is_file():
    raise SystemExit(f"expected file missing after extraction: {required_path}")
PY

rm -rf "${extract_dir}"
mv "${tmp_root}/fma_small" "${extract_dir}"
rm -f "${legacy_archive_path}"

printf 'Dataset ready at %s\n' "${extract_dir}"
