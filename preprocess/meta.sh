#!/usr/bin/bash
#
# meta.sh  Andrew Belles  April 10th, 2026
#
# Downloads the official FMA metadata archive and writes tracks.csv to
# preprocess/data/ for use across all fma_{size} subsets.
#

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="${script_dir}/data"
tracks_path="${data_dir}/tracks.csv"
metadata_url="https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
expected_sha1="f0df49ffe5f2a6008d7dc83c6915b31835dfe733"
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
  printf 'python3 or python is required to extract %s\n' "${metadata_url}" >&2
  exit 1
fi

tmp_archive="$(mktemp "${data_dir}/.fma_metadata.zip.XXXXXX")"
tmp_root="$(mktemp -d "${data_dir}/.fma_metadata.extract.XXXXXX")"

curl -fL --retry 3 --output "${tmp_archive}" "${metadata_url}"

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
target = tmp_root / "tracks.csv"

with zipfile.ZipFile(archive) as zf:
    for info in zf.infolist():
        raw_path = pathlib.PurePosixPath(info.filename)
        parts = [part for part in raw_path.parts if part not in ("", ".")]

        if any(part == ".." for part in parts):
            raise SystemExit(f"unsafe archive entry: {info.filename}")

        if info.is_dir():
            continue

        if parts[-1] != "tracks.csv":
            continue

        with zf.open(info, "r") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        break
    else:
        raise SystemExit("tracks.csv not found in metadata archive")
PY

mv "${tmp_root}/tracks.csv" "${tracks_path}"

printf 'Metadata ready at %s\n' "${tracks_path}"
