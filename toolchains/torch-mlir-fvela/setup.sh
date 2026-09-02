#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source "${script_dir}/versions.lock"

python_bin="${PYTHON_BIN:-python${PYTHON_VERSION}}"
jobs="${JOBS:-2}"
deps_dir="${script_dir}/.deps"
source_dir="${deps_dir}/torch-mlir"
venv_dir="${deps_dir}/venv"
build_dir="${script_dir}/build"
torch_mlir_patch="${script_dir}/patches/0001-preserve-tvela-ops-in-linalg.patch"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "error: ${python_bin} is required" >&2
  exit 1
fi

mkdir -p "${deps_dir}"
if [[ ! -d "${source_dir}/.git" ]]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/llvm/torch-mlir.git "${source_dir}"
fi

git -C "${source_dir}" fetch --depth=1 origin "${TORCH_MLIR_COMMIT}"
git -C "${source_dir}" checkout --detach FETCH_HEAD
git -C "${source_dir}" reset --hard FETCH_HEAD
git -C "${source_dir}" clean -f -- \
  lib/Dialect/TorchConversion/Transforms/ConvertTVelaOps.cpp
git -C "${source_dir}" submodule update --init --depth=1 externals/llvm-project

actual_llvm_commit="$(git -C "${source_dir}/externals/llvm-project" rev-parse HEAD)"
if [[ "${actual_llvm_commit}" != "${TORCH_MLIR_LLVM_COMMIT}" ]]; then
  echo "error: torch-mlir LLVM commit mismatch: ${actual_llvm_commit}" >&2
  exit 1
fi

if git -C "${source_dir}" apply --check \
    "${torch_mlir_patch}" >/dev/null 2>&1; then
  git -C "${source_dir}" apply "${torch_mlir_patch}"
else
  echo "error: T-Vela torch-mlir patch does not match ${TORCH_MLIR_COMMIT}" >&2
  exit 1
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  "${python_bin}" -m venv --without-pip "${venv_dir}"
fi

if [[ ! -x "${venv_dir}/bin/pip" ]]; then
  pip_bootstrap_python="${PIP_BOOTSTRAP_PYTHON:-${repo_root}/.conda-env/bin/python}"
  if [[ ! -x "${pip_bootstrap_python}" ]]; then
    echo "error: set PIP_BOOTSTRAP_PYTHON to a Python installation with pip" >&2
    exit 1
  fi
  "${pip_bootstrap_python}" -m pip --python "${venv_dir}/bin/python" \
    install --upgrade pip
fi

python_include_dir="$("${venv_dir}/bin/python" -c \
  'import sysconfig; print(sysconfig.get_path("include"))')"
python_library_dir="$("${venv_dir}/bin/python" -c \
  'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
python_library="${python_library_dir}/libpython${PYTHON_VERSION}.so.1.0"

if [[ ! -f "${python_include_dir}/Python.h" ]]; then
  python_dev_dir="${deps_dir}/python-dev"
  mkdir -p "${python_dev_dir}/packages" "${python_dev_dir}/root"
  shopt -s nullglob
  python_dev_packages=(
    "${python_dev_dir}/packages/libpython${PYTHON_VERSION}-dev_"*.deb
  )
  if [[ ${#python_dev_packages[@]} -eq 0 ]]; then
    if ! command -v apt-get >/dev/null 2>&1; then
      echo "error: Python ${PYTHON_VERSION} development headers are required" >&2
      exit 1
    fi
    (
      cd "${python_dev_dir}/packages"
      apt-get download "libpython${PYTHON_VERSION}-dev"
    )
    python_dev_packages=(
      "${python_dev_dir}/packages/libpython${PYTHON_VERSION}-dev_"*.deb
    )
  fi
  for package_file in "${python_dev_packages[@]}"; do
    dpkg-deb -x "${package_file}" "${python_dev_dir}/root"
  done
  python_include_dir="${python_dev_dir}/root/usr/include/python${PYTHON_VERSION}"
  python_multiarch="$("${venv_dir}/bin/python" -c \
    'import sysconfig; print(sysconfig.get_config_var("MULTIARCH"))')"
  if [[ -d "${python_dev_dir}/root/usr/include/${python_multiarch}" && \
        ! -e "${python_include_dir}/${python_multiarch}" ]]; then
    ln -s "../${python_multiarch}" "${python_include_dir}/${python_multiarch}"
  fi
fi

if [[ ! -f "${python_include_dir}/Python.h" || ! -f "${python_library}" ]]; then
  echo "error: incomplete Python ${PYTHON_VERSION} development files" >&2
  exit 1
fi

"${venv_dir}/bin/python" -m pip install --upgrade pip
"${venv_dir}/bin/python" -m pip install \
  -r "${source_dir}/build-requirements.txt" \
  -r "${source_dir}/externals/llvm-project/mlir/python/requirements.txt"
"${venv_dir}/bin/python" -m pip install \
  --find-links https://download.pytorch.org/whl/nightly/cpu/torch/ \
  --pre "torch==${PYTORCH_VERSION}"

"${venv_dir}/bin/cmake" -G Ninja \
  -S "${source_dir}/externals/llvm-project/llvm" \
  -B "${build_dir}" \
  -DCMAKE_MAKE_PROGRAM="${venv_dir}/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=torch-mlir \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="${source_dir}" \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=OFF \
  -DTORCH_MLIR_ENABLE_LTC=OFF \
  -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=OFF \
  -DTORCH_MLIR_ENABLE_REFBACKEND=OFF \
  -DTORCH_MLIR_ENABLE_STABLEHLO=OFF \
  -DTORCH_MLIR_ENABLE_TOSA=OFF \
  -DPython3_EXECUTABLE="${venv_dir}/bin/python" \
  -DPython3_INCLUDE_DIR="${python_include_dir}" \
  -DPython3_LIBRARY="${python_library}" \
  -DPython_EXECUTABLE="${venv_dir}/bin/python" \
  -DPython_INCLUDE_DIR="${python_include_dir}" \
  -DPython_LIBRARY="${python_library}"

"${venv_dir}/bin/cmake" --build "${build_dir}" \
  --target TorchMLIRPythonModules \
  -j "${jobs}"

touch "${build_dir}/.tvela-fvela-linalg-ready"

echo "Frontend Python: ${venv_dir}/bin/python"
echo "Torch MLIR Python path: ${build_dir}/tools/torch-mlir/python_packages/torch_mlir"
