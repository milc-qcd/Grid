#!/usr/bin/env bash
set -e

EIGEN_URL='https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2'
EIGEN_SHA256SUM='b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626'

echo "-- deploying Eigen source..."
ARC=$(basename ${EIGEN_URL})
wget ${EIGEN_URL} --no-check-certificate

# Verify checksum portably:
#   1. GNU sha256sum --check  (Linux, Homebrew coreutils with --check support)
#   2. macOS shasum -a 256 --check  (always present on macOS)
#   3. openssl dgst -sha256   (fallback)
EIGEN_ARC=$(basename "$EIGEN_URL")
CHECKLINE="${EIGEN_SHA256SUM}  ${EIGEN_ARC}"
if echo "$CHECKLINE" | sha256sum --check 2>/dev/null; then
  : # verified with sha256sum
elif command -v shasum &>/dev/null; then
  echo "$CHECKLINE" | shasum -a 256 --check || exit 1
elif command -v openssl &>/dev/null; then
  COMPUTED=$(openssl dgst -sha256 "$EIGEN_ARC" | awk '{print $NF}')
  if [ "$COMPUTED" != "$EIGEN_SHA256SUM" ]; then
    echo "ERROR: SHA256 mismatch for $EIGEN_ARC" >&2
    echo "  expected: $EIGEN_SHA256SUM" >&2
    echo "  got:      $COMPUTED" >&2
    exit 1
  fi
  echo "${EIGEN_ARC}: OK"
else
  echo "WARNING: could not verify checksum, please install sha256sum or shasum" >&2
fi

./scripts/update_eigen.sh "${ARC}"
rm "${ARC}"
echo '-- generating Make.inc files...'
./scripts/filelist
echo '-- generating configure script...'
autoreconf -fvi
