set -e

SO_FILE=/tmp/packaged.so
PY_FILE=./apps/cutlass/cutlass.py

python3 ${PY_FILE}
ldd ${SO_FILE}
