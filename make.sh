#!/bin/bash
# make.sh — minimal wrapper: no subshells, no external commands. The only extra
# cost over calling the exe directly is this one bash (which exec then replaces).
d=${0%/*}; [ "$d" = "$0" ] && d=.
[ -d /c/Temp ] || mkdir -p /c/Temp      # builtin test; mkdir only on first run
export TMPDIR='C:\Temp' TMP='C:\Temp' TEMP='C:\Temp'
exec "$d/vendor/gcc/bin/mingw32-make.exe" "$@"