#!/usr/bin/env python3
import argparse
import ctypes
import os
import sys


def _candidate_libs(explicit_lib: str | None = None):
    if explicit_lib:
        yield explicit_lib
    yield os.path.join(os.getcwd(), 'module.dll')
    yield os.path.join(os.getcwd(), 'module.so')


def _load_library(path: str):
    if os.name == 'nt':
        return ctypes.WinDLL(path)
    return ctypes.CDLL(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib', default=None, help='Path to the compiled shared library')
    args = parser.parse_args()

    lib_name = None
    for candidate in _candidate_libs(args.lib):
        if candidate and os.path.exists(candidate):
            lib_name = candidate
            break

    if not lib_name:
        print('No compiled library found (module.dll/module.so)')
        sys.exit(1)

    try:
        lib = _load_library(lib_name)
    except Exception as e:
        print('Failed to load library:', e)
        sys.exit(1)

    if not hasattr(lib, 'run_kernel'):
        print('Library does not export run_kernel; exiting')
        sys.exit(1)

    try:
        lib.run_kernel()
        print('run_kernel executed')
        sys.exit(0)
    except Exception as e:
        print('Error while calling run_kernel:', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
