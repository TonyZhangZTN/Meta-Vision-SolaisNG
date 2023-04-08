#!/bin/sh
cmake -B build -DQt5_DIR="/usr/include/aarch64-linux-gnu/qt5/" -GNinja -DSupport_RK3588=TRUE .
cmake --build build
