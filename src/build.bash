#!/bin/bash
set -e
echo "Building..."
make opt
echo "Running..."
./hello
echo "Displaying..."
firefox test.png
