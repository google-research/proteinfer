#!/bin/bash

while true; do
	echo "compiling"
	./do_compile.sh
	fswatch -1 ./
done
