#!/bin/bash
echo "Starting dummy root process $$"
sleep 1000 &
CHILD1=$!
sleep 2000 &
CHILD2=$!
echo "Children started: $CHILD1, $CHILD2"
wait
