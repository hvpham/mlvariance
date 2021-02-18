#!/bin/bash

COMMAND=${1}
LOG=${2}

now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE="$LOG_$now.log"

exec &>$LOG_FILE

eval $COMMAND
