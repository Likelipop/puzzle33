#!/usr/bin/env bash

set -e

if [ -z "$DRIVE_PATH" ]; then
  echo "Usage: DRIVE_PATH=/path/to/GoogleDrive ./scripts/push_to_drive.sh"
  exit 1
fi

mkdir -p "$DRIVE_PATH/puzzle33"
cp data/processed/training_data.csv "$DRIVE_PATH/puzzle33/"

echo "Copied training_data.csv to $DRIVE_PATH/puzzle33/"
