#!/bin/bash -e

test -f /etc/shinit_v2 && source /etc/shinit_v2

if [ ! -d "/root/scratch-space/models/whisper_small_en" ] || [ -z "$(ls -A /root/scratch-space/models/whisper_small_en)" ]; then
    echo "whisper_small_en directory does not exist or is empty. Running build-whisper.sh..."
    ./build-whisper.sh
else
    echo "whisper_small_en directory exists and is not empty. Skipping build-whisper.sh..."
fi