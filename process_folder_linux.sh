#!/bin/bash

# Check if the required argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <folder>"
    exit 1
fi

# Get the folder path
folder="$1"

# Check if the folder exists
if [ ! -d "$folder" ]; then
    echo "Folder $folder does not exist."
    exit 1
fi

# Loop over audio files in the folder
for file in "$folder"/*.{mp3,wav,ogg,flac}; do
    if [ -f "$file" ]; then
        echo "Processing file: $file"
        python3 diarize.py --audio "$file" \
                              --no-stem \
                              --suppress_numerals \
                              --whisper-model "openai/whisper-medium.en" \
                              --batch-size 8 \
                              --language en \
                              --device "cuda" \
                              --out-dir "/path/to/output/directory"
    fi
done