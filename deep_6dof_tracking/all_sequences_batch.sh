#!/bin/bash

# Specify the path to the directory you want to search
object='dragon'
directoryPath="E:/sequences/$object/processed"
output="../../../output/$object/dragon_wo_pml"
mkdir -p "$output"  # Create output directory if it doesn't exist

# Create errors file
touch "$output/dragon_wo_pml_errors.txt"

# Use find to list all folders in the specified directory
folders=($(find "$directoryPath" -mindepth 1 -maxdepth 1 -type d))

# Iterate through the folders using a for loop
for folder in "${folders[@]}"; do
    # Do something with each folder
    echo "Folder Name: $(basename "$folder")"
    
    # You can add your code here to perform actions on each folder
    mkdir -p "$output/$(basename "$folder")"
    echo "$(basename "$folder")" >> "$output/dragon_wo_pml_errors.txt"

    python ./sequence_test.py \
        -o "$output/$(basename "$folder")" \
        -s "E:/sequences/$object/processed/$(basename "$folder")" \
        -g "../../../data/data_tracking/models/$object" \
        -sv \
        -m "../../models/$object/dragon_wo_pml/model_best.pth.tar" \
        -a res \
        -k cpu \
        -r 15 \
        --results "$output/dragon_wo_pml_errors.txt" \
        -i 3
done
