#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <x_axis_file.dat> <y_axis_file.dat>"
    exit 1
fi

# Input files
x_file="$1"
y_file="$2"

# Output file
output_file="3D_coordinates.dat"

# Check if input files exist
if [ ! -f "$x_file" ] || [ ! -f "$y_file" ]; then
    echo "Error: One or both input files do not exist."
    exit 1
fi

# Create or clear the output file
> "$output_file"

# Process x-axis file (x and z coordinates, y=0)
while read -r x_coord z_coord; do
    printf "%.10f %.10f %.10f\n" "$x_coord" "0.0000000000" "$z_coord" >> "$output_file"
done < "$x_file"

# Process y-axis file (y and z coordinates, x=0)
while read -r y_coord z_coord; do
    printf "%.10f %.10f %.10f\n" "0.0000000000" "$y_coord" "$z_coord" >> "$output_file"
done < "$y_file"

echo "3D coordinates have been written to $output_file"
