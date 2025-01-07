#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Please provide the input Matlab file as an argument."
    exit 1
fi

input_file=$1
output_file="${input_file%.*}.dat"
mv "$output_file" .
output_file=$(basename "$output_file")


echo "Input file: $input_file"
echo "Output file: $output_file"

# Extract x_vec and totEnergy_vec values
x_vec_string=$(sed -n '/x_vec = \[/,/];/p' "$input_file" | grep -v 'x_vec' | tr -d '[];')
totEnergy_vec_string=$(sed -n '/totEnergy_vec = \[/,/];/p' "$input_file" | grep -v 'totEnergy_vec' | tr -d '[];')

# Convert strings to arrays
readarray -t x_vec <<< "$x_vec_string"
readarray -t totEnergy_vec <<< "$totEnergy_vec_string"

echo "Number of x values: ${#x_vec[@]}"
echo "Number of energy values: ${#totEnergy_vec[@]}"

# Print first few values for debugging
echo "First few x values: ${x_vec[@]:0:5}"
echo "First few energy values: ${totEnergy_vec[@]:0:5}"

# Write to output file
> "$output_file"  # Clear the file if it exists
for i in "${!x_vec[@]}"; do
    x=$(echo "${x_vec[$i]}" | tr -d ' ')
    energy=$(echo "${totEnergy_vec[$i]}" | tr -d ' ')
    if [[ $x =~ ^[0-9.eE+-]+$ ]] && [[ $energy =~ ^-?[0-9.eE+-]+$ ]]; then
        printf "%.10f %.10f\n" "$x" "$energy" >> "$output_file"
    else
        echo "Invalid data: x=$x, energy=$energy"
    fi
done

echo "Conversion complete. Output written to $output_file"
echo "Output file size: $(wc -c < "$output_file") bytes"

# Print the first few lines of the output file for verification
echo "First few lines of the output file:"
head -n 5 "$output_file"