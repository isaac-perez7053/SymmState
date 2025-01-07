#!/bin/bash

# Check if a filename was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

filename=$1

# Check if the file exists
if [ ! -f "$filename" ]; then
    echo "File not found: $filename"
    exit 1
fi

# Create a temporary file
temp_file=$(mktemp)

# Process the file
while IFS= read -r line
do
    # Process each number in the line
    for number in $line
    do
        if [[ $number =~ -?[0-9]+\.[0-9]+[Ee][-+][0-9]+ ]]; then
            # Extract the exponent and check its value
            exponent=$(echo "$number" | awk -F '[Ee]' '{print $2}' | sed 's/^+//')
            if [[ $exponent =~ ^-[1-9][0-9]+$ ]] || [[ $exponent -le -11 ]]; then
                replacement="0.0000000000"
            else
                replacement=$(awk 'BEGIN {printf "%.10f", '"$number"'}')
            fi
            # Use sed to replace the number in the line
            line=$(echo "$line" | sed "s/$number/$replacement/")
        fi
    done

    # Write the processed line to the temp file
    echo "$line" >> "$temp_file"
done < "$filename"

# Replace the original file with the temporary file
mv "$temp_file" "$filename"

echo "Conversion complete. File $filename has been updated."

