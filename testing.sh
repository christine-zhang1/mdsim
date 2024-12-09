#!/bin/bash

while IFS=' ' read -r number molecule; do
    # Use the number as a parameter
    echo "Processing number: $number"

    echo "Finished init_idx $number" >> run4_finished_idxs.txt
done < md_scripts/str_atoms_leq_6_copy.txt