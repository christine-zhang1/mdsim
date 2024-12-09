# for running simulations and calculating energies on results to check energy conservation
set -e

# Define the CSV file path
csv_file="md_scripts/spice_molecules_set_copy.csv"

# Skip the header and loop through the values in the 2nd column
tail -n +2 "$csv_file" | cut -d ',' -f 2 | while read -r init_idx; do
    echo "Processing init_idx: $init_idx"
    # run simulation
    CUDA_VISIBLE_DEVICES=5 python simulate.py --config_yml configs/simulate/spice_dt.yml --model_dir MODELPATH/maceoff_split_gemnet_dT_full --init_idx $init_idx

    # check if simulation ran to completion
    line_count1=$(wc -l < "MODELPATH/maceoff_split_gemnet_dT_full/md_25ps_123_init_$init_idx/thermo.log")
    if [ "$line_count1" -gt 500 ]; then
        CUDA_VISIBLE_DEVICES=5 python simulate.py --config_yml configs/simulate/spice_t.yml --model_dir MODELPATH/spice_all_gemnet_t_maceoff_split_mine --init_idx $init_idx
        line_count2=$(wc -l < "MODELPATH/spice_all_gemnet_t_maceoff_split_mine/md_25ps_123_init_$init_idx/thermo.log")
        if [ "$line_count2" -gt 500 ]; then
            python md_scripts/psi4_formatted.py $init_idx
        fi
    fi

    echo "Finished init_idx $init_idx" >> run5_finished_idxs.txt

done

echo "all done"
