# for running simulations and calculating energies on results to check energy conservation
set -e

# Loop through each line in the molecules file
while IFS=' ' read -r number molecule; do
    # Use the number as a parameter
    echo "Processing number: $number"
    # run simulation
    CUDA_VISIBLE_DEVICES=1 python simulate.py --config_yml configs/simulate/spice_dt.yml --model_dir MODELPATH/maceoff_split_gemnet_dT_full --init_idx $number

    # check if simulation ran to completion
    line_count1=$(wc -l < "MODELPATH/maceoff_split_gemnet_dT_full/md_25ps_123_init_$number/thermo.log")
    if [ "$line_count1" -gt 500 ]; then
        CUDA_VISIBLE_DEVICES=1 python simulate.py --config_yml configs/simulate/spice_t.yml --model_dir MODELPATH/spice_all_gemnet_t_maceoff_split_mine --init_idx $number
        line_count2=$(wc -l < "MODELPATH/spice_all_gemnet_t_maceoff_split_mine/md_25ps_123_init_$number/thermo.log")
        if [ "$line_count2" -gt 500 ]; then
            python md_scripts/psi4_formatted.py $number
        fi
    fi

    echo "Finished init_idx $number" >> run4_finished_idxs.txt
done < md_scripts/least_force_loss_molecules_copy.txt

echo "all done"
