# for running simulations and calculating energies on results to check energy conservation
set -e

# Loop through each line in the molecules file
while IFS=' ' read -r number molecule; do
    # Use the number as a parameter
    echo "Processing number: $number"
    # run simulation
    CUDA_VISIBLE_DEVICES=8 python simulate.py --config_yml configs/simulate/spice_dt.yml --model_dir MODELPATH/maceoff_split_gemnet_dT_30k --init_idx $number

    # check if simulation ran to completion
    line_count1=$(wc -l < "MODELPATH/maceoff_split_gemnet_dT_30k/md_25ps_123_init_$number/thermo.log")
    if [ "$line_count1" -gt 500 ]; then
        CUDA_VISIBLE_DEVICES=8 python simulate.py --config_yml configs/simulate/spice_t.yml --model_dir MODELPATH/spice_all_gemnet_t_maceoff_split_30k_mine --init_idx $number
        line_count2=$(wc -l < "MODELPATH/spice_all_gemnet_t_maceoff_split_30k_mine/md_25ps_123_init_$number/thermo.log")
        if [ "$line_count2" -gt 500 ]; then
            python md_scripts/psi4_30k.py $number
        fi
    fi
done < md_scripts/str_atoms_leq_6_30k.txt

echo "all done"