# for running simulations and calculating energies on results to check energy conservation
set -e

# Loop through each line in the molecules file
while IFS=' ' read -r number; do
    # Use the number as a parameter
    echo "Processing number: $number"
    # run simulation
    CUDA_VISIBLE_DEVICES=8 python simulate.py --config_yml configs/simulate/spice_t.yml --model_dir MODELPATH/spice_all_gemnet_t_maceoff_split_mine --init_idx $number

    # check if simulation ran to completion
    line_count1=$(wc -l < "MODELPATH/spice_all_gemnet_t_maceoff_split_mine/md_25ps_123_init_$number/thermo.log")
    if [ "$line_count1" -gt 500 ]; then
        python md_scripts/psi4_one_model.py $number
    fi
done < md_scripts/init_idx_gemnet_dt_maceoff.txt

echo "all done"
