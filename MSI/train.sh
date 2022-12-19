for seed in 0 1 2 3 4 5 6 7 8 9
do
    python train.py --seed $seed --run_name "c_dcdb_seed$seed" --group "c_dcdb"
done

for seed in 0 1 2 3 4 5 6 7 8 9
do
    python train.py --seed $seed --use_ddi True --ddi_dataset DB --run_name "c_dcdb_ddi(DB)_seed$seed" --group "c_dcdb_ddi(DB)"
done

for seed in 0 1 2 3 4 5 6 7 8 9
do
    python train.py --seed $seed --use_ddi True --ddi_dataset TWOSIDES --run_name "c_dcdb_ddi(TWOSIDES)_seed$seed" --group "c_dcdb_ddi(TWOSIDES)"
done

for seed in 0 1 2 3 4 5 6 7 8 9
do
    python train.py --database DC_combined --seed $seed --run_name "dc_combined_seed$seed" --group "dc_combined"
done

for seed in 0 1 2 3 4 5 6 7 8 9
do
    python train.py --database DC_combined --seed $seed --use_ddi True --ddi_dataset DB --run_name "dc_combined_ddi(DB)_seed$seed" --group "dc_combined_ddi(DB)"
done

for seed in 0 1 2 3 4 5 6 7 8 9
do
    python train.py --database DC_combined --seed $seed --use_ddi True --ddi_dataset TWOSIDES --run_name "dc_combined_ddi(TWOSIDES)_seed$seed" --group "dc_combined_ddi(TWOSIDES)"
done