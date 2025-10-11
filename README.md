## Datasets
Please donwload the datasets and put them under the data directory.
+ [MicroVideo](https://huggingface.co/datasets/reczoo/MicroVideo1.7M_x2)
+ [KuaiVideo](https://huggingface.co/datasets/reczoo/KuaiVideo_x2)
+ [EBNeRD-Small](https://huggingface.co/datasets/reczoo/Ebnerd_small_x1)

## Run Example
python run_expid.py --config benchmark/ETA_microvideo1.7m_x2/Microvideo --expid ETA_MicroVideo1.7M_x2_000 --gpu 0 --method CollectiveKV --usr_dim 1 --pool_size 10000 --loss_balance_weight 1.0 --loss_peak_weight 0.01
