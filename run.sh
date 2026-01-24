
datasets=("ETTh1" "electricity"  "traffic")

model=timesfm-2.5-200m-pytorch

for data in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d baseline  -data $data  &
    CUDA_VISIBLE_DEVICES=2 python src/disturb_run_pipeline.py -m $model -d prefix  -data $data  &
    CUDA_VISIBLE_DEVICES=3 python src/disturb_run_pipeline.py -m $model -d insert  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d gaussian_noise  -data $data  &
    CUDA_VISIBLE_DEVICES=5 python src/disturb_run_pipeline.py -m $model -d random_offset_noise  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d missing_data  -data $data  &
    CUDA_VISIBLE_DEVICES=7 python src/disturb_run_pipeline.py -m $model -d suffix  -data $data &

    CUDA_VISIBLE_DEVICES=5 python src/disturb_run_pipeline.py -m $model -d task_sensitive  -data $data  &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d task_dependent  -data $data &
    CUDA_VISIBLE_DEVICES=7 python src/disturb_run_pipeline.py -m $model -d task_reconstruct  -data $data  &

    CUDA_VISIBLE_DEVICES=2 python src/disturb_run_pipeline.py -m $model -d baseline -i 32  -data $data &
    CUDA_VISIBLE_DEVICES=3 python src/disturb_run_pipeline.py -m $model -d baseline -i 64  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 128  -data $data &
    CUDA_VISIBLE_DEVICES=5 python src/disturb_run_pipeline.py -m $model -d baseline -i 256  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d baseline -i 1024  -data $data &

    CUDA_VISIBLE_DEVICES=7 python src/disturb_run_pipeline.py -m timesfm-2.0-500m-pytorch -d baseline  -data $data &
    wait
done


model=TimeMoE-50M

for data in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d baseline  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d prefix  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d insert  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d gaussian_noise  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d random_offset_noise  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d missing_data  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d suffix  -data $data &

    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d task_sensitive  -data $data &
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d task_dependent  -data $data &
    CUDA_VISIBLE_DEVICES=2 python src/disturb_run_pipeline.py -m $model -d task_reconstruct  -data $data &

    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 32  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 64  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 128  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 256  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 1024  -data $data &

    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m TimeMoE-200M -d baseline  -data $data &
    wait
done


model=moirai-1.1-R-small

for data in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d baseline  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d prefix  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d insert  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d gaussian_noise  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d random_offset_noise  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d missing_data  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d suffix  -data $data &

    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d task_sensitive  -data $data &
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d task_dependent  -data $data &
    CUDA_VISIBLE_DEVICES=2 python src/disturb_run_pipeline.py -m $model -d task_reconstruct  -data $data &

    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 32  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 64  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 128  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 256  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 1024  -data $data &

    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.0  -data $data &
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.1  -data $data &
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.2  -data $data &
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.3  -data $data &
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.4  -data $data &
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.5  -data $data &
    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.6  -data $data &
    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.8  -data $data &
    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.9  -data $data &
    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d baseline -t 1.0  -data $data &
    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d baseline -t 1.1  -data $data &
    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d baseline -t 1.2  -data $data &

    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m moirai-1.1-R-base -d baseline  -data $data &
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m moirai-1.1-R-large -d baseline  -data $data &

    wait
done


model=chronos-t5-tiny

for data in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d baseline  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d prefix  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d insert  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d gaussian_noise  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d random_offset_noise  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d missing_data  -data $data &
    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m $model -d suffix  -data $data &

    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d task_sensitive  -data $data &
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d task_dependent  -data $data &
    CUDA_VISIBLE_DEVICES=2 python src/disturb_run_pipeline.py -m $model -d task_reconstruct  -data $data &

    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 32  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 64  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 128  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 256  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m $model -d baseline -i 1024  -data $data &

    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.0  -data $data &
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.1  -data $data &
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.2  -data $data &
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.3  -data $data &
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.4  -data $data &
    CUDA_VISIBLE_DEVICES=1 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.5  -data $data &
    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.6  -data $data &
    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.8  -data $data &
    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d baseline -t 0.9  -data $data &
    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d baseline -t 1.0  -data $data &
    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d baseline -t 1.1  -data $data &
    CUDA_VISIBLE_DEVICES=0 python src/disturb_run_pipeline.py -m $model -d baseline -t 1.2  -data $data &

    CUDA_VISIBLE_DEVICES=6 python src/disturb_run_pipeline.py -m chronos-t5-mini -d baseline  -data $data &
    CUDA_VISIBLE_DEVICES=5 python src/disturb_run_pipeline.py -m chronos-t5-small -d baseline  -data $data &
    CUDA_VISIBLE_DEVICES=7 python src/disturb_run_pipeline.py -m chronos-t5-base -d baseline  -data $data &
    CUDA_VISIBLE_DEVICES=4 python src/disturb_run_pipeline.py -m chronos-t5-large -d baseline  -data $data &

    wait
done
