cd ../src
python3 main.py --dir_data /DATA/i2r/guzw/dataset/kitti_depth/ --data_name KITTIDC --split_json ../data_json/kitti_dc.json \
    --patch_height 240 --patch_width 1216 --gpus 0,1,2,3,4,5,6,7 --loss 1.0*L1+1.0*L2 --lidar_lines 64 \
    --batch_size 5 --max_depth 90.0 --lr 0.001 --epochs 250 --milestones 150 180 210 240 \
    --top_crop 100 --test_crop --log_dir ../experiments/kitti/ --save train_init >> ../logs/kitti_train_8gpu.txt 2>&1 &


