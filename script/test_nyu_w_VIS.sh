# cd ../src
# python3 main_w_Vis.py --dir_data /DATA/i2r/guzw/workspace/3D/nyudepthv2 --data_name NYU  --split_json ../data_json/nyu.json --gpus 0 --max_depth 10.0 --num_sample 500 --save_image \
#     --test_only --pretrain /DATA/i2r/guzw/workspace/confidence/CompletionFormer/experiments/231116_055009_train_nyu_w_unc_loss_weight/model_00100.pt --log_dir ../experiments/Vis/ --save w_unc_loss_weight_ckpt_100 >> ../logs/Vis_w_unc_loss_weight_ckpt_100_rmse.txt 2>&1 &


# cd ../src
# CUDA_VISIBLE_DEVICES=7 python3 main_w_Vis.py --dir_data /DATA/i2r/guzw/workspace/3D/nyudepthv2 --data_name NYU  --split_json ../data_json/nyu.json --gpus 0 --max_depth 10.0 --num_sample 500 --save_image \
#     --test_only --pretrain /DATA/i2r/guzw/workspace/confidence/CompletionFormer/experiments/debugs/231117_070056_train_nyu_w_confidence/model_00095.pt --log_dir ../experiments/Vis/ --save w_confidence_ckpt_95 >> ../logs/Vis_w_confidence_ckpt_95_rmse.txt 2>&1 &

cd ../src
python3 main_w_Vis.py --dir_data /DATA/i2r/guzw/workspace/3D/nyudepthv2 --data_name NYU  --split_json ../data_json/nyu.json --gpus 7 --max_depth 10.0 --num_sample 500 --save_image \
    --test_only --pretrain /DATA/i2r/guzw/workspace/confidence/CompletionFormer/experiments/231115_141309_train_nyu_w_unc/model_00099.pt --log_dir ../experiments/Vis/ --save w_unc_ckpt_99 >> ../logs/Vis_w_unc_ckpt_99_rmse.txt 2>&1 &
