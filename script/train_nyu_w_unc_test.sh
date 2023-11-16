cd ../src
python main_w_unc.py --dir_data /DATA/i2r/guzw/workspace/3D/nyudepthv2 --data_name NYU  --split_json ../data_json/nyu.json \
    --gpus 0 --max_depth 10.0 --num_sample 500 --epoch 100 --save_image --test_only --pretrain /DATA/i2r/guzw/workspace/confidence/CompletionFormer/experiments/231115_141309_train_nyu_w_unc/model_00100.pt --save test_nyu_w_unc 
    


