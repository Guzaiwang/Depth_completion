cd ../src
python main_w_unc.py --dir_data /DATA/i2r/guzw/workspace/3D/nyudepthv2 --data_name NYU  --split_json ../data_json/nyu.json \
    --gpus 0,1,2,3,4,5,6,7 --loss 1.0*L1+1.0*L2 --batch_size 16 --start_epoch 72 --port 22570 --milestones 36 48 56 64 72 --epochs 100 --pretrain /DATA/i2r/guzw/workspace/confidence/CompletionFormer/data_json/NYUv2.pt \
    --log_dir ../experiments/ --save train_nyu_w_unc 
    

# >> ../logs/nyu_train_w_unc_resume41peoch_DGX5_gpu4567.txt --save_full 2>&1 &

