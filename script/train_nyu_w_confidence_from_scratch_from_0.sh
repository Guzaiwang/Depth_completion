cd ../src
python main_w_confidence.py --dir_data /DATA/i2r/guzw/workspace/3D/nyudepthv2 --data_name NYU  --split_json ../data_json/nyu.json \
    --gpus 2,3,4,5 --loss 1.0*L1+1.0*L2 --batch_size 16 --port 23171 --milestones 36 48 56 64 72 --epochs 100 --lr 0.0005 \
    --log_dir ../experiments/ --save train_nyu_w_confidence_from_epoch0 >> ../logs/train_nyu_w_confidence_from_epoch0_DGX02_gpu2345.txt --save_full 2>&1 &
    

# >> ../logs/nyu_train_w_unc_resume41peoch_DGX5_gpu4567.txt --save_full 2>&1 &

