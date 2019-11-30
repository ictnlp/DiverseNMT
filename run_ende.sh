export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 train.py $DATA \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09 --ddp-backend=no_c10d \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1\
    --max-tokens 2048  --update-freq 4 \
    --no-progress-bar --log-format json --log-interval 25 --save-interval-updates 1000 \
    --save-dir $SAVEPATH \
    --tensorboard-logdir $TBPATH  |tee -a  $LOGPATH
