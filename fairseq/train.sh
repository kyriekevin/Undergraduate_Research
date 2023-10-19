CUDA_VISIBLE_DEVICES=0,1,2,3
fairseq-train data-bin/wmt16_en_de_bpe32k    \
--arch transformer_wmt_en_de --share-all-embeddings \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
--lr 0.0007 --criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--weight-decay 0.0 --max-tokens 4096  \
--save-dir checkpoints/en-de-16-base   
