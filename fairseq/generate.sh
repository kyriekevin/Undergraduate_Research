fairseq-generate data-bin/wmt16_en_de_bpe32k \
--path checkpoint_best.pt \
--remove-bpe --beam 4 --batch-size 64 --lenpen 0.6 \
--max-len-a 1 --max-len-b 50|tee generate1.out

grep ^T generate.out | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate1.ref

grep ^H generate.out | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate1.sys