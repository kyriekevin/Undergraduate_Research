# wmt16en2de
# TEXT=wmt16_en_de_bpe32k
# mkdir $TEXT
# tar -xzvf wmt16_en_de.tar.gz -C $TEXT
# fairseq-preprocess --source-lang en --target-lang de \
#   --trainpref $TEXT/train.tok.clean.bpe.32000 \
#   --validpref $TEXT/newstest2013.tok.bpe.32000 \
#   --testpref $TEXT/newstest2014.tok.bpe.32000 \
#   --destdir data-bin/wmt16_en_de_bpe32k \
#   --nwordssrc 32768 --nwordstgt 32768 \
#   --joined-dictionary

TEXT=examples/translation/wmt17_en_de
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20