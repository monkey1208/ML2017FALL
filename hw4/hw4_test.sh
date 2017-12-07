wget 'https://www.dropbox.com/s/nfdhf9lnbr4bkeo/word_embedding50.model.bin?dl=1' -O models/word_embedding50.model.bin
wget 'https://www.dropbox.com/s/vycgwj3nyajiwmk/word_embedding250.model.bin.syn1neg.npy?dl=1' -O models/word_embedding250.model.bin.syn1neg.npy
wget 'https://www.dropbox.com/s/7ih0vfjkzuiotnv/word_embedding250.model.bin.wv.syn0.npy?dl=1' -O models/word_embedding250.model.bin.wv.syn0.npy
python3 ensemble.py $1 $2
