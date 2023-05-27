set -e

time python wsi_vocab.py

time python ../val_data_process/process_wiktionary.py

time python wsi_preprocessing.py