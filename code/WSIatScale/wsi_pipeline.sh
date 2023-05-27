set -e

time python create_inverted_index.py --replacements_dir /home/lucyl/language-map-of-science/logs/replacements/replacements --dataset s2orc --vocab_path /home/lucyl/language-map-of-science/logs/sense_vocab/wsi_vocab_set_98_50.txt --outdir /home/lucyl/language-map-of-science/logs/inverted_index --input_ids_path /home/lucyl/language-map-of-science/data/input_paper_ids/journal_analysis.txt

time python cluster_reps_per_token.py --data_dir /home/lucyl/language-map-of-science/logs/replacements/replacements --dataset s2orc --index_dir /home/lucyl/language-map-of-science/logs/inverted_index --out_dir /home/lucyl/language-map-of-science/logs/word_clusters_lemmed --lemmatize True --resolution 0.0

time python assign_clusters_to_tokens.py --out_dir /home/lucyl/language-map-of-science/logs/sense_assignments_lemmed --index_dir /home/lucyl/language-map-of-science/logs/inverted_index --dataset s2orc --data_dir /home/lucyl/language-map-of-science/logs/replacements/replacements --cluster_dir /home/lucyl/language-map-of-science/logs/word_clusters_lemmed --lemmatize True --resolution 0.0

time python ../data_process/get_docID_to_group.py 