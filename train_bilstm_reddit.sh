for embedding in NatashaGlove Word2Vec
do
    if [[ "$embedding" == "NatashaGlove" ]]; then
        embedding_path="./data/Embeddings/navec_hudlit_v1_12B_500K_300d_100q.tar"
        batch_size="2048"
    elif [[ "$embedding" == "Word2Vec" ]]; then
        batch_size="2048"
        embedding_path="./data/"
    fi

    python3 sarcsdet/train.py $embedding BiLSTM --batch_size $batch_size --embedding_path $embedding_path --data_path "data/Sarcasm_on_Reddit/rus-train-balanced-sarcasm-ling_feat.pkl" --data_source 'reddit'
done

