for extrafeatures in "" funny_mark interjections exclamation/question/quotes/dotes author/subreddit/score funny_mark/interjections/exclamation/question/quotes/dotes/author/subreddit/score
do
    for embedding in RuBERT NatashaGlove TFIDF Word2Vec
    do
        for model in LogisticRegression BernoulliNB Perceptron
        do
            batch_size="2048"
            if [[ "$embedding" == "RuBERT" ]]; then
                embedding_path="./data/Embeddings/rubert_cased_L-12_H-768_A-12_pt/"
                batch_size="300"
            elif [[ "$embedding" == "NatashaGlove" ]]; then
                embedding_path="./data/Embeddings/navec_hudlit_v1_12B_500K_300d_100q.tar"
                batch_size="8062"
            elif [[ "$embedding" == "Word2Vec" ]]; then
                batch_size="8062"
                embedding_path="./data/"
            else
                embedding_path="./data/"
            fi
            if [[ "$extrafeatures" == "" ]]; then
                python3 sarcsdet/train.py $embedding $model --batch_size $batch_size --embedding_path $embedding_path --data_path "data/Sarcasm_on_Reddit/rus-train-balanced-sarcasm-ling_feat.pkl" --data_source 'reddit'
            else
                python3 sarcsdet/train.py $embedding $model --extra_features $extrafeatures --batch_size $batch_size --embedding_path $embedding_path --data_path "data/Sarcasm_on_Reddit/rus-train-balanced-sarcasm-ling_feat.pkl" --data_source 'reddit'
            fi
        done
    done
done
