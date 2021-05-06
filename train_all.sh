for extrafeatures in "" funny_mark interjections exclamation/question/quotes/dotes rating/comments_count/source/submitted_by funny_mark/interjections/exclamation/question/quotes/dotes/rating/comments_count/source/submitted_by
do
    for embedding in NatashaGlove Word2Vec
    do
        for model in LogisticRegression BernoulliNB Perceptron
        do

            batch_size="2048"
            if [[ "$embedding" == "RuBERT" ]]; then
                embedding_path="./data/Embeddings/rubert_cased_L-12_H-768_A-12_pt/"
                batch_size="512"
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
                python3 sarcsdet/train.py $embedding $model --batch_size $batch_size --embedding_path $embedding_path
            else
                python3 sarcsdet/train.py $embedding $model --extra_features $extrafeatures --batch_size $batch_size --embedding_path $embedding_path
            fi

        done
    done
done
