DATA_NAME="md_agreement"

python main.py \
    --data_name $DATA_NAME \
    --approach "aart" \
    --batch_size 100 \
    --learning_rate 5e-5 \
    --num_epochs 20 \
    --lambda2 0.1 \
    --embedding_colnames annotator \
    --sort_instances_by text_id \
    --contrastive_alpha 0.1 \
    --num_fake_annotators 0 \
    --max_len 100 \
    --language_model_name roberta-base

#--num_fake_annotators 10
# --lambda2 0.9 --use_majority_weight --embedding_colnames annotatorGender,annotatorPolitics,annotatorRace,altruism,empathy,freeSpeech,harmHateSpeech,traditionalism,lingPurism,racism
# kumar ,race,age_range,lgbtq_status,religion_important,is_parent,political_affilation,education




