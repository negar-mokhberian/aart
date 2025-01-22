import os
from datetime import datetime
import pytz
import numpy as np
from pipelines.multitask_pipeline import MultiTaskPipeline
from pipelines.aart_pipeline import AARTPipeline


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The corpus name"
                        )
    
    parser.add_argument("--language_model_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The transformer model name e.g. answerdotai/ModernBERT-base, roberta-base or cardiffnlp/twitter-roberta-base-offensive"
                        )

    parser.add_argument("--approach",
                        default=None,
                        type=str,
                        required=True,
                        choices=["single", "multi_task", "aart"],
                        help="The name of the model to be trained from the list: single, multi_task, summing_embs.")

    parser.add_argument("--max_len",
                        type=int,
                        help="Maximum sentence length after tokenizing")

    parser.add_argument("--batch_size",
                        type=int,
                        help="the batch size for training")

    parser.add_argument("--learning_rate",
                        type=float,
                        help="Learning rate for training the model.")

    parser.add_argument("--num_epochs",
                        type=int,
                        help="maximum number of epochs during the early stopping")

    parser.add_argument("--random_state",
                        default=2023,
                        type=int)

    parser.add_argument("--skip_test",
                        action="store_true")

    # parser.add_argument("--top_n_annotators",
    #                     default=-1,
    #                     type=int,
    #                     help="size of the subset of annotators with maximum number of annotations "
    #                          "to be considered in the model training"
    #                     )
    parser.add_argument("--majority_inference",
                        action="store_true",
                        help="whether or not to infer the majority vote from the trained model"
                        )

    # parser.add_argument("--lambda1",
    #                     default=np.nan,
    #                     type=float,
    #                     help="coefficient
    #                     for l1 regularizing the annotator embeddings"
    #                     )

    parser.add_argument("--lambda2",
                        default=np.nan,
                        type=float,
                        help="coefficient for l2 regularizing the annotator embeddings"
                        )

    parser.add_argument("--contrastive_alpha",
                        default=0,
                        type=float,
                        help="coefficient for contrastive loss for annotator embeddings"
                        )

    parser.add_argument("--embedding_colnames",
                        default="",
                        type=str,
                        help="comma separated string of columns to make embeddings for, e.g. annotator,race,age..."
                        )

    # parser.add_argument("--epoch_freeze_bert",
    #                     default=50,
    #                     type=int,
    #                     help="which epoch start to freeze bert."
    #                     )

    parser.add_argument("--sort_instances_by",
                        default="",
                        type=str,
                        help="If provided then the instances will be sorted by this basis and there won't be shuffling of training instances after each epoch"
                        )

    parser.add_argument("--num_fake_annotators",
                        type=int,
                        default=0,
                        help="number of fake annotators to add. If N is given then will add N maj_vote fake annotators and N opp_maj_vote fake annotators and each are labeling 1/N of data randomly selected. ")

    args = parser.parse_args()
    return args


def get_pipeline(params):
    if params.approach == "multi_task" or params.approach == "single":
        return MultiTaskPipeline(params)
    elif params.approach == "aart":
        return AARTPipeline(params)


def main():
    main_args = parse_args()
    pipeline = get_pipeline(main_args)
    print(pipeline)

    #############################################################################
    ############################## Experiments ##################################
    #############################################################################

    score, test_preds_df = pipeline.run()

    #############################################################################
    ################################ Results ####################################
    #############################################################################
    # saving the results along with the hyper parameters of the model
    score["params"] = pipeline.get_param_combinations(": ", exclude_list=["random_state", "balance_annotator_weights", "lambda1"])
    score["rand_seed"] = main_args.random_state
    score["approach"] = main_args.approach

    pacific = pytz.timezone('US/Pacific')
    sa_time = datetime.now(pacific)
    name_time = sa_time.strftime('%m%d%y-%H:%M')
    score["time"] = name_time

    results_dir = f"./results/{main_args.approach}/{main_args.data_name}"
    os.makedirs(results_dir, exist_ok=True)
    print("Saving results to ", results_dir)


    if main_args.skip_test:
        scores_file = f'{results_dir}/dev_scores_classification_{main_args.approach}.csv'
    else:
        test_predictions_dir = f'{results_dir}/predictions_{name_time}_{main_args.approach}_{main_args.random_state}_{main_args.embedding_colnames}.csv'
        test_predictions_dir = test_predictions_dir.replace(':', '')
        print("predictions dir is: ", test_predictions_dir)
        test_preds_df.to_csv(test_predictions_dir, index=False)
        scores_file = f'{results_dir}/scores_classification_{main_args.approach}_{main_args.embedding_colnames}.csv'

    print("Scores dir is: ", scores_file)

    # the scores of each test is appended as a row to the scores file
    if os.path.exists(scores_file):
        score.to_csv(scores_file, header=False, index=False, mode="a")
    else:
        score.to_csv(scores_file, index=False)


if __name__ == "__main__":
    main()