import torch
import numpy as np
import pandas as pd
from params import params
from dataclasses import dataclass, field
from utils import load_compute_metrics
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler


def set_seed(seed):
    import os
    import random
    import transformers
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


@dataclass
class MyTrainingArguments(TrainingArguments):
    # lambda1: float = field(default=np.nan,
    #                        metadata={"help": "lambda1 coefficient for l1_norm of anntoator embeddings in the loss"})
    lambda2: float = field(default=np.nan,
                           metadata={"help": "lambda2 coefficient for l2_norm of annotator embeddings in the loss"})
    contrastive_alpha: float = field(default=np.nan,
                                     metadata={
                                         "help": "The coefficient for contrastive loss computed for annotator embeddings"})
    shuffle_train_data: bool = field(default=True,
                                     metadata={"help": "Whether to shuffle input data"})
    # epoch_freeze_bert: int = field(default=2,
    #                        metadata={"help": "which epoch to start freezing bert"})

    # ratio_inactive_steps: float = field(default=1.0,
    #                                     metadata={"help": "ratio of steps that lambda2 is 0.0"})


class GenericPipeline():
    """Creates a Classifier instance for training single, multi-task, and AART models."""

    def __init__(self, main_params):
        """Instantiates the MultiAnnotator model for training classifiers.

        Args:
          params: a Params instance which includes the hyper-parameters of the model
        """
        set_seed(main_params.random_state)
        self.params = params()
        self.params.update(main_params)
        print(self.params)
        self.data_dict = self.read_data()
        # self.weights = list()
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.tokenizations = {}

        self.compute_metrics_function = load_compute_metrics(self)
        print("params of CustomClassifier:")
        print([(k, v) for k, v in self.params.__dict__.items()])

    def get_param_combinations(self, sep1="_", sep2=", ", exclude_list=[]):
        exclude_list = exclude_list + ["skip_test", "skip_majority", "use_majority_weight",
                                       "num_epochs", "early_stopping_patience"]  # "max_len",
        param_combinations = sep2.join(
            key + sep1 + str(val) for key, val in self.params.__dict__.items() if
            key not in exclude_list)

        return param_combinations

    def read_data(self):
        print("~~~~ Reading data ", self.params.data_name)
        data_name_for_path = self.params.data_name
        if self.params.approach == "single":
            data_path = f'./data/multi_task/{data_name_for_path}'
        else:
            data_path = f'./data/{self.params.approach}/{data_name_for_path}'

        data_dict = {}

        if self.params.approach == "aart":
            if type(self.params.embedding_colnames) == str and self.params.embedding_colnames.strip() == "":
                self.params.embedding_colnames = []
            else:
                self.params.embedding_colnames = self.params.embedding_colnames.split(",")

        df = pd.read_csv(f"{data_path}/all_data.csv")
        if self.params.approach == "aart":
            df[self.params.embedding_colnames] = df[self.params.embedding_colnames].replace(
                ['Do Not Wish to Answer', 'MISSING'], "unknown")
            df[self.params.embedding_colnames] = df[self.params.embedding_colnames].fillna(
                'unknown')

        if 'label' in df.columns:
            df['label'] = df['label'].astype(int)

        self.instance_id_col = 'pair_id' if 'pair_id' in df.columns else 'text_id'
        print("Instance id is : ", self.instance_id_col)
        df_annotators = self.get_annotators(df)
        df[f'majority_label'] = self.calculate_majority(df, annotators=df_annotators)
        if 'disagreement_level' not in df.columns:
            df[f'disagreement_level'] = self.calculate_continous_disagreements(df=df, label_or_pred_col='label')
            df[f'disagreement_level'] = df[f'disagreement_level'].round(1)

        df = self.add_fake_annotators(df)
        train_idx = open(f"../splits/{data_name_for_path}/train_{self.params.random_state}.txt").read().splitlines()
        dev_idx = open(
            f"../splits/{data_name_for_path}/dev_{self.params.random_state}.txt").read().splitlines()
        test_idx = open(
            f"../splits/{data_name_for_path}/test_{self.params.random_state}.txt").read().splitlines()

        assert set(df[self.instance_id_col].astype(str)) == set(train_idx + dev_idx + test_idx)

        data_dict["train"] = df.loc[df[self.instance_id_col].astype(str).isin(train_idx)].reset_index(
            drop=True)
        data_dict["dev"] = df.loc[df[self.instance_id_col].astype(str).isin(dev_idx)].reset_index(
            drop=True)
        data_dict["test"] = df.loc[df[self.instance_id_col].astype(str).isin(test_idx)].reset_index(
            drop=True)

        if self.params.sort_instances_by:
            data_dict["train"] = data_dict["train"].sort_values(by=[self.params.sort_instances_by])

        assert set(data_dict["train"][self.instance_id_col]).isdisjoint(
            set(data_dict["test"][self.instance_id_col]))
        assert set(data_dict["train"][self.instance_id_col]).isdisjoint(
            set(data_dict["dev"][self.instance_id_col]))
        assert set(data_dict["dev"][self.instance_id_col]).isdisjoint(
            set(data_dict["test"][self.instance_id_col]))
        assert not (data_dict["train"].empty or data_dict["dev"].empty or data_dict["test"].empty)

        assert len(
            set(self.get_annotators(data_dict["test"])) - set(self.get_annotators(data_dict["train"]))) == 0  # < 5
        assert len(
            set(self.get_annotators(data_dict["dev"])) - set(self.get_annotators(data_dict["train"]))) == 0  # < 5

        assert (len(df_annotators) + 2 * self.params.num_fake_annotators) == len(
            self.get_annotators(data_dict[
                                    "train"])), f"num of annotators in train set must be {(len(df_annotators) + 2 * self.params.num_fake_annotators)}"
        print("Count of annotators in all-data: ", len(df_annotators))
        print("Count of annotators in train:", len(self.get_annotators(data_dict["train"])))
        print("Count of annotators in dev:", len(self.get_annotators(data_dict["dev"])))
        print("Count of annotators in test:", len(self.get_annotators(data_dict["test"])))

        print(f"Approach name is {self.params.approach}")
        return data_dict

    def get_text_embeddings(self, df, language_model, saving_name=""):
        import os
        import pickle
        if df[self.instance_id_col].duplicated().any():
            import pdb;
            pdb.set_trace()
            df = df.drop_duplicates(self.instance_id_col).copy()
        dataset = self.get_batches(df)
        text_to_embeddings_dict = {}
        text_embeddings = []
        for i in range(0, len(dataset), self.params.batch_size):
            input_ids = torch.tensor(dataset['input_ids'][i:i + self.params.batch_size], device="cuda")
            attention_masks = torch.tensor(dataset['attention_mask'][i:i + self.params.batch_size], device="cuda")
            outputs = language_model(input_ids=input_ids,
                                     attention_mask=attention_masks)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            text_embeddings.append(cls_embeddings)
            for j in range(min(self.params.batch_size, cls_embeddings.shape[0])):
                text_to_embeddings_dict[df['text_id'].iloc[i + j]] = cls_embeddings[j]

            input_ids = input_ids.detach().cpu()
            attention_masks = attention_masks.detach().cpu()

        if saving_name:
            # Saving the embedding vectors
            save_dir = f"./results/{self.params.approach}/{self.params.data_name}/text_embeddings"
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/{saving_name}_{self.params.data_name}_{self.params.random_state}_embeddings.pkl",
                      'wb') as fp:
                pickle.dump(text_to_embeddings_dict, fp)
            print(f'Text embeddings saved successfully to file')

            # Saving the plot of text embeddings
            from umap import UMAP
            import seaborn as sns
            import matplotlib.pyplot as plt
            text_embeddings = np.concatenate(text_embeddings, axis=0)
            color_col = df['disagreement_level']

            umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine')
            two_d_results = umap_model.fit_transform(text_embeddings)

            plt.figure(figsize=(16, 10))
            print(two_d_results.shape)
            sns.scatterplot(
                x=two_d_results[:, 0], y=two_d_results[:, 1],
                hue=color_col,
                legend="full",
                alpha=0.7,
            )
            save_dir = f"./results/{self.params.approach}/{self.params.data_name}/text_embeddings_plots"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/{saving_name}_{self.params.data_name}_{self.params.random_state}_embeddings.png')
            return text_to_embeddings_dict

    # def plot_text_embs(self, df, language_model, name_plot):
    #     from umap import UMAP
    #     import seaborn as sns
    #     import matplotlib.pyplot as plt
    #
    #     train_dataset = self.get_batches(df)
    #     color_col = df['disagreement_level']
    #
    #     text_embeddings = []
    #     for i in range(0, len(color_col), 20):
    #         # print(i)
    #         input_ids = torch.tensor(train_dataset['input_ids'][i:i + 20], device="cuda")
    #         attention_masks = torch.tensor(train_dataset['attention_mask'][i:i + 20], device="cuda")
    #         outputs = language_model(input_ids=input_ids,
    #                                  attention_mask=attention_masks)
    #         # input_ids = input_ids.detach().cpu()
    #         # attention_masks = attention_masks.detach().cpu()
    #         hidden = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    #         text_embeddings.append(hidden)
    #     text_embeddings = np.concatenate(text_embeddings, axis=0)
    #
    #     umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine')
    #     two_d_results = umap_model.fit_transform(text_embeddings)
    #
    #     plt.figure(figsize=(16, 10))
    #     print(two_d_results.shape)
    #     sns.scatterplot(
    #         x=two_d_results[:, 0], y=two_d_results[:, 1],
    #         hue=color_col,
    #         legend="full",
    #         alpha=0.7,
    #     )
    #
    #     plt.savefig(f'{name_plot}_{self.params.data_name}_{self.params.random_state}_embeddings.png')

    def run(self):
        score, test_preds_df = self.train_and_test_on_splits(
            train=self.data_dict['train'],
            dev=self.data_dict['dev'],
            test=self.data_dict['test'])
        score = pd.DataFrame(score)
        return score, test_preds_df

    def print_emb_info(self, **kwargs):
        pass

    def save_embeddings(self, **kwargs):
        pass

    def train_and_test_on_splits(self, train, dev, test):
        print("train shape: ", train.shape)
        print("dev shape: ", dev.shape)
        print("test shape: ", test.shape)
        scores = []
        if self.params.approach == "aart":
            train, dev, test = self.encode_values(train.copy(), dev.copy(), test.copy())
        lm = "cardiffnlp/twitter-roberta-base-offensive" if self.params.data_name in ["large",
                                                                                      "risk"] else "roberta-base"
        print("Name of pretrained language model: ", lm)
        model = self._new_model(train_df=train,
                                language_model=lm)
        train_dataset = self.get_batches(train)
        dev_dataset = self.get_batches(dev)
        # param_combinations = )
        print("*** PARAMS *** \n", self.get_param_combinations(sep1=": "))

        self.print_emb_info(model=model)

        epoch_steps = int(train.shape[0] / self.params.batch_size)
        print("Epoch Steps: ", epoch_steps)
        param_combinations = self.get_param_combinations(
            exclude_list=["balance_annotator_weights", "data_name", "approach",
                          "embedding_colnames", "max_len"])
        saving_models_dir = f"./saved_models/{self.params.approach}/{self.params.data_name}_{self.params.embedding_colnames}/{param_combinations}"
        training_args = self.get_trainingargs(num_save_eval_log_steps=int(epoch_steps / 2),
                                              saving_models_dir=saving_models_dir)

        trainer = self.get_trainer(model=model, train_dataset=train_dataset, dev_dataset=dev_dataset,
                                   training_args=training_args)
        print(trainer.args)
        self.get_text_embeddings(df=train.drop_duplicates(self.instance_id_col).copy(),
                                 language_model=model.language_model, saving_name="before")
        trainer.train()
        # print(trainer.state)
        # self.plot_text_embs(df=train.drop_duplicates(self.instance_id_col).copy(), language_model=model.roberta,
        #                     name_plot="after")

        print("viewing embeddings details and ")
        self.print_emb_info(model=model)
        self.save_embeddings(model=model, train_df= train, param_combinations=param_combinations)
        self.get_text_embeddings(df=train.drop_duplicates(self.instance_id_col).copy(),
                                 language_model=model.language_model, saving_name="after")

        print("~~~~~ Dev Masked Preds (individually):")
        dev_preds = trainer.predict(dev_dataset)
        scores_dict = {k[5:]: v for k, v in dev_preds.metrics.items()}

        scores_dict['type'] = "dev"
        dev = self.add_predictions(df=dev, preds=dev_preds)
        dev, scores_dict['corr_disagreement'], scores_dict['avg_disagreement_labels'], scores_dict[
            'avg_disagreement_preds'] = self.calculate_disagreement(df=dev)

        scores_dict['rand_seed'] = self.params.random_state
        scores_dict['early_stop_epoch'] = float(trainer.state.best_model_checkpoint.split("-")[-1]) / epoch_steps
        print("Early Stop Epoch: ", scores_dict['early_stop_epoch'])
        print(scores_dict)
        # scores_dict = {k:v.replace("test", "dev") for k,v in scores_dict.items()}
        scores.append(scores_dict)

        if not self.params.skip_test:
            scores_test, test_df = self.run_tests(trainer, train, test, )
            test_df["rand_seed"] = self.params.random_state
            scores = scores + scores_test
            scores_tmp = None
        else:
            test_df = None
            # import os
            print("Removing the saved models at: ", saving_models_dir)
            # os.rmdir(saving_models_dir)
            import shutil
            shutil.rmtree(saving_models_dir)

        if self.params.approach == "aart":
            for k in model.emb_names:
                print('~' * 30)
                print(k)
                # mean_l1 = torch.norm(getattr(model, f"{k}_embeddings").weight.detach(), p=1, dim=1)#.mean().item()
                l1 = torch.norm(getattr(model, f"{k}_embeddings").weight.detach(), p=1, dim=1)
                print('~' * 30)

                for s in scores:
                    if s['type'] in ['dev', 'test']:
                        s[f"l1_{k}"] = round(l1.mean().item(), 2)
                    else:
                        for i in range(len(self.data_dict[f'{k}_map'])):
                            if self.data_dict[f'{k}_map'][i] == s['type']:
                                s[f"l1_{k}"] = round(l1[i].item(), 2)
                                break
        return scores, test_df

    def get_trainer(self, model, train_dataset, dev_dataset, training_args):
        if self.params.approach == "aart":
            from .aart_pipeline import AARTTrainer
            return AARTTrainer(model=model, train_dataset=train_dataset, eval_dataset=dev_dataset,
                               tokenizer=self.tokenizer, args=training_args,
                               compute_metrics=self.compute_metrics_function,
                               callbacks=[
                                   EarlyStoppingCallback(early_stopping_patience=self.params.early_stopping_patience,
                                                         early_stopping_threshold=0.01)]
                               )
        else:
            from transformers import Trainer
            return Trainer(model=model, train_dataset=train_dataset, eval_dataset=dev_dataset,
                           tokenizer=self.tokenizer, args=training_args,
                           compute_metrics=self.compute_metrics_function,
                           callbacks=[EarlyStoppingCallback(early_stopping_patience=self.params.early_stopping_patience,
                                                            early_stopping_threshold=0.01)]
                           )

    def get_trainingargs(self, num_save_eval_log_steps, saving_models_dir):
        metric_for_best_model = "eval_f1" if self.params.approach == "single" else "eval_macro_f1"
        # metric_for_best_model = "eval_loss"
        training_args = {
            "output_dir": saving_models_dir,
            "evaluation_strategy": "steps",
            "eval_steps": num_save_eval_log_steps,  # Evaluation and Save happens every half_epoch_steps
            "logging_strategy": "steps",
            "logging_steps": num_save_eval_log_steps,
            "load_best_model_at_end": True,
            "metric_for_best_model": metric_for_best_model,
            "greater_is_better": False if metric_for_best_model == "eval_loss" else True,
            "save_strategy": "steps",
            "save_steps": num_save_eval_log_steps,
            "save_total_limit": 3,  # Only last3 models are saved. Older ones are deleted.
            "num_train_epochs": self.params.num_epochs,
            "per_device_train_batch_size": self.params.batch_size,
            "per_device_eval_batch_size": 128,
            "learning_rate": self.params.learning_rate,
            # "weight_decay": 0.01,
            "warmup_steps": 2 * num_save_eval_log_steps,
            "remove_unused_columns": False,
            "seed": self.params.random_state,
            "label_names": self.task_labels,
            # ratio_inactive_steps=0.0
        }
        if self.params.approach == "aart":
            # lambda1=self.params.lambda1,
            training_args["lambda2"] = self.params.lambda2
            training_args["contrastive_alpha"] = self.params.contrastive_alpha
            training_args["shuffle_train_data"] = False if self.params.sort_instances_by else True
            # training_args["epoch_freeze_bert"] = self.params.epoch_freeze_bert

        training_args = MyTrainingArguments(**training_args)
        return training_args

    def get_annotators(self, df):
        pass

    def _get_top_annotators(self, annotators):
        """Finds the top N annotators with highest number of annotations.

        Args:
          annotators: the list of all annotators.

        Returns:
          the list of top N annotators
        """
        # if only a subset of annotators with the highest number of annotations
        # are to be considered in the modeling
        import pdb;
        pdb.set_trace()
        return [anno for anno, count in self.data[annotators].count(axis=0).sort_values(
            ascending=False).items()][:min(self.params.top_n_annotators, len(annotators))]

    def get_batches(self, df):
        pass

    def _new_model(self, train_df, language_model):
        pass

    def _create_loss_label_weights(self, data):
        pass

    def calculate_disagreement(self, df):
        df['continuous_disagreement_labels'] = self.calculate_continous_disagreements(df=df,
                                                                                      label_or_pred_col='label')
        df['continuous_disagreement_preds'] = self.calculate_continous_disagreements(df=df,
                                                                                     label_or_pred_col='pred')
        temp = df.drop_duplicates(self.instance_id_col)
        corr_disagreement = temp['continuous_disagreement_preds'].corr(temp['continuous_disagreement_labels'],
                                                                       method='pearson')
        return df, corr_disagreement.round(3), temp['continuous_disagreement_labels'].mean().round(3), temp[
            'continuous_disagreement_preds'].mean().round(3)

    def run_tests(self, trainer, train, test):
        """
        Gets the trained model, reports performance based on:
            1) majority label vs majority pred
            2) annotators macro or micro performance
        """
        preds = trainer.predict(self.get_batches(test))
        test = self.add_predictions(df=test, preds=preds)
        scores = self._calculate_annotator_performance(train=train, test=test)
        test, scores[-1]['corr_disagreement'], scores[-1]['avg_disagreement_labels'], scores[-1][
            'avg_disagreement_preds'] = self.calculate_disagreement(test)
        epoch_steps = int(train.shape[0] / self.params.batch_size)
        scores[-1]['early_stop_epoch'] = float(trainer.state.best_model_checkpoint.split("-")[-1]) / epoch_steps
        print("~~~~~ Test Masked Preds (individually):")
        print(scores[-1])

        if self.params.majority_inference:
            maj_scores_dict, test_expanded_results = self._calculate_majority_performance(trainer=trainer, test=test,
                                                                                          train=train)
            print("~~~~~ Test All Annotator Head Preds (Majority Vote):")
            print(maj_scores_dict)
            scores.append(maj_scores_dict)
            assert not test_expanded_results.empty
            return scores, test_expanded_results

        return scores, test

    def tokenize_function(self, x):
        if self.instance_id_col == "pair_id":
            if (x["text"], x["parent_text"]) in self.tokenizations:
                return self.tokenizations[(x["text"], x["parent_text"])]

            tokenized_inputs = self.tokenizer(text=x["text"], text_pair=x["parent_text"], padding="max_length",
                                              truncation=True, max_length=self.params.max_len)
            self.tokenizations[(x["text"], x["parent_text"])] = tokenized_inputs
        else:
            if x["text"] in self.tokenizations:
                return self.tokenizations[x["text"]]

            tokenized_inputs = self.tokenizer(text=x["text"], padding="max_length", truncation=True,
                                              max_length=self.params.max_len)
            self.tokenizations[x["text"]] = tokenized_inputs
        return tokenized_inputs

# eMFD, ghc, attitudes
# if "attitudes" in self.params.data_name:
#     self.multilabel = True
#     if self.params.approach == "aart":
#         df = pd.read_csv(f"{data_path}/all_data.csv")
#     else:
#         df = data_dict['df'] = pd.read_csv(f"{data_path}/all_data.csv", header=[0, 1])
#
#     if "small_large" in self.params.data_name.lower() or "large_small" in self.params.data_name.lower():
#         data_dict['df'] = df[df['domain'].isin(["smallScale", "largeScale"])].reset_index(
#             drop=True)
#     elif "risk_large" in self.params.data_name.lower() or "large_risk" in self.params.data_name.lower():
#         data_dict['df'] = df[df['domain'].isin(["sap2019", "largeScale"])].reset_index(drop=True)
#     elif "small" in self.params.data_name.lower():
#         data_dict['df'] = df[df['domain'] == "smallScale"].reset_index(drop=True)
#     elif "large" in self.params.data_name.lower():
#         data_dict['df'] = df[df['domain'] == "largeScale"].reset_index(drop=True)
#     elif "risk" in self.params.data_name.lower():
#         data_dict['df'] = df[df['domain'] == "sap2019"].reset_index(drop=True)
#     else:
#         data_dict['df'] = df
# else:
