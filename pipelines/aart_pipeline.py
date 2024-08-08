import torch
import numpy as np
import pandas as pd
from transformers import Trainer
from pipelines.generic_pipeline import GenericPipeline
from model_architectures import AARTClassifier
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from utils import get_a_p_r_f


class AARTTrainer(Trainer):

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        return DataLoader(train_dataset, batch_size=self._train_batch_size, shuffle=self.args.shuffle_train_data,
                          collate_fn=data_collator)  # , num_workers=4)

   
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


    def lambda_scheduler(self):
        # assert ~(self.args.lambda1 == self.args.lambda1) or ~(self.args.lambda2 == self.args.lambda2)
        # if self.args.lambda1 == self.args.lambda1:
        #     lambda_end = self.args.lambda1
        if self.args.lambda2 == self.args.lambda2:
            lambda_end = self.args.lambda2
        else:
            lambda_end = 0

        current_lambda = lambda_end
        # current_lambda = self._get_cosine_schedule_with_inactive_steps(
        #     current_step=self.state.global_step,
        #     ratio_inactive_steps=self.args.ratio_inactive_steps,
        #     num_training_steps=self.state.max_steps,
        #     num_cycles=2.5,
        #     lambda_max=lambda_end
        # )

        # current_lambda = self._get_polynomial_lambda_schedule_with_inactive_steps(
        #     current_step=self.state.global_step,
        #     ratio_inactive_steps=self.args.ratio_inactive_steps,
        #     num_training_steps=self.state.max_steps,
        #     lambda_end=lambda_end,
        #     power=1,
        #     lambda_init=0.0
        # )
        return current_lambda

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            import pdb;
            pdb.set_trace()
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        assert self.args.past_index < 0
        # Save past state if it exists
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]

        if labels is not None:
            import pdb;
            pdb.set_trace()
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            current_lambda = self.lambda_scheduler()
            loss = current_lambda * outputs["l2_norm"] + (1 - current_lambda - self.args.contrastive_alpha) * outputs[
                "ce_loss"] + self.args.contrastive_alpha * outputs['contrastive_loss']
            if self.state.global_step % 300 == 0:
                print(
                    f"step {self.state.global_step}, lambda: {current_lambda}, contrastive_alpha: {self.args.contrastive_alpha}, \ncurrent l2_norm: {outputs.l2_norm}, \ncurrent ce_loss: {outputs.ce_loss}, \ncurrent contrastive_loss: {outputs.contrastive_loss}")

        return (loss, outputs) if return_outputs else loss


class AARTPipeline(GenericPipeline):

    def calculate_continous_disagreements(self, df, label_or_pred_col='label'):
        majority = df.groupby(self.instance_id_col)[label_or_pred_col].mean() >= 0.5
        count = df.groupby(self.instance_id_col)[label_or_pred_col].count()
        sum = df.groupby(self.instance_id_col)[label_or_pred_col].sum()

        disagreements = [
            1.0 - float(sum[t_i]) / float(count[t_i]) if majority[t_i] else float(sum[t_i]) / float(count[t_i]) for t_i
            in df[self.instance_id_col]]
        return disagreements

    def add_fake_annotators(self, df):
        N = self.params.num_fake_annotators
        if N <= 0:
            return df
        df_annotators = self.get_annotators(df)

        for i in range(N):
            for type in ["maj", "opp"]:
                fake_ann_name = f"annotator_fake_{type}_{i}"
                assert fake_ann_name not in df_annotators
                print(f"*** Adding {fake_ann_name}")
                tmp_df = df.drop_duplicates(self.instance_id_col).copy()
                tmp_df = tmp_df.sample(frac=1 / N, random_state=self.params.random_state)
                tmp_df['annotator'] = fake_ann_name
                tmp_df['label'] = np.abs(
                    tmp_df['majority_label'] - np.random.choice([0, 1], size=tmp_df.shape[0], p=[.9, .1]))
                if type == "opp":
                    tmp_df['label'] = 1 - tmp_df['label']

                df = pd.concat([df, tmp_df], axis=0, ignore_index=True)

        return df.copy()

    def calculate_majority(self, df, annotators=None):
        t = df.groupby(self.instance_id_col)["label"].agg(pd.Series.mode)
        aggregated_labels = t[df[self.instance_id_col]].tolist()
        # There could be a list of modes for each instance (in case there is more than one mode)
        # then we need to choose one of them
        aggregated_labels = [agg_vote if isinstance(agg_vote, (int, np.integer)) else agg_vote[0] for agg_vote in aggregated_labels]
        aggregated_labels = [int(e) for e in aggregated_labels]
        assert "majority_label" not in df.columns
        return aggregated_labels

    def _create_loss_label_weights(self, labels):
        weights = compute_class_weight(class_weight='balanced',
                                       classes=np.unique(labels),
                                       y=labels)
        print("Weights used for labels: ", weights)
        if len(weights) == 1:
            weights = [0.01, 1]
        weights = torch.tensor(weights, dtype=torch.float32, device="cuda")  # .to(self.device)
        return weights

    def _create_loss_annotator_weights(self, annotators):
        annot_codes = np.unique(annotators)
        for i in range(len(annot_codes)):
            assert annot_codes[i] == i
        weights = compute_class_weight(class_weight='balanced',
                                       classes=annot_codes,
                                       y=annotators)
        # print("Weights used for annotators: ", weights)
        if len(weights) == 1:
            weights = [0.01, 1]
        weights = torch.tensor(weights, dtype=torch.float32, device="cuda")  # .to(self.device)
        return weights

    def add_predictions(self, df, preds):
        df['pred'] = preds.predictions[0][:, 1:].argmax(axis=1)
        return df.copy()

    def encode_values(self, train_df, dev_df, test_df):
        encoding_colnames = self.params.embedding_colnames
        if "annotator" not in encoding_colnames:
            encoding_colnames = encoding_colnames + ["annotator"]
        print("encoding colnames: ")
        print(encoding_colnames)

        from sklearn.preprocessing import LabelEncoder
        label_encoders_dict = {}
        ### integer mapping using LabelEncoder
        for emb_col in encoding_colnames:
            assert f"{emb_col}_int_encoded" not in train_df.columns
            label_encoders_dict[emb_col] = LabelEncoder()
            label_encoders_dict[emb_col].fit(train_df[emb_col].squeeze())
            self.data_dict[f'{emb_col}_map'] = {k: v for k, v in
                                                zip(label_encoders_dict[emb_col].transform(
                                                    train_df[emb_col].squeeze().unique()),
                                                    train_df[emb_col].squeeze().unique())}

        for emb_col in encoding_colnames:
            ignore_error = False
            if (emb_col == "annotator") and ("emfd" in self.params.data_name.lower()):
                ignore_error = True
            train_df[f'{emb_col}_int_encoded'] = label_encoders_dict[emb_col].transform(train_df[emb_col].squeeze())
            if ignore_error:
                print(dev_df.shape)
                dev_df = dev_df[dev_df[emb_col].isin(train_df[emb_col])].reset_index(drop=True).copy()
                print(dev_df.shape)
                print(test_df.shape)
                test_df = test_df[test_df[emb_col].isin(train_df[emb_col])].reset_index(drop=True).copy()
                print(test_df.shape)

            dev_df[f'{emb_col}_int_encoded'] = label_encoders_dict[emb_col].transform(dev_df[emb_col].squeeze())
            test_df[f'{emb_col}_int_encoded'] = label_encoders_dict[emb_col].transform(test_df[emb_col].squeeze())

        return train_df, dev_df, test_df

    def _new_model(self, train_df, language_model):
        self.task_labels = None
        embd_type_cnt = {}
        for emb_col in self.params.embedding_colnames:
            assert train_df[emb_col].notnull().all()
            print(
                f"will make {train_df[emb_col].nunique()} embeddings for unique values of {emb_col}: {train_df[emb_col].unique()}")
            embd_type_cnt[emb_col] = train_df[emb_col].nunique()
        print(embd_type_cnt)

        train_labels_list = train_df.drop_duplicates(self.instance_id_col)['majority_label'].astype(int).tolist()
        num_labels = len(set(train_labels_list))

        label_weights = self._create_loss_label_weights(train_labels_list)

        train_annotators_list = train_df['annotator_int_encoded'].astype(int).tolist()

        if self.params.balance_annotator_weights:
            annotator_weights = self._create_loss_annotator_weights(train_annotators_list)
        else:
            annotator_weights = []
        classifier = AARTClassifier.from_pretrained(
            pretrained_model_name_or_path=language_model,
            num_labels=num_labels,
            embd_type_cnt=embd_type_cnt,
            label_weights=label_weights,
            annotator_weights=annotator_weights
        )
        return classifier

    def get_batches(self, df):
        from datasets import Dataset
        ds_dict = {}
        ds_dict['labels'] = df['label'].values
        ds_dict['text_ids'] = df[self.instance_id_col].values
        ds_dict["text"] = df.prep_text.astype(str)
        for colname in df.columns:  # self.params.embedding_colnames:
            if "_int_encoded" in colname:
                emb_col = colname.split("_int_encoded")[0]
                ds_dict[f"{emb_col}_ids"] = df[f"{emb_col}_int_encoded"]

        if 'pair_id' in df:
            ds_dict["parent_text"] = df.prep_parent_text.astype(str)
        ds = Dataset.from_dict(ds_dict)
        tokenized_ds = ds.map(lambda x: self.tokenize_function(x))
        tokenized_ds = tokenized_ds.remove_columns("text")
        if 'pair_id' in df:
            tokenized_ds = tokenized_ds.remove_columns("parent_text")
        print(tokenized_ds)
        return tokenized_ds

    def get_annotators(self, df):
        annotators_list = df.annotator.unique().tolist()
        return annotators_list

    def plot_heatmap(self, mat, fig_dir, fig_name):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        os.makedirs(fig_dir, exist_ok=True)

        cmap = sns.diverging_palette(0, 255, sep=1, n=256)
        plt.figure(figsize=(30, 12))
        y = [self.data_dict['annotator_map'][i] for i in range(mat.shape[0])]
        sns.heatmap(mat, center=0, cmap=cmap, yticklabels=y)
        plt.savefig(f"{fig_dir}/{fig_name}")

    def expand_test(self, df, unique_annotator_int):
        all_texts_df = df[[self.instance_id_col, 'prep_text']].drop_duplicates().copy()
        assert len(unique_annotator_int) == len((set(unique_annotator_int)))
        all_annotators_df = pd.DataFrame({'annotator_int_encoded': unique_annotator_int})
        all_texts_all_annots = pd.merge(all_texts_df.assign(key=1), all_annotators_df.assign(key=1), on='key').drop(
            'key', axis=1)
        all_texts_all_annots['label'] = np.NAN

        print("df shape before appending the missing annotators: ", df.shape)
        result_df = pd.concat([df, all_texts_all_annots], axis=0, ignore_index=True)
        result_df = result_df.drop_duplicates([self.instance_id_col, 'annotator_int_encoded'], keep='first')
        print("df shape after appending the missing annotators: ", result_df.shape)
        assert result_df.shape[0] == df[self.instance_id_col].nunique() * len(unique_annotator_int)
        return result_df

    def plot_embeddings(self, mat, fig_dir, fig_name):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.manifold import TSNE
        import os
        os.makedirs(fig_dir, exist_ok=True)
        plt.figure(figsize=(8, 8))

        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(mat)
        # df_subset['tsne-2d-one'] = tsne_results[:, 0]
        # df_subset['tsne-2d-two'] = tsne_results[:, 1]
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            # palette=sns.color_palette("hls", 10),
            legend="full",
            alpha=0.7
        )

        def label_point(x, y, val, ax):
            a = pd.DataFrame({'x': x, 'y': y, 'val': val})
            for i, point in a.iterrows():
                ax.text(point['x'] + .02, point['y'], str(point['val']))

        y = [self.data_dict['annotator_map'][i] for i in range(mat.shape[0])]
        label_point(tsne_results[:, 0], tsne_results[:, 1], y, plt.gca())
        plt.savefig(f"{fig_dir}/{fig_name}")

    def _calculate_majority_performance(self, trainer, test, train):
        import time
        unique_annotator_codes = test['annotator_int_encoded'].unique().tolist()

        start = time.time()
        test_expanded = self.expand_test(test,
                                         unique_annotator_int=unique_annotator_codes)
        end = time.time()
        print(f'Time spent on expanding test {end - start}')
        test_dataset_expanded = self.get_batches(test_expanded)
        test_dataset_expanded = test_dataset_expanded.remove_columns("labels")
        preds_expanded = trainer.predict(test_dataset_expanded)
        test_expanded['pred'] = preds_expanded.predictions[0][:, 1:].argmax(axis=1).tolist()
        assert test_expanded['pred'].isna().sum() == 0

        test_expanded_results = test_expanded.groupby(self.instance_id_col)[["label", "pred"]].mean()
        test_expanded_results['maj_label'] = test_expanded_results['label'] >= 0.5
        test_expanded_results['maj_pred'] = test_expanded_results['pred'] >= 0.5
        scores_dict_test = {'type': 'majority_all', 'rand_seed': self.params.random_state}
        scores_dict_test['accuracy'], scores_dict_test['precision'], scores_dict_test['recall'], \
            scores_dict_test['f1'] = get_a_p_r_f(labels=test_expanded_results["maj_label"],
                                                 preds=test_expanded_results["maj_pred"])
        return scores_dict_test, test_expanded

    def _calculate_annotator_performance(self, train, test):
        annotator_scores = []
        all_labels = []
        all_predictions = []
        all_f1s = []
        for a in train['annotator_int_encoded'].unique():
            a_results = test.loc[test['annotator_int_encoded'] == a].copy()
            if a_results.empty or train.loc[train['annotator_int_encoded'] == a].shape[0] < 5 or len(a_results) < 5:
                continue
            print('~' * 30)
            print('~' * 30)
            print(f" * * * Performance of {a_results['annotator'].iloc[0]}")
            scores_dict = {}
            scores_dict['accuracy'], scores_dict['precision'], scores_dict['recall'], scores_dict['f1'] = get_a_p_r_f(
                labels=a_results["label"], preds=a_results["pred"])
            all_labels.append(a_results["label"])
            all_predictions.append(a_results["pred"])
            all_f1s.append(scores_dict['f1'])
            scores_dict['type'] = f"{a_results['annotator'].iloc[0]}"
            scores_dict["cnt_train"] = train.loc[train['annotator_int_encoded'] == a].shape[0]
            scores_dict["cnt_test"] = a_results.shape[0]
            train_cnt_dict = train.loc[train['annotator_int_encoded'] == a][
                "label"].squeeze().value_counts().to_dict()
            test_cnt_dict = a_results["label"].squeeze().value_counts().to_dict()

            scores_dict['cnt_contributions_train'] = train_cnt_dict
            scores_dict['cnt_contributions_test'] = test_cnt_dict
            scores_dict['cnt_train_positive'] = train_cnt_dict[1] if 1 in train_cnt_dict else 0
            scores_dict['cnt_test_positive'] = test_cnt_dict[1] if 1 in test_cnt_dict else 0
            scores_dict['sim_to_maj'] = round((a_results['label'] == a_results['majority_label']).mean(), 3)
            print(scores_dict)
            annotator_scores.append(scores_dict)

        all_labels = pd.concat(all_labels, axis=0, ignore_index=True)
        all_predictions = pd.concat(all_predictions, axis=0, ignore_index=True)
        scores_dict_test = {'type': 'test', 'rand_seed': self.params.random_state}
        scores_dict_test['micro_accuracy'], scores_dict_test['micro_precision'], scores_dict_test['micro_recall'], \
            scores_dict_test['micro_f1'] = get_a_p_r_f(labels=all_labels, preds=all_predictions)
        scores_dict_test["macro_f1"] = round(np.mean(all_f1s), 2)
        annotator_scores.append(scores_dict_test)
        return annotator_scores
