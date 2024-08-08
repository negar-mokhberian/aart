import torch
import numpy as np
import pandas as pd
from pipelines.generic_pipeline import GenericPipeline
from model_architectures import MultiTaskClassifier
from sklearn.utils.class_weight import compute_class_weight
from utils import get_a_p_r_f


class MultiTaskPipeline(GenericPipeline):

    def calculate_continous_disagreements(self, df, label_or_pred_col='label'):
        # todo fix for multiclass
        if self.params.approach == "single":
            return [0] * df.shape[0]
        df = df.copy()
        if label_or_pred_col == 'label':
            label_cols = [c for c in df.columns if "annotator_" in c and "pred_annotator_" not in c]
        elif label_or_pred_col == "pred":
            label_cols = [c for c in df.columns if "pred_annotator_" in c]
            for pred_col in label_cols:
                df.loc[df[pred_col[5:]].isna(), pred_col] = np.nan
        df = df.copy()
        df.set_index(self.instance_id_col, inplace=True)
        majority = df[label_cols].mean(axis=1) >= 0.5
        count = df[label_cols].count(axis=1)
        sum = df[label_cols].sum(axis=1)
        df.reset_index(inplace=True)
        disagreements = [
            1.0 - float(sum[t_i]) / float(count[t_i]) if majority[t_i] else float(sum[t_i]) / float(count[t_i]) for t_i
            in df[self.instance_id_col]]
        # print("disagreements")
        # print(len(disagreements))
        # print(disagreements)
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
                labels = np.abs(df['majority_label'] - np.random.choice([0, 1, np.nan], size=df.shape[0],
                                                                        p=[.9 / N, .1 / N, (N - 1) / N]))
                if type == "opp":
                    labels = 1 - labels
                df[fake_ann_name] = labels

        return df.copy()

    def calculate_majority(self, df, annotators):
        return df[annotators].mode(axis=1)[0].astype(int)

    def get_annotators(self, df):
        annotators_list = [c for c in df.columns if
                           (("annotator_" in c) and df[c].count() > 0)]
        # annotators_list = list(set(annotators_list))
        assert len(annotators_list) == len(set(annotators_list))
        return annotators_list

    def add_predictions(self, df, preds):
        annotators = self.get_annotators(df)
        pred_cols = {}
        if self.params.approach == "single":
            df['majority_pred'] = pd.Series(np.argmax(preds.predictions['majority_label'], axis=-1))
        else:
            for i, true_label_col in enumerate(annotators):
                pred_label_col = 'majority_label' if self.params.approach == "single" else true_label_col
                assert df.shape[0] == len(preds.predictions[pred_label_col])
                pred_cols[f'pred_{pred_label_col}'] = pd.Series(np.argmax(preds.predictions[pred_label_col], axis=-1))

            pred_cols = pd.DataFrame(pred_cols)
            df = pd.concat([df, pred_cols], axis=1)

            assert pred_cols.shape[1] == len(annotators)
            df['majority_pred'] = self.calculate_majority(pred_cols, pred_cols.columns.tolist())

        return df.copy()

    def _create_loss_weights(self, data):
        """Creates the weights for each task_label based on the sparsity of data.
        Args:
          data: the dataset, based on which the weights are to be calculated
        """
        class_weights = dict()
        for task_label in self.task_labels:
            if self.params.use_majority_weight:
                # all classifier heads are using the label weight of the majority label
                labels = data["majority_label"].dropna().values.astype(int)
            # else:
            #     # each label weight is calculated separately
            #     labels = data[task_label].dropna().values.astype(int)

            weight = compute_class_weight(
                class_weight="balanced", classes=np.unique(labels), y=labels)
            if len(weight) == 1:
                # if a label does not appear in the data
                weight = [0.5, 0.5]
            weight = torch.tensor(weight, dtype=torch.float32, device="cuda")  # .to(self.device)
            class_weights[task_label] = weight
        return class_weights

    def _new_model(self, train_df, language_model):
        if self.params.approach == "single":
            self.task_labels = ["majority_label"]
        elif self.params.approach == "multi_task":
            self.task_labels = self.get_annotators(train_df)

        class_weights = self._create_loss_weights(train_df)
        classifier = MultiTaskClassifier.from_pretrained(
            pretrained_model_name_or_path=language_model,
            balancing_weights=class_weights,
            task_labels=self.task_labels,
        )  # .to(self.device)
        return classifier

    def get_batches(self, df):
        from datasets import Dataset
        # replacing the unavailable labels with -1
        # -1 will then be masked when calculating the loss in the multi_task_loss function
        d = {t: df[t].replace(np.nan, -1).astype(int) for t in self.task_labels}

        d["text"] = df.prep_text.squeeze().astype(str)
        # d[self.instance_id_col] = df[self.instance_id_col].squeeze()

        if 'pair_id' in df:
            d["parent_text"] = df.prep_parent_text.squeeze().astype(str)

        ds = Dataset.from_dict(d)
        tokenized_ds = ds.map(lambda x: self.tokenize_function(x))
        tokenized_ds = tokenized_ds.remove_columns("text")
        if 'pair_id' in df:
            tokenized_ds = tokenized_ds.remove_columns("parent_text")
        print(tokenized_ds)
        return tokenized_ds

    def _calculate_majority_performance(self, test, trainer=None, train=None):
        scores_dict_test = {'type': 'majority_all', 'rand_seed': self.params.random_state}
        scores_dict_test['accuracy'], scores_dict_test['precision'], scores_dict_test['recall'], \
            scores_dict_test['f1'] = get_a_p_r_f(labels=test['majority_label'], preds=test['majority_pred'])

        return scores_dict_test, test

    def _calculate_annotator_performance(self, train, test):
        annotator_scores = []
        all_labels = []
        all_predictions = []
        all_f1s = []
        for i, true_label_col in enumerate(self.get_annotators(train)):
            annotator_labels = test.loc[test[true_label_col].notnull(), true_label_col]
            if test[true_label_col].count() < 5 or train[true_label_col].count() < 5:
                continue
            pred_label_col = 'majority_pred' if self.params.approach == "single" else f'pred_{true_label_col}'
            annotator_preds = test[pred_label_col][test[true_label_col].notnull()]
            scores_dict = {}
            scores_dict["accuracy"], scores_dict["precision"], scores_dict["recall"], scores_dict["f1"] = get_a_p_r_f(
                labels=annotator_labels, preds=annotator_preds)

            print('~' * 30)
            print('~' * 30)
            print(f" * * * Performance for {true_label_col}")
            all_labels.append(annotator_labels)
            all_predictions.append(annotator_preds)
            all_f1s.append(scores_dict['f1'])

            # scores_dict['rand_seeed'] = self.params.random_state
            scores_dict['type'] = true_label_col

            train_cnt_dict = train[true_label_col].squeeze().value_counts().to_dict()
            test_cnt_dict = test[true_label_col].squeeze().value_counts().to_dict()
            scores_dict['cnt_contributions_train'] = train_cnt_dict
            scores_dict['cnt_contributions_test'] = test_cnt_dict
            assert (1 in train_cnt_dict) or (0 in train_cnt_dict)
            scores_dict['cnt_train_positive'] = train_cnt_dict[1] if 1 in train_cnt_dict else 0
            scores_dict['cnt_test_positive'] = test_cnt_dict[1] if 1 in test_cnt_dict else 0
            scores_dict["cnt_train"] = train[true_label_col].count()
            scores_dict["cnt_test"] = test[true_label_col].count()
            scores_dict['sim_to_maj'] = round(
                (test[true_label_col] == test['majority_label']).sum() / test[true_label_col].count(), 3)
            print(scores_dict)
            annotator_scores.append(scores_dict)

        all_labels = pd.concat(all_labels, axis=0, ignore_index=True)
        all_predictions = pd.concat(all_predictions, axis=0, ignore_index=True)
        scores_dict_test = {'type': 'test', 'rand_seed': self.params.random_state}
        scores_dict_test['micro_accuracy'], scores_dict_test['micro_precision'], scores_dict_test['micro_recall'], \
            scores_dict_test['micro_f1'] = get_a_p_r_f(labels=all_labels, preds=all_predictions)
        scores_dict_test["macro_f1"] = round(np.mean(all_f1s), 2)
        annotator_scores.append(scores_dict_test)
        # print("~~~~~ Test All Annotator Head Preds (Majority Vote):")
        # test = pd.concat([test, pd.DataFrame(preds_all_annotator)], axis=1).copy()

        return annotator_scores
