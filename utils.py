import collections
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score


def load_compute_metrics(pipeline_obj):
    if 'AART' in str(type(pipeline_obj)):
        return _compute_metrics_aart
    elif 'MultiTask' in str(type(pipeline_obj)):
        return _compute_metrics_multi_task


def _compute_metrics_aart(eval_pred):
    logits, labels = eval_pred
    logits = logits[0]
    annotator_ids = logits[:, 0].astype(int)
    logits = logits[:, 1:]

    predictions = np.argmax(logits, axis=-1)
    metric_res = {}
    metric_res["micro_accuracy"], metric_res["micro_precision"], metric_res["micro_recall"], metric_res[
        "micro_f1"] = get_a_p_r_f(labels=labels, preds=predictions)
    all_f1s = []
    for annot_id in set(annotator_ids):
        annotator_labels = labels[annotator_ids == annot_id]
        annotator_preds = predictions[annotator_ids == annot_id]
        assert len(annotator_labels) == len(annotator_preds)
        _, _, _, f = get_a_p_r_f(labels=annotator_labels, preds=annotator_preds, agg_method="macro")
        if len(annotator_labels) > 5:
            all_f1s.append(f)

    metric_res["macro_f1"] = np.mean(all_f1s).round(3)
    return metric_res


def _compute_metrics_multi_task(eval_pred):
    logits, labels = eval_pred
    metric_res = {}
    if len(logits.keys()) == 1 and list(logits.keys())[0] == "majority_label":
        maj_preds = np.argmax(logits['majority_label'], axis=-1)
        assert labels.shape == maj_preds.shape

        metric_res["accuracy"], metric_res["precision"], metric_res["recall"], metric_res["f1"] = get_a_p_r_f(
            labels=labels, preds=maj_preds)

    elif len(logits.keys()) > 1:
        all_predictions = []
        all_labels = []
        all_f1s = []

        for i, t_label in enumerate(logits.keys()):
            labeled_samples = labels[i] > -1
            if labeled_samples.sum() < 5:
                continue

            annotator_labels = labels[i][labeled_samples]
            annotator_logits = logits[t_label][labeled_samples]
            annotator_preds = np.argmax(annotator_logits, axis=-1)
            assert annotator_labels.shape == annotator_preds.shape
            _, _, _, f = get_a_p_r_f(labels=annotator_labels, preds=annotator_preds, agg_method="macro")
            all_f1s.append(f)
            all_labels.append(annotator_labels)
            all_predictions.append(annotator_preds)

        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        metric_res["micro_accuracy"], metric_res["micro_precision"], metric_res["micro_recall"], metric_res[
            "micro_f1"] = get_a_p_r_f(labels=all_labels, preds=all_predictions)
        metric_res["macro_f1"] = np.mean(all_f1s).round(3)

    return metric_res


def get_a_p_r_f(labels, preds, agg_method="macro"):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    a = accuracy_score(y_true=labels, y_pred=preds)
    p, r, f, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds,
                                                 average=agg_method, zero_division=0.0)

    scores = (round(a * 100, 3),
              round(p * 100, 3),
              round(r * 100, 3),
              round(f * 100, 3))
    return scores
