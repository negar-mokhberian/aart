import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaModel, RobertaForSequenceClassification, BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Union, Tuple
from dataclasses import dataclass, field
from updated_metrics import losses


class MultiTaskClassifier(RobertaForSequenceClassification):
    def __init__(self, config, balancing_weights, task_labels=["majority"]):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # might need to rename this depending on the model
        self.language_model = RobertaModel(config)
        nhid = self.language_model.config.hidden_size
        print("@@@ nhid: ", nhid)

        self.task_labels = task_labels
        self.linear_layer = dict()
        for task in task_labels:
            self.linear_layer[task] = nn.Linear(nhid, self.num_labels).to(torch.device('cuda'))
        self.balancing_weights = balancing_weights
        self.create_loss_functions()
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            # labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.language_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # output will be (batch_size, seq_length, hidden_size)
        hidden = outputs.last_hidden_state[:, 0, :]

        logits = dict()
        for task in self.task_labels:
            logits[task] = self.linear_layer[task](hidden)

        # predictions = {task: [x.item() for x in torch.argmax(logits[task], dim=-1)] for task in self.task_labels}
        labels = {k: kwargs[k] for k in kwargs.keys() if k in self.task_labels}
        loss = self.calculate_loss(labels=labels, logits=logits)

        # logits = torch.cat([logits_tensor for logits_tensor in logits.values()])

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def calculate_loss(self, labels, logits):
        task_loss_dict = dict()

        for task_label in self.task_labels:
            if (labels[task_label] == -1).all().item():
                continue
            if labels[task_label].isnan().all().item():
                continue
            task_loss_dict[task_label] = self.losses[task_label](
                logits[task_label][~torch.any(labels[task_label].isnan().view(-1, 1), dim=1)],
                target=labels[task_label][~torch.any(labels[task_label].isnan().view(-1, 1), dim=1)])
            # (logits[task_label], target=labels[task_label])

        total_loss = sum(task_loss_dict.values())
        return total_loss

    def create_loss_functions(self):
        self.losses = dict()

        for task_label in self.task_labels:
            self.losses[task_label] = nn.CrossEntropyLoss(weight=self.balancing_weights[task_label],
                                                          ignore_index=-1)


@dataclass
class AARTSequenceClassifierOutput(SequenceClassifierOutput):
    ce_loss: Optional[torch.FloatTensor] = None
    l2_norm: Optional[torch.FloatTensor] = None
    contrastive_loss: Optional[torch.FloatTensor] = None


class AARTClassifier(RobertaForSequenceClassification):

    def __init__(self, config, label_weights, annotator_weights=[], embd_type_cnt={}):
        super().__init__(config)
        self.config = config
        self.language_model = RobertaModel(config)
        nhid = self.language_model.config.hidden_size
        print("@@@ nhid: ", nhid)
        print("@@ num labels:", config.num_labels)
        self.emb_names = list(embd_type_cnt.keys())
        for k, cnt in embd_type_cnt.items():
            rand_weight = torch.rand(size=(cnt, nhid))
            setattr(self, f"{k}_embeddings",
                    nn.Embedding.from_pretrained(rand_weight, freeze=False).to(torch.device('cuda')))

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(nhid, config.num_labels)
        self.label_balancing_weights = label_weights
        if not embd_type_cnt:
            self.annotator_balancing_weights = []
        else:
            self.annotator_balancing_weights = annotator_weights

        # Initialize weights and apply final processing
        self.post_init()

    def calculate_loss(self, labels, logits, text_ids, other_args):
        # elif self.config.problem_type == "single_label_classification":
        if len(self.annotator_balancing_weights):
            loss_fct = nn.CrossEntropyLoss(weight=self.label_balancing_weights, ignore_index=-1,
                                           reduction="none")
            classification_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            classification_loss = (classification_loss * self.annotator_balancing_weights[
                kwargs[f"annotator_ids"]]).sum() / \
                                  self.annotator_balancing_weights[kwargs[f"annotator_ids"]].sum()
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.label_balancing_weights, ignore_index=-1)
            classification_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if self.emb_names:
            contrastive_loss_funct = losses.ContrastiveLoss()  # losses.NTXentLoss()
            l2_norm = torch.tensor(0., requires_grad=True)
            contrastive_loss = torch.tensor(0., requires_grad=True)

            for k in self.emb_names:
                l2_norm = l2_norm + torch.linalg.vector_norm(getattr(self, f"{k}_embeddings").weight, dim=1,
                                                             ord=2).mean()
                # todo what will happen to the same embeddings? for example a0 and a0? or hispanic and hispanic?
                contrastive_loss = contrastive_loss + contrastive_loss_funct(
                    getattr(self, f"{k}_embeddings")(other_args[f"{k}_ids"]),
                    labels=labels.view(-1),
                    mask_labels=text_ids)

        return classification_loss, l2_norm, contrastive_loss

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.language_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_embeddings = self.dropout(cls_embeddings)
        batch_embeddings = cls_embeddings
        for k in self.emb_names:
            batch_embeddings = batch_embeddings + getattr(self, f"{k}_embeddings")(kwargs[f"{k}_ids"])

        batch_embeddings = self.LayerNorm(batch_embeddings)
        # batch_embeddings = self.dropout(batch_embeddings)
        logits = self.classifier(batch_embeddings)
        if self.training:
            classification_loss, l2_norm, contrastive_loss = self.calculate_loss(logits=logits, labels=labels,
                                                                                 text_ids=kwargs['text_ids'],
                                                                                 other_args=kwargs)
        else:
            classification_loss = torch.tensor(0., requires_grad=False)
            l2_norm = torch.tensor(0., requires_grad=False)
            contrastive_loss = torch.tensor(0., requires_grad=False)

        logits = torch.cat((kwargs[f"annotator_ids"].reshape(-1, 1), logits), 1)

        return AARTSequenceClassifierOutput(
            ce_loss=classification_loss,
            l2_norm=l2_norm,
            contrastive_loss=contrastive_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
