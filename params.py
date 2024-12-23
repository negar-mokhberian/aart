class params():
    def __init__(self):
        self.data_name = None
        # self.transformer_model_name = 'roberta-base'
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.max_len = 128
        self.num_epochs = 20
        self.random_state = 2023
        self.approach = "baseline" #"single
        self.skip_test = False
        # self.top_n_annotators = -1
        self.use_majority_weight = True
        self.majority_inference = False
        self.lambda2 = 0.0
        # self.lambda2 = 0.0
        self.balance_annotator_weights = False
        self.early_stopping_patience = 15
        # self.epoch_freeze_bert = 1
        self.embedding_colnames = ""
        self.sort_instances_by = ""
        self.num_fake_annotators = 0
        self.contrastive_alpha = 0



    def update(self, new):
        if new.approach != "aart":
            try:
                delattr(new, 'lambda2')
                delattr(self, 'lambda2')
                # delattr(new, 'lambda2')
                # delattr(self, 'lambda2')
                # delattr(new, 'embedding_colnames')
                # delattr(self, 'embedding_colnames')
                delattr(new, 'balance_annotator_weights')
                delattr(self, 'balance_annotator_weights')
            except:
                pass

        for k, v in new.__dict__.items():
            if getattr(new, k) is not None and v != "nan":
                print("Changing the default value of {} from {} to {}".format(k, getattr(self, k), v))
                setattr(self, k, v)

    def update_dict(self, new):
        for k, v in new.items():
            if v is not None and v != "nan":
              print("Changing the default value of {} from {} to {}".format(k, getattr(self, k), v))
              setattr(self, k, v)


    # def remove_params(self, drop_list):
    #     for droping_param in drop_list:
    #         delattr(self, droping_param)

    def __str__(self):
       return str(self.__dict__)
