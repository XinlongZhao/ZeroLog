import sys
sys.path.extend([".", ".."])
from module.Attention import *
from module.CPUEmbedding import *
from module.Common import *
from module.Optimizer import Optimizer


def get_updated_network(old, new, lr, load=False):
    updated_theta = {}
    state_dicts = old.state_dict()
    param_dicts = dict(old.named_parameters())

    for i, (k, v) in enumerate(state_dicts.items()):
        if k in param_dicts.keys() and param_dicts[k].grad is not None:
            updated_theta[k] = param_dicts[k] - lr * param_dicts[k].grad
        else:
            updated_theta[k] = state_dicts[k]
    if load:
        new.load_state_dict(updated_theta)
    else:
        new = put_theta(new, updated_theta)
    return new


def put_theta(model, theta):
    def k_param_fn(tmp_model, name=None):
        if len(tmp_model._modules) != 0:
            for (k, v) in tmp_model._modules.items():
                if name is None:
                    k_param_fn(v, name=str(k))
                else:
                    k_param_fn(v, name=str(name + '.' + k))
        else:
            for (k, v) in tmp_model._parameters.items():
                if not isinstance(v, torch.Tensor):
                    continue
                tmp_model._parameters[k] = theta[str(name + '.' + k)]

    k_param_fn(model)
    return model


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class AttGRUFeatureExtractor(nn.Module):
    # Dispose Loggers.
    _logger = logging.getLogger('AttGRUFeatureExtractor')
    _logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'AttGRU.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for Attention-Based GRU succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    @property
    def logger(self):
        return AttGRUFeatureExtractor._logger

    def __init__(self, vocab, lstm_layers, lstm_hiddens, dropout):
        super(AttGRUFeatureExtractor, self).__init__()

        self.dropout = dropout
        self.logger.info('==== Model Parameters ====')

        vocab_size, word_dims = vocab.vocab_size, vocab.word_dim

        self.word_embed = CPUEmbedding(vocab_size, word_dims, padding_idx=vocab_size - 1)
        self.word_embed.weight.data.copy_(torch.from_numpy(vocab.embeddings))
        self.word_embed.weight.requires_grad = False

        self.logger.info('Input Dimension: %d' % word_dims)
        self.logger.info('Hidden Size: %d' % lstm_hiddens)
        self.logger.info('Num Layers: %d' % lstm_layers)
        self.logger.info('Dropout %.3f' % dropout)

        self.rnn = nn.GRU(input_size=word_dims, hidden_size=lstm_hiddens, num_layers=lstm_layers, batch_first=True, bidirectional=True, dropout=dropout)

        self.sent_dim = 2 * lstm_hiddens
        self.atten_guide = Parameter(torch.Tensor(self.sent_dim))
        self.atten_guide.data.normal_(0, 1)
        self.atten = LinearAttention(tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim)

    def reset_word_embed_weight(self, vocab, pretrained_embedding):
        vocab_size, word_dims = pretrained_embedding.shape
        self.word_embed = CPUEmbedding(vocab.vocab_size, word_dims, padding_idx=vocab.PAD)
        self.word_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.word_embed.weight.requires_grad = False

    def forward(self, inputs):
        words, masks, word_len = inputs
        embed = self.word_embed(words)
        if self.training:
            embed = drop_input_independent(embed, self.dropout)
        embed = embed.cuda(device)
        batch_size = embed.size(0)
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)
        hiddens, state = self.rnn(embed)
        sent_probs = self.atten(atten_guide, hiddens, masks)
        batch_size, srclen, dim = hiddens.size()
        sent_probs = sent_probs.view(batch_size, srclen, -1)
        represents = hiddens * sent_probs
        represents = represents.sum(dim=1)
        return represents


class AttGRUClassifier(nn.Module):
    def __init__(self, input_dim):
        super(AttGRUClassifier, self).__init__()
        self.proj = NonLinear(input_dim, 2)

    def forward(self, features):
        clssifier_x  = self.proj(features)
        return clssifier_x


class SourceClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SourceClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(),
            nn.LayerNorm(128),  
            nn.Dropout(0.3),  
            nn.Linear(128, 2)
        )

    def forward(self, features):
        discriminator_x = self.fc(features)
        return discriminator_x


class Learner(nn.Module):
    def __init__(self, vocab, num_layer, lstm_hiddens, dropout):
        super(Learner, self).__init__()

        self.feature_extractor = AttGRUFeatureExtractor(vocab, num_layer, lstm_hiddens, dropout)
        self.classifier = AttGRUClassifier(input_dim=2*lstm_hiddens)
        self.discriminator = SourceClassifier(input_dim=2*lstm_hiddens)

    def forward(self, inputs):
        # Step 1: Feature Extraction
        features = self.feature_extractor.forward(inputs)
        
        # Step 2: Classification Prediction
        clssifier_x = self.classifier.forward(features)
        
        # Step 3: Source Domain Classification
        discriminator_x = self.discriminator.forward(features)

        return clssifier_x, discriminator_x


class Meta(nn.Module):
    _logger = logging.getLogger('Meta')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'ZeroLog.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
    _logger.addHandler(console_handler)
    # _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for ZeroLog succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    @property
    def logger(self):
        return Meta._logger

    def __init__(self, vocab, num_layer, lstm_hiddens, label2id, dropout, gc_lr, d_lr):
        super(Meta, self).__init__()
        self.vocab = vocab
        self.num_layer = num_layer
        self.hidden_size = lstm_hiddens
        self.label2id = label2id
        self.gc_lr = gc_lr
        self.d_lr = d_lr

        self.model = Learner(self.vocab, self.num_layer, self.hidden_size, dropout)
        self.bk_model = Learner(self.vocab, self.num_layer, self.hidden_size, dropout)
        if torch.cuda.is_available():
            self.model = self.model.cuda(device)
            self.bk_model = self.bk_model.cuda(device)

        self.meta_optim_gc = Optimizer(filter(lambda p: p.requires_grad, self.model.feature_extractor.parameters()), lr=self.gc_lr)
        self.meta_optim_c = Optimizer(filter(lambda p: p.requires_grad, self.model.classifier.parameters()), lr=self.d_lr)
        self.meta_optim_d = optim.Adam(self.model.discriminator.parameters(), lr=self.d_lr)

        self.loss = nn.BCELoss()
        self.test_batch_size = 1024

    def cross_entropy(self, input, target):
        return F.cross_entropy(input, target)

    def forward(self, SX_spt, TX_spt, SY_spt, gamma, beta):
        src_words, _, _ = SX_spt
        batch_size = src_words.shape[0]
        domain_label_0 = torch.zeros(batch_size)
        domain_label_0 = domain_label_0.long().to(device)

        src_words, _, _ = TX_spt
        batch_size = src_words.shape[0]
        domain_label_1 = torch.ones(batch_size)
        domain_label_1 = domain_label_1.long().to(device)

        # update D
        class_output, domain_output = self.model(SX_spt)
        err_s_domain = self.cross_entropy(domain_output, domain_label_0)

        _, domain_output = self.model(TX_spt)
        err_t_domain = self.cross_entropy(domain_output, domain_label_1)

        losses_d = beta * (err_t_domain + err_s_domain)

        self.meta_optim_d.zero_grad()
        losses_d.backward()
        self.meta_optim_d.step()

        # update C
        SX_spt_feature = self.model.feature_extractor(SX_spt)
        class_output = self.model.classifier(SX_spt_feature)
        class_output = F.softmax(class_output, dim=1)
        err_s_label = self.loss(class_output, SY_spt)

        losses_c = gamma * err_s_label
        self.meta_optim_c.zero_grad()
        losses_c.backward()
        self.meta_optim_c.step()

        # update G
        SX_spt_feature = self.model.feature_extractor(SX_spt)
        class_output = self.model.classifier(SX_spt_feature)
        class_output = F.softmax(class_output, dim=1)
        domain_output = self.model.discriminator(GradientReversalLayer.apply(SX_spt_feature))
        err_s_label = self.loss(class_output, SY_spt)
        err_s_domain = self.cross_entropy(domain_output, domain_label_0)

        TX_spt_feature = self.model.feature_extractor(TX_spt)
        domain_output = self.model.discriminator(GradientReversalLayer.apply(TX_spt_feature))
        err_t_domain = self.cross_entropy(domain_output, domain_label_1)

        loss_q = beta * (err_s_domain + err_t_domain) + gamma * err_s_label

        return loss_q

    def finetunning(self, SX_qry, TX_qry, SY_qry, gamma, beta):
        src_words, _, _ = SX_qry
        batch_size = src_words.shape[0]
        domain_label_0 = torch.zeros(batch_size)
        domain_label_0 = domain_label_0.long().to(device)

        src_words, _, _ = TX_qry
        batch_size = src_words.shape[0]
        domain_label_1 = torch.ones(batch_size)
        domain_label_1 = domain_label_1.long().to(device)

        # # update D
        # class_output, domain_output = self.model(SX_qry)
        # err_s_domain = self.cross_entropy(domain_output, domain_label_0)
        # 
        # _, domain_output = self.model(TX_qry)
        # err_t_domain = self.cross_entropy(domain_output, domain_label_1)
        # 
        # losses_d = beta * (err_t_domain + err_s_domain)
        # 
        # self.meta_optim_d.zero_grad()
        # losses_d.backward()
        # self.meta_optim_d.step()
        # 
        # # update C
        # SX_qry_feature = self.model.feature_extractor(SX_qry)
        # class_output = self.model.classifier(SX_qry_feature)
        # class_output = F.softmax(class_output, dim=1)
        # err_s_label = self.loss(class_output, SY_qry)
        # losses_c = gamma * err_s_label
        # self.meta_optim_c.zero_grad()
        # losses_c.backward()
        # self.meta_optim_c.step()

        self.bk_model = get_updated_network(self.model, self.bk_model, self.gc_lr).train().cuda(device)

        # update G
        SX_qry_feature = self.bk_model.feature_extractor(SX_qry)
        class_output = self.bk_model.classifier(SX_qry_feature)
        class_output = F.softmax(class_output, dim=1)

        domain_output = self.bk_model.discriminator(GradientReversalLayer.apply(SX_qry_feature))

        err_s_label = self.loss(class_output, SY_qry)
        err_s_domain = self.cross_entropy(domain_output, domain_label_0)

        TX_qry_feature = self.bk_model.feature_extractor(TX_qry)
        domain_output = self.bk_model.discriminator(GradientReversalLayer.apply(TX_qry_feature))
        err_t_domain = self.cross_entropy(domain_output, domain_label_1)

        loss_q = beta * (err_s_domain + err_t_domain) + gamma * err_s_label

        return loss_q

    def predict(self, inputs, threshold=None):
        with torch.no_grad():
            tag_logits, _ = self.model(inputs)
            tag_logits = F.softmax(tag_logits, dim=1)
        if threshold is not None:
            probs = tag_logits.detach().cpu().numpy()
            anomaly_id = self.label2id['Anomalous']
            pred_tags = np.zeros(probs.shape[0])
            for i, logits in enumerate(probs):
                if logits[anomaly_id] >= threshold:
                    pred_tags[i] = anomaly_id
                else:
                    pred_tags[i] = 1 - anomaly_id
        else:
            pred_tags = tag_logits.detach().max(1)[1].cpu()
        return pred_tags, tag_logits

    def evaluate(self, instances, threshold, vocab_dataset, id2tag):
        self.logger.info('Start evaluating by threshold %.3f' % threshold)
        with torch.no_grad():
            self.model.eval()
            globalBatchNum = 0
            TP, TN, FP, FN = 0, 0, 0, 0
            tag_correct, tag_total = 0, 0
            for onebatch in data_iter(instances, self.test_batch_size, False):
                tinst = generate_tinsts_binary_label(onebatch, vocab_dataset, False)
                tinst.to_cuda(device)
                self.model.eval()
                pred_tags, tag_logits = self.predict(tinst.inputs, threshold)
                for inst, bmatch in batch_variable_inst(onebatch, pred_tags, tag_logits, id2tag):
                    tag_total += 1
                    if bmatch:
                        tag_correct += 1
                        if inst.label == 'Normal':
                            TN += 1
                        else:
                            TP += 1
                    else:
                        if inst.label == 'Normal':
                            FP += 1
                        else:
                            FN += 1
                globalBatchNum += 1
            if TP + FP != 0 and TP != 0:
                precision = 100 * TP / (TP + FP)
                recall = 100 * TP / (TP + FN)
                f = 2 * precision * recall / (precision + recall)
                fpr = 100 * FP / (FP + TN)
                self.logger.info('Precision = %d / %d = %.4f, Recall = %d / %d = %.4f F1 score = %.4f, FPR = %.4f'
                                   % (TP, (TP + FP), precision, TP, (TP + FN), recall, f, fpr))
            else:
                self.logger.info('Precision is 0 and therefore f is 0')
                precision, recall, f = 0, 0, 0
        return precision, recall, f

    def evaluate_domin(self, instances, vocab_dataset, source):
        all_preds = []
        all_labels = []
        with torch.no_grad():
            self.model.eval()
            for onebatch in data_iter(instances, self.test_batch_size, False):
                tinst = generate_tinsts_binary_label(onebatch, vocab_dataset, False)
                tinst.to_cuda(device)
                self.model.eval()
                _, domain_output = self.model(tinst.inputs)
                domain_output = F.softmax(domain_output, dim=1)
                predicted_labels = torch.argmax(domain_output, dim=1)

                all_preds.extend(predicted_labels.cpu().numpy())
                if source == 1:
                    all_labels.extend(torch.ones(predicted_labels.size(0)).cpu().numpy())
                else:
                    all_labels.extend(torch.zeros(predicted_labels.size(0)).cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        self.logger.info('Accuracy = %.4f' % (accuracy))
        return accuracy