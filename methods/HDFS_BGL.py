import sys
sys.path.extend([".", ".."])
from representations.templates.statistics import Simple_template_TF_IDF, Template_TF_IDF_without_clean
from preprocessing.Preprocess import Preprocessor
from preprocessing.datacutter.SimpleCutting import cut_by_BGL_eq_filter, cut_by_HDFS_eq_filter
from utils.Vocab import Vocab
from Meta_Learner import *


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', default='train', type=str, help='train or test')
    argparser.add_argument('--epochs', type=int, default=100, help="epochs")
    argparser.add_argument('--threshold', type=float, default=0.5, help="Anomaly threshold.")
    argparser.add_argument('--parser', default='IBM', type=str, help='Select parser, please see parser list for detail. Default Official.')
    argparser.add_argument('--batch_size', type=int, default=256, help="batch_size")
    argparser.add_argument('--alpha', type=float, default=1.0, help="weight for meta testing")
    argparser.add_argument('--beta', type=float, default=1.0, help="beta")
    argparser.add_argument('--gamma', type=float, default=1.0, help="gamma")
    argparser.add_argument('--num_layer', type=int, default=2, help="num_layer")
    argparser.add_argument('--lstm_hiddens', type=int, default=344, help="lstm_hiddens")
    argparser.add_argument('--dropout', type=float, default=0.5, help="dropout")
    argparser.add_argument('--gc_lr', type=float, default=3e-3, help="gc_lr")
    argparser.add_argument('--d_lr', type=float, default=3e-3, help="d_lr")
    args, extra_args = argparser.parse_known_args()

    parser = args.parser
    mode = args.mode
    threshold = args.threshold
    alpha = args.alpha
    gamma = args.gamma
    beta = args.beta
    num_layer = args.num_layer
    lstm_hiddens = args.lstm_hiddens
    batch_size = args.batch_size
    epochs = args.epochs
    dropout = args.dropout
    gc_lr = args.gc_lr
    d_lr = args.d_lr
    label2id = {'Normal': 0, 'Anomalous': 1}

    # process BGL
    dataset = 'BGL'
    # Training, Validating and Testing instances.
    processor_BGL = Preprocessor()
    template_encoder_BGL = Template_TF_IDF_without_clean() if dataset == 'NC' else Simple_template_TF_IDF()  
    train_BGL, dev_BGL, test_BGL = processor_BGL.process(dataset=dataset, parsing=parser, cut_func=cut_by_BGL_eq_filter, template_encoding=template_encoder_BGL.present)
    # Load Embeddings
    vocab_BGL = Vocab()
    vocab_BGL.load_from_dict(processor_BGL.embedding)

    # process HDFS
    dataset = 'HDFS'
    # Training, Validating and Testing instances.
    processor_HDFS = Preprocessor()
    template_encoder_HDFS = Template_TF_IDF_without_clean() if dataset == 'NC' else Simple_template_TF_IDF()
    train_HDFS, dev_HDFS, test_HDFS = processor_HDFS.process(dataset=dataset, parsing=parser, cut_func=cut_by_HDFS_eq_filter, template_encoding=template_encoder_HDFS.present)
    # Load Embeddings
    vocab_HDFS = Vocab()
    vocab_HDFS.load_from_dict(processor_HDFS.embedding)

    # aggregate vocab and label2id
    vocab = Vocab()
    new_embedding = {}
    for key in processor_BGL.embedding.keys():
        new_embedding[key] = processor_BGL.embedding[key]
    for key in processor_HDFS.embedding.keys():
        new_embedding[key + 432] = processor_HDFS.embedding[key]
    print(new_embedding.keys())
    vocab.load_from_dict(new_embedding)

    dataset = 'BGL'
    save_dir = os.path.join(PROJECT_ROOT, 'outputs')
    output_model_dir = os.path.join(save_dir, 'models/ZeroLog/HDFS_BGL/' + dataset + '_' + parser + '/model')
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    best_file_name = f"HDFSeq_BGLeq_{alpha:.2f}_{beta:.2f}_{gamma:.2f}_numlayer{num_layer}_lstm{lstm_hiddens}_dropout{dropout:.2f}_gc{gc_lr:.1e}_d{d_lr:.1e}_best.pt"
    last_file_name = f"HDFSeq_BGLeq_{alpha:.2f}_{beta:.2f}_{gamma:.2f}_numlayer{num_layer}_lstm{lstm_hiddens}_dropout{dropout:.2f}_gc{gc_lr:.1e}_d{d_lr:.1e}_last.pt"
    best_model_file = os.path.join(output_model_dir, best_file_name)
    last_model_file = os.path.join(output_model_dir, last_file_name)

    # meta learning
    zerolog = Meta(vocab, num_layer, lstm_hiddens, label2id, dropout, gc_lr, d_lr)

    if mode == 'train':
        # Train
        global_step = 0
        bestF = 0
        for epoch in range(epochs):

            zerolog.model.train()
            zerolog.bk_model.train()

            start = time.strftime("%H:%M:%S")
            zerolog.logger.info("Starting epoch: %d | phase: train | start time: %s | learning rate: %s" % (
            epoch, start, zerolog.meta_optim_gc.lr))

            batch_num_train_HDFS = int(np.ceil(len(train_HDFS) / float(batch_size)))
            batch_iter_train_HDFS = 0

            batch_num_train_BGL = int(np.ceil(len(train_BGL) / float(batch_size)))
            batch_iter_train_BGL = 0

            batch_num_dev_HDFS = int(np.ceil(len(dev_HDFS) / float(batch_size)))
            batch_iter_dev_HDFS = 0

            batch_num_dev_BGL = int(np.ceil(len(dev_BGL) / float(batch_size)))
            batch_iter_dev_BGL = 0

            total_bn = max(batch_num_train_HDFS, batch_num_dev_HDFS, batch_num_train_BGL, batch_num_dev_BGL)

            meta_train_loader_HDFS = data_iter(train_HDFS, batch_size, True)
            meta_train_loader_BGL = data_iter(train_BGL, batch_size, True)

            meta_test_loader_HDFS = data_iter(dev_HDFS, batch_size, True)
            meta_test_loader_BGL = data_iter(dev_BGL, batch_size, True)

            for i in range(total_bn):
                zerolog.meta_optim_gc.zero_grad()
                # meta train
                meta_train_batch_HDFS = meta_train_loader_HDFS.__next__()
                tinst_tr_HDFS = generate_tinsts_binary_label(meta_train_batch_HDFS, vocab_HDFS)
                tinst_tr_HDFS.to_cuda(device)

                meta_train_batch_BGL = meta_train_loader_BGL.__next__()
                tinst_tr_BGL = generate_tinsts_binary_label(meta_train_batch_BGL, vocab_BGL)
                tinst_tr_BGL.to_cuda(device)

                loss = zerolog.forward(tinst_tr_HDFS.inputs, tinst_tr_BGL.inputs, tinst_tr_HDFS.targets, gamma, beta)

                loss_value = loss.data.cpu().numpy()
                loss.backward(retain_graph=True)

                batch_iter_train_HDFS += 1
                batch_iter_train_BGL += 1

                # meta test
                meta_test_batch_HDFS = meta_test_loader_HDFS.__next__()
                tinst_test_HDFS = generate_tinsts_binary_label(meta_test_batch_HDFS, vocab_HDFS)
                tinst_test_HDFS.to_cuda(device)

                meta_test_batch_BGL = meta_test_loader_BGL.__next__()
                tinst_test_BGL = generate_tinsts_binary_label(meta_test_batch_BGL, vocab_BGL)
                tinst_test_BGL.to_cuda(device)

                loss_te = alpha * zerolog.finetunning(tinst_test_HDFS.inputs, tinst_test_BGL.inputs,
                                                    tinst_test_HDFS.targets, gamma, beta)
                loss_value_te = loss_te.data.cpu().numpy() / alpha
                loss_te.backward()

                batch_iter_dev_HDFS += 1
                batch_iter_dev_BGL += 1
                # aggregate
                zerolog.meta_optim_gc.step()
                global_step += 1
                if global_step % 100 == 0:
                    zerolog.logger.info("Step:%d, Epoch:%d, meta train loss:%.2f, meta test loss:%.2f" % (
                    global_step, epoch, loss_value, loss_value_te))

                if batch_iter_train_HDFS == batch_num_train_HDFS:
                    meta_train_loader_HDFS = data_iter(train_HDFS, batch_size, True)
                    batch_iter_train_HDFS = 0

                if batch_iter_train_BGL == batch_num_train_BGL:
                    meta_train_loader_BGL = data_iter(train_BGL, batch_size, True)
                    batch_iter_train_BGL = 0

                if batch_iter_dev_HDFS == batch_num_dev_HDFS:
                    meta_test_loader_HDFS = data_iter(dev_HDFS, batch_size, True)
                    batch_iter_dev_HDFS = 0

                if batch_iter_dev_BGL == batch_num_dev_BGL:
                    meta_test_loader_BGL = data_iter(dev_BGL, batch_size, True)
                    batch_iter_dev_BGL = 0

            if test_BGL:
                zerolog.logger.info('Testing on test set.')
                _, _, f = zerolog.evaluate(test_BGL, threshold, vocab_BGL, processor_BGL.id2tag)
                zerolog.evaluate_domin(test_BGL, vocab_BGL, source=1)
                if f > bestF:
                    zerolog.logger.info("Exceed best f: history = %.2f, current = %.2f" % (bestF, f))
                    torch.save(zerolog.model.state_dict(), best_model_file)
                    bestF = f
            zerolog.logger.info('Training epoch %d finished.' % epoch)
            torch.save(zerolog.model.state_dict(), last_model_file)
    else:
        if os.path.exists(last_model_file):
            zerolog.logger.info('=== Final Model ===')
            zerolog.model.load_state_dict(torch.load(last_model_file))
            accuracy = zerolog.evaluate_domin(test_BGL, vocab_BGL, source=1)
            print("BGL Last_model_Accuracy:", accuracy)
            precision, recall, f = zerolog.evaluate(test_BGL, threshold, vocab_BGL, processor_BGL.id2tag)
            print("BGL Last_model_precision:", precision)
            print("BGL Last_model_recall:", recall)
            print("BGL Last_model_f:", f)
        if os.path.exists(best_model_file):
            zerolog.logger.info('=== Best Model ===')
            zerolog.model.load_state_dict(torch.load(best_model_file))
            accuracy = zerolog.evaluate_domin(test_BGL, vocab_BGL, source=1)
            print("BGL Best_model_Accuracy:", accuracy)
            precision, recall, f = zerolog.evaluate(test_BGL, threshold, vocab_BGL, processor_BGL.id2tag)
            print("BGL Best_model_precision:", precision)
            print("BGL Best_model_recall:", recall)
            print("BGL Best_model_f:", f)
        zerolog.logger.info('All Finished')
        