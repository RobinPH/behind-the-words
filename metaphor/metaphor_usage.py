import torch


class MetaphorUsage():
    def __init__(self, get_nlp, use_gpu=True):
        self.nlp = get_nlp
        self.use_gpu = use_gpu

    def predict(self, sentence):
        doc = self.nlp(sentence)

        words = [token.lower_ for token in doc]

        embedded_sentence = self.word2vec.get_vec(words)


        eval_text = torch.stack([embedded_sentence])
        eval_lengths = torch.LongTensor([embedded_sentence.shape[0]])

        self.model.eval()

        with torch.no_grad():
            if self.use_gpu:
                eval_text = eval_text.cuda()
                eval_lengths = eval_lengths.cuda()

            # predicted shape: (batch_size, seq_len, 2)
            predicted = self.model(eval_text, eval_lengths)
            # get 0 or 1 predictions
            # predicted_labels: (batch_size, seq_len)
            x, predicted_labels = torch.max(predicted.data, 2)

        result = [label.item() for label in predicted_labels[0]]

        # print(sentence)
        # print(result)

        return result

    def load_model(self, model_path):
        print(f"Loading Model {model_path}")

        if self.use_gpu:
            self.model = torch.load(model_path)
            self.model = self.model.cuda()
        else:
            self.model = torch.load(
                model_path, map_location=torch.device('cpu'))

        self.model.eval()
        print("Model Loaded")

    def load_word2vec(self, word2vec):
        self.word2vec = word2vec
