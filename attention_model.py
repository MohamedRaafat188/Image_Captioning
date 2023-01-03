import torch
from torch import nn
from torchvision import models


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim=1):
        super(AdditiveAttention, self).__init__()

        self.W_a = nn.Linear(encoder_dim, decoder_dim)
        self.U_a = nn.Linear(decoder_dim, decoder_dim)
        self.V_a = nn.Linear(decoder_dim, attention_dim)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, features, hidden_states):
        f = self.W_a(features)
        h = self.U_a(hidden_states)
        combined_states = torch.tanh(f + h.unsqueeze(1))

        attention_scores = self.V_a(combined_states)
        attention_scores = attention_scores.squeeze(2)
        attention_scores = self._softmax(attention_scores)

        context = features * attention_scores.unsqueeze(2)

        context = context.sum(dim=1)

        return attention_scores, context


class MultiplicativeAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim=1):
        super(MultiplicativeAttention, self).__init__()

        self.W_a = nn.Linear(encoder_dim, decoder_dim)
        self.U_a = nn.Linear(decoder_dim, decoder_dim)
        # self.V_a = nn.Linear(decoder_dim, attention_dim)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, features, hidden_states):
        f = self.W_a(features)
        h = self.U_a(hidden_states)

        attention_scores = torch.einsum('abc, ac -> ab', f, h)
        attention_scores = self._softmax(attention_scores)

        context = features * attention_scores.unsqueeze(2)

        context = context.sum(dim=1)

        return attention_scores, context


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)
        
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.shape[0], -1, features.shape[-1])

        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = AdditiveAttention(encoder_dim, decoder_dim, attention_dim)
        self.lstm_cell = nn.LSTMCell(encoder_dim + embed_size, decoder_dim, bias=True)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        embeds = self.embedding(captions)

        h, c = self.init_hidden_states(features)

        batch_size = features.shape[0]
        len_seq = captions.shape[1]
        num_features = features.shape[1]

        preds = torch.zeros(batch_size, len_seq, self.vocab_size).cuda()
        weights = torch.zeros(batch_size, len_seq, num_features).cuda()

        for i in range(len_seq):
            word_embed = embeds[:, i]
            scores, context = self.attention(features, h)
            lstm_input = torch.cat((context, word_embed), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fc(self.drop(h))

            preds[:, i] = output
            weights[:, i] = scores

        return weights, preds

    def generate_captions(self, features, vocab, max_length=20):
        h, c = self.init_hidden_states(features)

        batch_size = features.shape[0]

        word = torch.zeros(batch_size).long().cuda()
        word_embed = self.embedding(word)

        predicted_tokens = []
        weights = torch.zeros(1, max_length, features.shape[1])

        for i in range(max_length):
            scores, context = self.attention(features, h)
            lstm_input = torch.cat((context, word_embed), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fc(self.drop(h))

            _, output= torch.max(output, 1)

            if features.shape[0] == 1:
                predicted_tokens.append(output.item())
                weights[0, i, :] = scores.squeeze()
                if output.item() == 1:
                    break
            else:
                predicted_tokens.append(output.tolist())
            word_embed = self.embedding(output)

        return weights.squeeze(), predicted_tokens

    def init_hidden_states(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  
        c = self.init_c(mean_encoder_out)
        return h, c


class Attention_Model(nn.Module):
    def __init__(self, embed_size, vocab_size, encoder_dim, decoder_dim, attention_dim):
        super(Attention_Model, self).__init__()

        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.5)

    def forward(self, images, captions):
        features = self.encoder(images)
        weights, preds = self.decoder(features, captions)

        return weights, preds
        
