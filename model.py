import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # for param in resnet.parameters():
        #     param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.hidden2target = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).cuda(), torch.zeros(1, batch_size, self.hidden_size).cuda()

    def forward(self, features, captions):
        self.batch_size = features.shape[0]
        self.hidden = self.init_hidden(self.batch_size)
        embeds = self.word_embeddings(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeds), dim=1)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        outputs = self.hidden2target(lstm_out)
        return outputs

    def generate_captions(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_tokens = []
        for _ in range(max_len):
            out, states = self.lstm(inputs, states)
            out = self.hidden2target(out.squeeze(1))
            _, out = torch.max(out, 1)

            if inputs.shape[0] == 1:
                predicted_tokens.append(out.item())
                if out == 1:
                    break
            else:
                predicted_tokens.append(out.tolist())
            inputs = self.word_embeddings(out).unsqueeze(1)

        return predicted_tokens
