from modules.layers.embedders import *
from modules.layers.decoders import *
from modules.layers.layers import BiLSTM, MultiHeadAttention
from modules.layers.lan import LSTM_attention
import torch.nn.functional as F
import torch.nn as nn
import abc

class BERTNerModel(nn.Module, metaclass=abc.ABCMeta):
    """Base class for all BERT Models"""

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError("abstract method forward must be implemented")

    @abc.abstractmethod
    def score(self, batch):
        raise NotImplementedError("abstract method score must be implemented")

    @abc.abstractmethod
    def create(self, *args, **kwargs):
        raise NotImplementedError("abstract method create must be implemented")

    def get_n_trainable_params(self):
        pp = 0
        for p in list(self.parameters()):
            if p.requires_grad:
                num = 1
                for s in list(p.size()):
                    num = num * s
                pp += num
        return pp


class BERTCRF(BERTNerModel):

    def __init__(self, embeddings,lstm_attention1,label_embedding,crf,device="cuda"):
        super(BERTCRF, self).__init__()
        self.embeddings = embeddings
        self.lstm_attention1 = lstm_attention1
        self.label_embedding = label_embedding
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        input_, labels_mask, input_type_ids = batch[:3]
        input_embeddings = self.embeddings(batch)
        label_embeddings = self.label_embedding(batch)
        lstm_out,label_out = self.lstm_attention1.forward(input_embeddings,label_embeddings,labels_mask)
        batch_size  = input_.size(0)
        seq_len     = input_.size(1)
        outs        = label_out.view(batch_size * seq_len, -1)
        _, tag_seq  = torch.max(outs, 1)
        tag_seq     = tag_seq.view(batch_size, seq_len)
        # filter padded position with zero
        tag_seq = labels_mask.long() * tag_seq
        return tag_seq

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        label_embeddings = self.label_embedding(batch)
        lstm_out,label_out = self.lstm_attention1.forward(input_embeddings,label_embeddings,labels_mask)
        batch_size = input_embeddings.size(0)
        seq_len = input_embeddings.size(1)
        loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        outs = label_out.view(batch_size * seq_len, -1)
        score = F.log_softmax(outs, 1)
        total_loss = loss_function(score, labels.view(batch_size * seq_len))
        total_loss = total_loss / batch_size
        return total_loss

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # BiLSTM params
               embedding_size=768, hidden_dim=768, rnn_layers=1, lstm_dropout=0.3,
               # CRFDecoder params
               crf_dropout=0.5,
               # Global params
               device="cuda"):
        
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        label_embedding = LabelEmbedder.create(device=device)
        lstm_attention1 = LSTM_attention(embed_size=embedding_size, hidden_dim=hidden_dim,num_heads=8,att_hidden_dim=768//2,dropout_rate=0.3,mode = "concat")
        crf = NCRFDecoder.create(label_size, hidden_dim, crf_dropout)
        
        return cls(embeddings,lstm_attention1,label_embedding,crf,device)







