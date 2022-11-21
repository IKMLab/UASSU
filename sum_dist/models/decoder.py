import torch
import torch.nn as nn

import math


class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        encoder_embed_size, 
        vocab_size, 
        num_layer=2, 
        num_head=8, 
        dim_feedforward=2048, 
        decoder_dropout=0.1, 
        pos_dropout=0.1, 
        pos_max_len=512, 
        activation='relu'):

        super().__init__()

        self.embeddings = None

        self.pos_encoder = PositionalEncoding(
            embed_size=encoder_embed_size, 
            dropout=pos_dropout, 
            max_len=pos_max_len)

        decoder_layer_unit = nn.TransformerDecoderLayer(
            d_model=encoder_embed_size, 
            nhead=num_head, 
            dim_feedforward=dim_feedforward, 
            dropout=decoder_dropout, 
            activation=activation)

        layer_norm = nn.LayerNorm(encoder_embed_size)

        self.decoder_layer = nn.TransformerDecoder(
            decoder_layer_unit, 
            num_layers=num_layer, 
            norm=layer_norm) # input_size = S x B x E

        self.linear_layer = nn.Linear(
            in_features=encoder_embed_size, 
            out_features=vocab_size)
        return


    def forward(
        self, 
        tgt, 
        span_reprs, 
        tgt_mask, 
        memory_mask, 
        tgt_key_padding_mask, 
        memory_key_padding_mask):
        """
        Args:
            `tgt`: (S x B) Token ids.
            `span_reprs`: (S x B x E)
        """
        if self.embeddings is None:
            raise NameError(f'Embedding layer in decoder has not been initiated.')

        tgt_embeds = self.embeddings(tgt) # S x B x E
        
        out = self.pos_encoder(span_reprs)
        
        out = self.decoder_layer(
            tgt=tgt_embeds, 
            memory=out, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask, 
            memory_key_padding_mask=memory_key_padding_mask)
        
        out = self.linear_layer(out) # S x B x V
        
        return torch.transpose(out, 0, 1) # B x S x V

    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Max_S x 1 x E
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            `x`: (S x B x E)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
