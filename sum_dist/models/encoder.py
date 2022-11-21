import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, 
        embedding_dim, 
        num_layer=2, 
        num_head=8, 
        dim_feedforward=2048, 
        decoder_dropout=0.1, 
        activation='relu',
        num_embeddings=None, 
        embeddings=None):

        super().__init__()

        encoder_layer_unit = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_head, 
            dim_feedforward=dim_feedforward, 
            dropout=decoder_dropout, 
            activation=activation)
        
        layer_norm = nn.LayerNorm(embedding_dim)

        self.encoder_layer = nn.TransformerEncoder(
            encoder_layer_unit, 
            num_layers=num_layer, 
            norm=layer_norm)

        if embeddings is not None:
            self.embedding_layer = embeddings
        else:
            self.embedding_layer = nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim)

        return

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Dim:
            input_ids: (B x S)
            src_key_padding_mask: (B x S)
                Bool - True: masked / False: unmasked
        """
        out = self.embedding_layer(input_ids)
        out = self.encoder_layer(src=out.permute(1, 0, 2), src_key_padding_mask=attention_mask)
        out = out.permute(1, 0, 2)  # B x S x E
        return out  # B x S x E

    def get_embedding_layer(self):
        return self.embedding_layer
