"""
A Pytorch Implement of Vision Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.vit_embedding = VitEmbedding()
        self.transformer_encoder = TransformerEncoder()
        self.mlp_head = MlpHead()
        self.fc_out = nn.Linear(768, 10)

    def forward(self, pixel_value): 
        ### pixel_value is torch.size([b, 3, 224, 224])
        hidden_states = self.vit_embedding(pixel_value)  # [b, 197, 768]
        hidden_states = self.transformer_encoder(hidden_states) # [b, 197, 768]
        poolout, last_hidden = self.mlp_head(hidden_states)  # [b, 768]
        logtic = self.fc_out(poolout) # [b, 2]
        return logtic
    
class VitEmbedding(nn.Module):
    def __init__(self):
        super(VitEmbedding, self).__init__()
        image_size = 224
        patch_size = 16
        num_channels = 1
        hidden_states = 768
        num_patches = (image_size//patch_size)**2
        self.image2patch = PatchEmbedding(image_size, patch_size, num_channels, num_patches, hidden_states)
        self.cls_token = nn.Parameter(torch.rand(1, 1, hidden_states))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_states))

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        patch_embeddings = self.image2patch(hidden_states)
        
        # expand
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        position_embeddings = self.position_embedding.expand(batch_size, -1, -1)
        
        # [b, 197, 768]
        patch_embeddings = torch.cat((patch_embeddings, cls_tokens), dim = 1)
        
        # position_embedding + embeddings 
        embeddings = position_embeddings + patch_embeddings
        return embeddings
    
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, num_patches, hidden_states):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Conv2d(num_channels, hidden_states, patch_size, patch_size)

    def forward(self, hidden_states):
        hidden_states = self.projection(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1,2)
        return hidden_states

class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.ModuleList(EncoderLayer() for _ in range(6))
    
    def forward(self, hidden_states):
        for layer in self.encoder_layer:
            hidden_states = layer(hidden_states)
        return hidden_states

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(768)
        self.multi_head_attention = MultiHeadSelfAttention(768, 4)
        self.norm2 = nn.LayerNorm(768)
        self.mlp = Mlp(768)

    def forward(self, hidden_states):
        x0 = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.multi_head_attention(hidden_states)
        hidden_states = x0 + hidden_states

        x1 = hidden_states
        hidden_states = self.norm2(x1)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + x1
        
        return hidden_states
    
class Mlp(nn.Module):
    def __init__(self, embedding_dim):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim*4)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embedding_dim*4, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        head_dim = embedding_size // num_heads
        assert (head_dim * num_heads == embedding_size), "embedding_size should be divisible by num_heads"
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.w_q = nn.Linear(embedding_size, embedding_size, bias = False)
        self.w_k = nn.Linear(embedding_size, embedding_size, bias = False)
        self.w_v = nn.Linear(embedding_size, embedding_size, bias = False)
        self.fc_out = nn.Linear(embedding_size, embedding_size)

    def forward(self, hidden_states):
        batch_size, sequence_length = hidden_states.shape[0], hidden_states.shape[1]

        query = self.w_q(hidden_states)
        key = self.w_k(hidden_states)
        value = self.w_v(hidden_states)

        query = query.reshape(batch_size, self.num_heads, sequence_length, self.head_dim)
        key = key.reshape(batch_size, self.num_heads, sequence_length, self.head_dim)
        value = value.reshape(batch_size, self.num_heads, sequence_length, self.head_dim)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * (self.embedding_size ** -0.5)
        attention_scores = F.softmax(attention_scores, dim = -1)
        attention_out = torch.matmul(attention_scores, value)
        attention_out = attention_out.transpose(1,2).flatten(2)
        
        attention_out = self.fc_out(attention_out)
        return attention_out

class MlpHead(nn.Module):
    def __init__(self):
        super(MlpHead, self).__init__()
        self.linear1 = nn.Linear(768, 768)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(768, 768)

    def forward(self, hidden_states):
        cls_token = hidden_states[:,0]
        poolout = self.linear1(cls_token)
        poolout = self.activation(poolout)
        poolout = self.linear2(poolout)
        return poolout, hidden_states
    

if __name__ == "__main__":
    input_tensor = torch.randn(10, 1, 224, 224)
    vit = VisionTransformer()
    output_tensor = vit(input_tensor)
    print(output_tensor.shape)
