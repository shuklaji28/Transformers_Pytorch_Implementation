import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) #we provide vocabulary size and dimension of embedding to nn.embedding
        
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) #it takes values between 0 and 1. 1 means all neurons are dropout. 0 means none. Usually values lies between the range 0.1 to 0.5
    
        #creating a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) #this part is numerator in the formula
        div_term = torch.exp(torch.arange(0, d_model,2).float() *(-math.log(10000.0)/d_model)) #this part is denomminator in the formula. It
        #we find these values in log space. 
        
        #now applying sin and cos to our values
        
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) #it'll become a tensor of dimension (1, seq_len, d_model)
        self.register_buffer('pe', pe) #it means our value will be saves when we save the model. We register it as buffer for this purpose.
        
        
    def forward(self,x):
        #we add PE to Embedding
        
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) #this makes our value fixed and not learned during training.
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #mutiplied
        self.bias = nn.Parameter(torch.zeros(1)) #added
    
    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x-mean)/ (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout : float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #here it is W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #here it is w2 and b2
        
    def forward(self,x):
        #What's happening here is -> it'll transform 
        # (Batch, seq_len, d_model) -> (batch, seq_len, d_ff) in linear_1
        # and from d_ff to d_model back in layer_2. 
        
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) #using the format as described in the image above

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h #d_model needs to be divided into h heads. 
        assert d_model % h == 0, "d_model is not divisble by h" #this makes sure that d_model is divisible by h :)
        self.d_k = d_model // h 
        
        self.w_q = nn.Linear(d_model, d_model) #this has dimension d_model by d_model so that when mutiplied later with q, the output will be seq by d_model
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model) #output matrix
        self.dropout = nn.Dropout(dropout)
    
    #this step of writing static method makes sure that you do not need to provide any instance of this class and can be called separately.
    @staticmethod 
    def attention(query,key,value, mask, dropout : nn.Dropout):
        d_k = query.shape[-1] 
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        # we also want to hide certain interactions and therefore hide them before using softmax to get the final output.
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores #here first one is the output and second one will be used for visualization of what our output looks like xd.
            
        
        
    def forward(self, q, k ,v, masks): #masks is used to hide those values in our output which were not related to each other and were not used to find weights for a particular vector.
        query = self.w_q(q) #going from (batch, seq, d_model) -> (batch, seq_dmodel)
        key = self.w_k(k) #going from (batch, seq, d_model) -> (batch, seq_dmodel)
        value = self.w_v(v) #going from (batch, seq, d_model) -> (batch, seq_dmodel)
        
        #now we want to divide these matrices into smaller matrices to give it into heads.
        #note that we do not want to split the sentence. We want to split the embeddings.
        #we are going from (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k) and we do the same thing for all three matrices.
        #This reshaping and transposing allows for separate attention calculations for each head in a parallel manner, enhancing model's ability to capture 
        # diverse relationships within the input.
        
        query  = query.view(query.shape[0], query.shape[1],self.h, self.d_k).transpose(1,2) #this command essentially converts a query tensor into new 4 dimensioanl tensor with dimensions.
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        #we want to two things from here. First output of softmax score and attention scores.
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, masks, self.dropout)
        
        # (batch , h, seq_len, d_k) ---> (batch, seq_len, h, d_k) ---> (batch, seq_len, d_model) --- we go back to original dimension.
        
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
        
        
        # we are going from (batch, seq_len, d_model) ---> (batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        
        
class EncoderBlock(nn.Module):
    def __init__(self, features:int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)]) # we have two skip connections as you can see in the image above. This command basically creates these two connections.
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x,src_mask)) #this is the first connection where input is first paassed through the muti head attention block and then to the add and norm block.
        x = self.residual_connections[1](x, self.feed_forward_block) #this is the second residual block where input is passed to the feed forward block and then again to the Add and norm layer. This command creates the other upper connection as can be seen in image.
        return x
class Encoder(nn.Module):
    
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x= layer(x, mask)
        return self.norm(x)
            
    #this completes the upper part in the Encoder Block which can be run N number of times. We have not combined embedding input as of now in the Encoder Block. We'll do it later.
    # the output after the N number of operations on Encoder block, will be passed to the Decoder Block.

class DecoderBlock(nn.Module):
    def __init__(self, features:int, self_attention_block : MultiHeadAttention, cross_attention_block : MultiHeadAttention, feed_forward_block: FeedForwardBlock,dropout :float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x:self.cross_attention_block(x, encoder_output, encoder_output, src_mask ))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self,x, encoder_output , src_mask, tgt_mask):
        for layer in self.layers:
            x= layer(x, encoder_output,src_mask, tgt_mask)
        return self.norm(x)
    #it ends out decoder block. Now we'll move towards last stage of transformer that is Projection Layer.
    
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model:int, vocab_size:int ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) #we are basically going from size d_model to size of vocabulary. simple mapping from Decoder Blocm to Vocabulary.
    def forward(self, x):
        #we want to go from batch, seq_len, d_model to batch, seq_len, vocab_size
        return torch.log_softmax(self.proj(x), dim = -1)
    
    
class Transformer(nn.Module):
    def __init__(self, encoder : Encoder, decoder = Decoder, src_embed = InputEmbeddings, tgt_embed = InputEmbeddings, src_pos = PositionalEncoding, tgt_pos = PositionalEncoding, projection_layer = ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed  = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        #now we'll define three methods. One to encode, one to decode and one to Project.
    
    def encode(self, src, src_mask):
        src = self.src_embed(src) #we first apply embeddings to the source
        src = self.src_pos(src) #then apply  positional encoding to the input 
        return self.encoder(src, src_mask) #and pass it to the encoder layer
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt  = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)
    
    
        #we havent build a single block that when passed some parameters perform all the operations under the Transformers Architecture.
def build_transformer(src_vocab_size:int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int,
                     d_model:int = 512, N :int = 5, h:int = 8, dropout:float = 0.1, d_ff:int=2048) -> Transformer:
    
    #first we create embedding
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    #create the positional encoding layer
    
    src_pos = PositionalEncoding(d_model,src_seq_len, dropout) #tgt_pos will be same so we might not need to create that.
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    #create the encoder blocks
    
    encoder_blocks = []
    
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    #create the decoder blocks
    
    decoder_blocks = []
    
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model,h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    #create the projection layer
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
        
    #then we build the transformer
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    #initialise parameters so models does nor start with random values for training
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
            
    return transformer
    