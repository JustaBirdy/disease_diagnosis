import torch
import torch.nn as nn

class SelfAttention(nn.Module): #自注意力模块
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()   #调用父类初始化方法
        self.embed_size = embed_size    #输入向量的长度
        self.heads = heads          #注意力头的个数
        self.head_dim = embed_size // heads #每个注意力头所负责计算的向量长度
        
        assert(self.head_dim * self.heads == self.embed_size),'embed size need to be divisible by heads'#输入向量长度需要能被注意力头的个数整除
        
        self.values = nn.Linear(self.head_dim,self.head_dim,bias=False) #通过计算query与所有位置的key的相似度，可以获得该位置与其他位置的关联程度
        self.keys = nn.Linear(self.head_dim,self.head_dim,bias=False)   #用来衡量与query的相似性的向量表示
        self.queries = nn.Linear(self.head_dim,self.head_dim,bias=False)#根据注意力权重对输入进行加权聚合的向量表示
        self.fc_out = nn.Linear(self.heads*self.head_dim,embed_size,bias=False) #全连接输出
        #每个位置的输入同样会经过线性变换，得到一个用于加权聚合的value向量。
        #注意力权重表示了query与key的相似度，通过对value向量进行加权聚合，可以得到最终的注意力表示
        #Transformer中的多头注意力机制允许模型学习多个不同的query、key和value表示，以捕捉输入序列不同位置之间的多个关联性和重要性
        
    def forward(self,values,keys,queries,mask):
        N = queries.shape[0]    #样本个数
        value_len,key_len,query_len = values.shape[1],keys.shape[1],queries.shape[1]    #三者的长度与源句子和目标句子相关联

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        #将每个样本重塑为划分给多个注意力头以进行并行计算注意力权重的形状
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum('nqhd,nkhd->nhqk',[queries,keys]) 
        #einsum进行特定规则的矩阵乘法计算能量矩阵,n代表样本容量,q代表query_len,k代表key_len,h代表heads,d代表heads_dim
        #能量矩阵(energy matrix)是自注意力机制中的一个重要概念。它表示了每个查询向量(queries)与每个键向量(keys)之间的相似度或相关性,从而在后续步骤中计算注意力权重
        
        # if mask is not None:
        #     energy = energy.masked_fill(mask == 0, float('-1e20'))
        #能量矩阵的掩码操作，目的是将注意力能量矩阵中的某些位置的能量值置为一个很小的负数（接近负无穷），以起到屏蔽（mask）对应位置的作用
        #mask是一个与能量矩阵相同形状的张量，其中的元素为0或1。当mask中的元素为0时，表示对应位置需要被掩码，即忽略该位置的能量值。
        #通过使用masked_fill函数，将能量矩阵中对应掩码位置的能量值替换为一个较小的负数
            
        attention = torch.softmax( energy / (self.embed_size**(1/2)) , dim=3)
        out = torch.einsum('nhql,nlhd->nqhd',[attention,values])
        #代入注意力权重计算公式,Attention(Q,K,V) = softmax( (Q*K^T) / dk**(1/2) ) * V
        #dim=3代表使用energy的第三个维度即key_len进行softmax归一化
        #attention形状:(N,heads,query_len,key_len),values形状:(N,value_len,heads,head_dim),out形状:(N,query_len,heads,head_dim)

        out = out.reshape(N, query_len, self.heads * self.head_dim)
        #拼接注意力头与注意力头维度
        out = self.fc_out(out)
        #线性变换,得到最终的注意力矩阵
        
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        
        self.attention = SelfAttention(embed_size,heads)    #自注意力模块实例化以调用注意力权重计算的方法
        
        self.norm1 = nn.LayerNorm(embed_size)    #自注意力层后的归一化
        #归一化操作有助于缓解模型训练过程中的梯度消失和梯度爆炸问题，提升模型的稳定性和收敛速度。
        #它还有助于模型的泛化能力和表达能力，减少模型的过拟合风险

        self.feed_forward = nn.Sequential(      #前馈神经网络层
            nn.Linear(embed_size,forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.norm2 = nn.LayerNorm(embed_size)    #前馈神经网络层后的归一化

        self.dropout = nn.Dropout(dropout)  
        #随机失活(dropout)是一种正则化技术通过在训练过程中随机忽略（即设置为0）神经网络的部分节点，可以有效防止过拟合。
        #这种操作可以让模型变得更加健壮，因为它不会过度依赖于任何一个节点
        #输入是一个标量参数dropout，该参数指定了以多大的概率将输入置为零

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask) #Multihead Attention部分
        x = self.dropout( self.norm1(attention + query) )   #第一个Add & Norm部分
        forward = self.feed_forward(x)                      #FeedForward部分
        out = self.dropout( self.norm2(forward + x) )       #第二个Add & Norm 部分
        return out

class Encoder(nn.Module):
    def __init__(
            self, 
            src_vocab_size, #源语言词汇表的大小，它表示源语言中唯一词汇的数量。词汇表大小直接影响了嵌入层的输入维度
            embed_size, #嵌入层的维度，嵌入层用于将离散的词汇转换为连续的、稠密的词向量表示
            num_layers, #编码器中的层数，编码器通常由多个相同结构的层堆叠而成,每个层都会对输入进行一定的变换和特征提取
            heads,      #注意力机制中的头数,注意力机制常用于处理序列数据，在编码器中可以通过并行计算多个注意力头来提取不同的特征
            device,     #所用设备CPU/GPU,可以选择在 CPU 或 GPU 上训练和运行模型
            forward_expansion,  #前向传播时的扩展维度,使用前向传播时的扩展维度来增加模型的表达能力和非线性性质
            dropout,    #随机失活概率,用于正则化,减少过拟合
            max_length  #输入序列的最大长度,在处理序列数据时,为了方便计算和内存管理，通常会限制输入序列的最大长度
        ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.word_embedding = nn.Linear(src_vocab_size, embed_size)  
        #Input Embedding部分,将源句子的每一个词映射为一个固定长度的词向量
        self.position_embedding = nn.Embedding(max_length, embed_size)  
        #Positional Embedding部分,由于多头注意力机制并行处理单词会导致丢失单词的位置信息,故要用位置编码
        #nn.Embedding模块完成词嵌入工作,词嵌入是将离散的词语或标记映射到低维的实数向量表示的技术,可将文本数据表示为计算机能够理解和处理的形式
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        N, seq_length, embed_size = x.shape
        positions = torch.arange(0,seq_length).expand(N, seq_length).to(self.device)    #生成一个形状为(N, seq_length)的位置编码矩阵
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions)) #Input Embedding + Positional Embedding 部分

        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            src_pad_idx,
            embed_size = 256,
            num_layers = 1,    #六层encoder和六层decoder
            forward_expansion = 4,
            heads = 8,
            dropout=0,
            device = 'cuda',
            max_length=100,
            out_dim = 2
        ):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,     
            device,    
            forward_expansion,
            dropout,   
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.device = device
        self.max_length = max_length
        self.fc = nn.Sequential(
            nn.Linear(max_length * embed_size, 2048),
            nn.BatchNorm1d(2048),
            nn.Linear(2048,out_dim)
        )

    def make_src_mask(self, src):
        src_mask = torch.ones(self.max_length).unsqueeze(1).unsqueeze(2)  #(src != src_pad_idx)为(2,9),在1维和2维增加一个大小为1的维度变为(2,1,1,9)
        return src_mask.to(self.device)
    
    def forward(self, src):
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)
        enc_src = enc_src.reshape(enc_src.shape[0], -1)
        output = self.fc(enc_src)
        return output