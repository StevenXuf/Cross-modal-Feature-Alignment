import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,
            in_size=768,
            hidden_sizes=[512,512],
            out_size=1,
            init='kaiming'):

        super(MLP,self).__init__()
        self.in_size=in_size*2
        self.hidden_sizes=hidden_sizes
        self.out_size=out_size
        self.n_hidden_layers=len(hidden_sizes)
        self.init=init

        self.layers=[]
        
        for i in range(self.n_hidden_layers+1):
            name='linear'+str(i)
            if i<self.n_hidden_layers:
                self.__setattr__(name,nn.Sequential(
                    nn.Linear(self.in_size,self.hidden_sizes[i]),
                    nn.ReLU()))
                self.in_size=self.hidden_sizes[i]
            else:
                self.in_size=self.hidden_sizes[-1]
                self.__setattr__(name,nn.Sequential(
                    nn.Linear(self.in_size,self.out_size),
                    nn.Tanh()))

            self.layers.append(self.__getattr__(name))

        if self.init=='xavier':
            for module in self.modules():
                if isinstance(module,nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
        elif self.init=='kaiming':
            for module in self.modules():
                if isinstance(module,nn.Linear):
                    nn.init.kaiming_uniform_(module.weight)

    def forward(self,x,img_feat=None):
        # Accept either concatenated features or the two inputs used by train_main.
        if img_feat is not None:
            x=torch.cat((x,img_feat),dim=1)
        for layer in self.layers:
            x=layer(x)
        return x
    

class NewMLP(nn.Module):
    def __init__(self,
            in_size=768,
            hidden_sizes=[512,512],
            out_size=1,
            dropout=0.1,
            init='kaiming'):

        super(NewMLP,self).__init__()
        self.in_size=in_size*2
        self.hidden_sizes=hidden_sizes
        self.out_size=out_size
        self.n_hidden_layers=len(hidden_sizes)
        self.init=init
        self.dropout=dropout

        self.layers=[]
        
        self.layers = nn.ModuleList()

        for i in range(self.n_hidden_layers + 1):
            if i < self.n_hidden_layers:
                self.layers.append(nn.Sequential(
                    nn.Linear(self.in_size, self.hidden_sizes[i]),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                ))
                self.in_size = self.hidden_sizes[i]
            else:
                self.layers.append(nn.Sequential(
                    nn.Linear(self.in_size, self.out_size),
                    nn.Tanh()
                ))

        if self.init=='xavier':
            for module in self.modules():
                if isinstance(module,nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
        elif self.init=='kaiming':
            for module in self.modules():
                if isinstance(module,nn.Linear):
                    nn.init.kaiming_uniform_(module.weight)

    def forward(self,txt_feat,img_feat):
        x=torch.cat((txt_feat,img_feat),dim=1)
        for layer in self.layers:
            x=layer(x)
        return x
    
class CrossAttentionSimilarity(nn.Module):
    def __init__(
        self,
        in_size=512,
        hidden_dim=128,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()

        self.image_proj = nn.Linear(in_size, hidden_dim)
        self.text_proj = nn.Linear(in_size, hidden_dim)

        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),   # output range [-1, 1], like cosine similarity
        )

    def forward(self, image_feat, text_feat):
        """
        image_feat: [B, in_size]
        text_feat:  [B, in_size]

        returns:
            predicted cosine similarity: [B]
        """

        img = self.image_proj(image_feat).unsqueeze(1)  # [B, 1, D]
        txt = self.text_proj(text_feat).unsqueeze(1)    # [B, 1, D]

        for layer in self.layers:
            img, txt = layer(img, txt)

        img = img.squeeze(1)  # [B, D]
        txt = txt.squeeze(1)  # [B, D]

        joint = torch.cat([
            img,
            txt,
            # img * txt,
            # torch.abs(img - txt),
        ], dim=-1)

        sim = self.regressor(joint).squeeze(-1)
        return sim


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()

        self.img_to_txt_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.txt_to_img_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.img_norm1 = nn.LayerNorm(dim)
        self.txt_norm1 = nn.LayerNorm(dim)

        self.img_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

        self.txt_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

        self.img_norm2 = nn.LayerNorm(dim)
        self.txt_norm2 = nn.LayerNorm(dim)

    def forward(self, img, txt):
        """
        img: [B, 1, D]
        txt: [B, 1, D]
        """

        img_attn, _ = self.img_to_txt_attn(
            query=img,
            key=txt,
            value=txt,
        )

        txt_attn, _ = self.txt_to_img_attn(
            query=txt,
            key=img,
            value=img,
        )

        img = self.img_norm1(img + img_attn)
        txt = self.txt_norm1(txt + txt_attn)

        img = self.img_norm2(img + self.img_ffn(img))
        txt = self.txt_norm2(txt + self.txt_ffn(txt))

        return img, txt


if __name__=='__main__':
    in_size = 512
    new_mlp=NewMLP(in_size=in_size,hidden_sizes=[512,256])
    transformer=CrossAttentionSimilarity(in_size=in_size)
    # print(new_mlp)
    # print(transformer)

    for model in [new_mlp, transformer]:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {num_params:,}")