import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from modules import DynamicFilterGNN

from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union


class DFGNN(nn.Module):
    def __init__(self, K, A, feature_size, Clamp_A=True):
        super(DFGNN, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        self.K = K
        self.A_list = []

        D_inverse = torch.diag(1 / (torch.sum(A, 0) + 1e-8))
        norm_A = torch.matmul(D_inverse, A)
        A = norm_A

        A_temp = torch.eye(feature_size, feature_size)
        for i in range(K):
            A_temp = torch.matmul(A_temp, A)
            if Clamp_A:
                A_temp = torch.clamp(A_temp, max=1.)
            self.A_list.append(A_temp)

        self.gc_list = nn.ModuleList([DynamicFilterGNN(feature_size, feature_size, self.A_list[i], bias=False) for i in range(K)])
        hidden_size = self.feature_size
        gc_input_size = self.feature_size * K

        self.fl = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.il = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(gc_input_size + hidden_size, hidden_size)


        self.Neighbor_weight = Parameter(torch.FloatTensor(feature_size))
        stdv = 1. / math.sqrt(feature_size)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)


        input_size = self.feature_size

        self.rfl = nn.Linear(input_size + hidden_size, hidden_size)
        self.ril = nn.Linear(input_size + hidden_size, hidden_size)
        self.rol = nn.Linear(input_size + hidden_size, hidden_size)
        self.rCl = nn.Linear(input_size + hidden_size, hidden_size)

        # addtional vars
        self.c = torch.nn.Parameter(torch.Tensor([1]))

        self.fc1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 64)
        self.fc5 = nn.Linear(64, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, 64)
        
    def forward(self, input, Hidden_State=None, Cell_State=None, rHidden_State=None, rCell_State=None):
        batch_size, time_steps, _ = input.shape
        if Hidden_State is None:
            Hidden_State = Variable(torch.zeros(batch_size,self.feature_size).cuda())
        if Cell_State is None:
            Cell_State = Variable(torch.zeros(batch_size,self.feature_size).cuda())
        if rHidden_State is None:
            rHidden_State = Variable(torch.zeros(batch_size,self.feature_size).cuda())
        if rCell_State is None:
            rCell_State = Variable(torch.zeros(batch_size,self.feature_size).cuda())

        Hidden_State = Hidden_State.unsqueeze(1).expand(-1, time_steps, -1)
        Cell_State = Cell_State.unsqueeze(1).expand(-1, time_steps, -1)
        rHidden_State = rHidden_State.unsqueeze(1).expand(-1, time_steps, -1)
        rCell_State = rCell_State.unsqueeze(1).expand(-1, time_steps, -1)
            
        x = input
        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)

        combined = torch.cat((gc, Hidden_State), 1)
        dim1=combined.shape[0]
        dim2=combined.shape[1]
        dim3=combined.shape[2]
        combined=combined.view(dim1,dim2//4,dim3*4)

        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))

        NC = torch.mul(Cell_State,
                       torch.mv(Variable(self.A_list[-1], requires_grad=False).cuda(), self.Neighbor_weight))
        Cell_State = f * NC + i * C
        Hidden_State = o * torch.tanh(Cell_State)

        # LSTM
        rcombined = torch.cat((input, rHidden_State), 1)
        d1=rcombined.shape[0]
        d2=rcombined.shape[1]
        d3=rcombined.shape[2]
        rcombined=rcombined.view(d1,d2//2,d3*2)
        rf = torch.sigmoid(self.rfl(rcombined))
        ri = torch.sigmoid(self.ril(rcombined))
        ro = torch.sigmoid(self.rol(rcombined))
        rC = torch.tanh(self.rCl(rcombined))
        rCell_State = rf * rCell_State + ri * rC
        rHidden_State = ro * torch.tanh(rCell_State)

        # Kalman Filtering
        var1, var2 = torch.var(input), torch.var(gc)

        pred = (Hidden_State * var1 * self.c + rHidden_State * var2) / (var1 + var2 * self.c)

        return pred
        #return Hidden_State, Cell_State, gc, rHidden_State, rCell_State, pred

    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a

    def loop(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State, rHidden_State, rCell_State = self.initHidden(batch_size)
        for i in range(time_step):
            Hidden_State, Cell_State, gc, rHidden_State, rCell_State, pred = self.forward(
                torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State, rHidden_State, rCell_State)
        return pred


    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            rHidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            rCell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State, rHidden_State, rCell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            rHidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            rCell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State, rHidden_State, rCell_State

    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            Cell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            rHidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            rCell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State, rHidden_State, rCell_State
        else:
            Hidden_State = Variable(Hidden_State_data, requires_grad=True)
            Cell_State = Variable(Cell_State_data, requires_grad=True)
            rHidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            rCell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State, rHidden_State, rCell_State

# Mamba Network
@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    features: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    K: int = 3
    A: torch.Tensor = None
    feature_size: int = None
    use_hypergraph:bool = True
    lr: float = None
    knn_k: float=None

    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class MVSTH(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dfgnn = DFGNN(K=args.K, A=args.A, feature_size=args.feature_size)
        self.encode = nn.Linear(args.features, args.d_model)
        self.encoder_norm = RMSNorm(args.d_model)

        self.encoder_layers = nn.ModuleList([ResidualBlock(args,self.dfgnn) for _ in range(args.n_layer)])

        # Decoder (identical to Encoder)
        ##self.decoder_layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)]) #You can optionally uncommand these lines to use the identical Decoder.
        ##self.decoder_norm = RMSNorm(args.d_model) #You can optionally uncommand these lines to use the identical Decoder.
        self.decode = nn.Linear(args.d_model, args.features)

    def forward(self, x):
        x = self.encode(x)

        for layer in self.encoder_layers:
            # residual = x
            x = layer(x)
            # x = x + residual

        x = self.encoder_norm(x)
        x = self.decode(x)
        
        return x

        

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs, dfgnn: DFGNN):
        super().__init__()
        self.args = args
        self.dfgnn = DFGNN(K=args.K, A=args.A, feature_size=args.feature_size)       
        self.mixer = MambaBlock(args,dfgnn)
        self.norm = RMSNorm(args.d_model)

        if self.args.use_hypergraph:
            self.hyper_processor = HypergraphProcessor(
                feature_size=args.feature_size,
                num_clusters=int(args.feature_size**args.knn_k)
            )
            self.fusion_gate = nn.Linear(args.d_model*2, args.d_model)

    def forward(self, x):
        x0 = x
        x1 = self.norm(x)
        
        dfgnn_out = self.dfgnn(x1)
        
        if self.args.use_hypergraph:
            dynamic_adj = self.dfgnn.gc_list[-1].get_transformed_adjacency()
            
            hyper_feat = self.hyper_processor(
                dynamic_adj.unsqueeze(0),  
                x1.unsqueeze(1)            
            ).squeeze(1)
            
            fused = torch.cat([dfgnn_out, hyper_feat], dim=-1)
            fused = self.fusion_gate(fused)
            dfgnn_out = fused + dfgnn_out
            
        x2 = dfgnn_out
        x3 = self.mixer(x2)
        
        return x3 + x0


class MambaBlock(nn.Module):    
    def __init__(self, args: ModelArgs, dfgnn: DFGNN):
        super().__init__()
        self.args = args
        self.dfgnn = dfgnn

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        

    def forward(self, x):
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
    
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)
        
        return y
    
    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        temp_adj = self.dfgnn.gc_list[-1].get_transformed_adjacency()        
        temp_adj_padded = torch.ones(d_in, d_in, device=temp_adj.device)       
        temp_adj_padded[:temp_adj.size(0), :temp_adj.size(1)] = temp_adj
        
        delta_p = torch.matmul(delta,temp_adj_padded)
        
        deltaA = torch.exp(einsum(delta_p, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta_p, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        
        y = y + u * D
    
        return y

    
class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output



class HypergraphProcessor(nn.Module):
    def __init__(self, feature_size, num_clusters, temperature=0.1):
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        
        self.cluster_centers = Parameter(torch.Tensor(num_clusters, feature_size))
        nn.init.xavier_uniform_(self.cluster_centers)
        
        self.hgcn = HyperGCN(feature_size)

        self.distance_sensitivity = nn.Parameter(torch.tensor(1.0))
        self.norm_factor = nn.Parameter(torch.tensor(1.0))

        self.feature_transform = torch.nn.Linear(12, 400)
        self.feature_invertransform = torch.nn.Linear(400, 12)

    def forward(self, adj_matrix, node_features):
        B = node_features.shape[0]
        adj_matrix = adj_matrix.expand(B, -1, -1)

        node_features = node_features.squeeze(1)
        node_features = node_features.permute(0, 2, 1)
        node_features = self.feature_transform(node_features)
        D= node_features.shape[2]
        
        adj_weights = adj_matrix.mean(dim=1)  # (B, N)

        node_vectors = torch.einsum('bn,bnd->bnd', adj_weights, node_features)
        
        similarity = F.cosine_similarity(
            node_vectors.unsqueeze(2), 
            self.cluster_centers.view(1,1,self.num_clusters,D), 
            dim=-1
        ) / self.temperature

        cluster_assign = F.softmax(similarity, dim=-1)  # (B, N, K)
        
        dist = 1 - similarity
        H = torch.exp(-self.distance_sensitivity * dist)
        
        D_v = torch.sum(H, dim=2, keepdim=True)
        D_e = torch.sum(H, dim=1, keepdim=True)
        H_norm = H / (torch.sqrt(D_v) * torch.sqrt(D_e) + 1e-8)
        
        hyper_features = self.hgcn(node_features, H_norm)
        
        return self.feature_invertransform(hyper_features).permute(0, 2, 1)

class HyperGCN(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.mlp_edge = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
            nn.LayerNorm(feature_size)
        )
        self.mlp_node = nn.Linear(feature_size, feature_size)
        
    def forward(self, X, H):
        E = torch.einsum('bnk,bnd->bkd', H, X)  # (B, K, D)
        E = self.mlp_edge(E)
        
        X_new = torch.einsum('bnk,bkd->bnd', H, E)
        X_new = self.mlp_node(X_new)
        
        return F.relu(X_new + X)