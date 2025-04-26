import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1): #nn.GELU
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 定义专家网络
class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_features):
        super(Expert, self).__init__()
        self.Mlp = Mlp(input_size, hidden_features, output_size)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x) + self.Mlp(x)


# 定义门控网络
class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.linear = nn.Linear(input_size, num_experts)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        #print("x:::", x.shape)
        x = self.linear(x)
        x = F.softmax(x, dim=-1)
        #print("x:::", x.shape)
        return x

class TWS(nn.Module): #Task-wise supervisor
    def __init__(self, input_size=768, output_size=768, hidden_features = 768, num_experts=100):
        super(TWS, self).__init__()
        self.experts = nn.ModuleList([Expert(input_size, output_size, hidden_features) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_size, num_experts)

    def forward(self, x):
        #gate_output = self.gate(x)
        #print("gate_output", gate_output.shape)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        #print("expert_outputs", expert_outputs.shape)
        #output = torch.sum(gate_output.unsqueeze(-2) * expert_outputs, dim=-1)
        return expert_outputs

# 定义MoE模型
'''class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_features):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([Expert(input_size, output_size, hidden_features) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_size, num_experts)  #128,10
        self.frozen_experts = set()


    def forward(self, x, top_k):
        #print("x...:", x.shape, top_k)
        # 获取门控网络输出
        gate_outputs = self.gating_network(x)

        # 获取Top-k专家网络索引
        topk_values, topk_indices = torch.topk(gate_outputs, top_k, dim=-1)
        #print("topk_indices:", topk_indices)
        # 将冻结的专家索引添加到当前任务的专家中
        combined_indices = torch.tensor(list(self.frozen_experts.union(set(topk_indices.cpu().tolist()[0]))),
                                        device=topk_indices.device)

        expert_outputs = torch.stack([self.experts[idx](x) for idx in combined_indices], dim=1)

        # 仅使用选中的门控权重
        combined_gate_outputs = gate_outputs[:, combined_indices]
        #print("combined_gate_outputs", combined_gate_outputs.shape)

        output = torch.sum(combined_gate_outputs.unsqueeze(2) * expert_outputs, dim=1)

        # 冻结当前任务的Top-k专家
        self.frozen_experts.update(topk_indices.cpu().tolist()[0])

        return output'''