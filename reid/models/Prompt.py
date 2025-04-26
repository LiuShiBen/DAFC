import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import copy
from collections import OrderedDict
from sklearn.metrics.pairwise import rbf_kernel

class TextPrompt(nn.Module):
    def __init__(self, emb_d=768, n_tasks=1, prompt_param=[128,10,20], key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self.ker_size = 3
        self.stride = 1
        self.dilation = 1
        self.num_heads = 8
        self.pool_size = 20
        self._init_smart(prompt_param)
        self.top_prompt = int(prompt_param[1])

        #self.prompt_G = nn.Sequential(*[nn.Linear(512, emb_d) for _ in range(int(prompt_param[2]))])  #  # 768


        self.prompt_embed_matcher = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(emb_d, emb_d // 2)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(emb_d // 2, emb_d // 4))
        ]))
        self.k_conv_vals = nn.ModuleDict()
        self.v_conv_vals = nn.ModuleDict()

        for i in range(self.num_heads):
            self.k_conv_vals[str(i)] = nn.ModuleList()
            for k in range(self.pool_size):
                conv_val = nn.Conv2d(1, 1, (self.ker_size, self.ker_size),  \
                                     stride=self.stride, dilation=self.dilation)
                # conv_val.weight.data = torch.nn.init.normal_(conv_val.weight.data)
                self.k_conv_vals[str(i)].append(conv_val)

        for j in range(self.num_heads):
            self.v_conv_vals[str(j)] = nn.ModuleList()
            for k in range(self.pool_size):
                conv_val = nn.Conv2d(1, 1, (self.ker_size, self.ker_size), \
                                     stride=self.stride, dilation=self.dilation)
                # conv_val.weight.data = torch.nn.init.normal_(conv_val.weight.data)
                self.v_conv_vals[str(j)].append(conv_val)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)  #20,8,768
            k = tensor_prompt(self.e_pool_size, self.key_d)  #20,768
            a = tensor_prompt(self.e_pool_size, self.key_d)  #20,768
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0]) #20
        self.e_p_length = int(prompt_param[1])  #8
        self.e_layers = [0,1,2,3,4,5,6,7,8,9,10,11]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]  #0.0
        
    '''def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)'''

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu)



    def forward(self, image_query, text_prompt, l, x_block):

        '''
        x_querry: B,768
        K: B
        '''
        #print(x_querry.shape, l, x_block.shape)
        # e prompts
        e_valid = False
        '''text_prompt = []
        for i, prompt_g in enumerate(self.prompt_G):
            x = prompt_g(text_query)
            text_prompt.append(x)'''
        if l in self.e_layers:
            e_valid = True
            B, C = image_query.shape

            K = getattr(self,f'e_k_{l}')  #20,768
            A = getattr(self,f'e_a_{l}')  #20,768
            #p = getattr(self,f'e_p_{l}')  #20,8,768
            #print("image_query:", image_query.shape, text_query.shape, text_image.shape)
            a_querry = torch.einsum('bd,kd->bkd', image_query, A)  # Hadamard product
            #print("....:", K.shape, A.shape, a_querry.shape)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            #print("....:", aq_k.shape, n_K.shape, prompt.shape)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            #print("P_:", aq_k.shape, prompt.shape)
            P_ = torch.einsum('bk,bld->bld', aq_k, text_prompt)
            #print("P_:", aq_k.shape)

            topk_prompt, topk_indices = torch.topk(P_, self.top_prompt, dim=1)
            #print("topk_prompt:", topk_prompt.shape)
            # select prompts
            i = int(self.e_p_length/2)
            Ek = topk_prompt[:,:i,:]  #B,4,768
            Ev = topk_prompt[:,i:,:]
            #print("x:", Ek.shape, Ev.shape)


        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, x_block

    def compute_conv_over_prompt(self, batched_prompt, f, layer_num, similarity):
        #  batch_size, dual, length // dual, self.num_heads, embed_dim // self.num_heads
        batched_prompt = batched_prompt.permute(1, 3, 0, 2, 4)  # dual, num_heads, B, length, head_dim
        k_prompt_list = []
        v_prompt_list = []

        k_prompt_layer = batched_prompt[0]  # num_heads, B,  length, head_dim
        v_prompt_layer = batched_prompt[1]  # num_heads, B,length, head_dim
        n_heads, batch_size, length, head_dim = k_prompt_layer.shape
        # print("K prompt layer shape: ", k_prompt_layer.shape)
        length = length - self.ker_size + 1
        head_dim = head_dim - self.ker_size + 1
        new_k_prompt_layer = torch.zeros((n_heads, batch_size, length, head_dim), device=k_prompt_layer.device)
        new_v_prompt_layer = torch.zeros((n_heads, batch_size, length, head_dim), device=k_prompt_layer.device)
        for h in range(self.num_heads):
            k_conv_vals = self.k_conv_vals[str(h)]
            v_conv_vals = self.v_conv_vals[str(h)]
            k_prompt_head = k_prompt_layer[h].unsqueeze(1)  # B, 1, length, head_dim
            v_prompt_head = v_prompt_layer[h].unsqueeze(1)  # B, 1, length, head_dim
            for p in range(f):
                k_conv_val = k_conv_vals[p]
                v_conv_val = v_conv_vals[p]
                new_k_prompt_layer[h] += k_conv_val(k_prompt_head).squeeze(1) * similarity[:, p].unsqueeze(1).unsqueeze(
                    2)
                new_v_prompt_layer[h] += v_conv_val(v_prompt_head).squeeze(1) * similarity[:, p].unsqueeze(1).unsqueeze(
                    2)
            # k_prompt_list.append(new_k_prompt_layer)  # num_layers, num_heads, B,length, head_dim
            # v_prompt_list.append(new_v_prompt_layer)
        # new_k_prompt = torch.stack(k_prompt_list, dim=0) # num_layers, num_heads, B,length, head_dim
        # new_v_prompt = torch.stack(v_prompt_list, dim=0) # num_layers, num_heads, B,length, head_dim
        new_batched_prompt = torch.stack([new_k_prompt_layer, new_v_prompt_layer],
                                         dim=0)  # dual, num_heads, B, length, head_dim
        new_batched_prompt = new_batched_prompt.permute(2, 0, 3, 1, 4)  # B, dual, length, num_heads, head_dim

        return new_batched_prompt

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }

# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p    

'''class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None):
        super(ViTZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None

        # get feature encoder
        if pt:
            zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                        num_heads=12, ckpt_layer=0,
                                        drop_path_rate=0
                                        )
            from timm.models import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict)

        # classifier
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None
        
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
        
    # pen: get penultimate features    
    def forward(self, x, pen=False, train=False):

        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(x)
                q = q[:,0,:]
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:,0,:]
        else:
            out, _ = self.feat(x)
            out = out[:,0,:]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out
            
def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param)'''


