import math
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn.functional
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform

def block_expansion(ckpt, split, original_layers):

    layer_cnt = 0
    selected_layers = []
    output = {}

    for i in range(original_layers):
        for k in ckpt:
            if ('layer.' + str(i) + '.') in k:
                output[k.replace(('layer.' + str(i) + '.'), ('layer.' + str(layer_cnt) + '.'))] = ckpt[k]
        layer_cnt += 1
        if (i+1) % split == 0:
            for k in ckpt:
                if ('layer.' + str(i) + '.') in k:
                    if 'attention.output' in k or str(i)+'.output' in k:
                        output[k.replace(('layer.' + str(i) + '.'), ('layer.' + str(layer_cnt) + '.'))] = torch.zeros_like(ckpt[k])
                        selected_layers.append(layer_cnt)
                    else:
                        output[k.replace(('layer.' + str(i) + '.'), ('layer.' + str(layer_cnt) + '.'))] = ckpt[k]
            layer_cnt += 1

    for k in ckpt:
        if not 'layer' in k:
            output[k] = ckpt[k]
        elif k == "vit.layernorm.weight" or k == "vit.layernorm.bias" or k == "dinov2.layernorm.bias" or k == "dinov2.layernorm.weight":
            output[k] = ckpt[k]
    
    selected_layers = list(set(selected_layers))

    return output, selected_layers


class RBF(torch.nn.Module):
  def __init__(self, sigma=None):
    super(RBF, self).__init__()

    self.sigma = sigma

  def forward(self, X): # X.shape = [n_particle, n_dim_of_theta]

    if X.shape[0] == 0:
        return 0

    distances = torch.cdist(X, X, p=2)
    dnorm2 = distances ** 2

    # Apply the median heuristic (PyTorch does not give true median)
    if self.sigma is None:
      np_dnorm2 = dnorm2.detach().cpu().numpy()
      h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
      sigma = np.sqrt(h).item()
    else:
      sigma = self.sigma

    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    K_XY = (-gamma * dnorm2).exp()
        
    return K_XY

class SVGD(torch.optim.Adam):
    def __init__(self, param, rho, sigma, lr, betas, weight_decay, num_particles, train_module, net):
        super(SVGD, self).__init__(param, lr, betas, weight_decay)
        self.K = RBF(sigma)
        self.net = net
        self.num_particles = num_particles
        self.lr = lr
        self.rho = rho
        self.train_module = train_module
        self.sigma = sigma
        
    def get_learnable_block(self): #for LoRA        
        q_A = torch.empty(0).cuda()
        q_B = torch.empty(0).cuda()
        v_A = torch.empty(0).cuda()
        v_B = torch.empty(0).cuda()
        cls_w = torch.empty(0).cuda()
        cls_b = torch.empty(0).cuda()
        tq_A = torch.empty(0).cuda()
        tq_B = torch.empty(0).cuda()
        tv_A = torch.empty(0).cuda()
        tv_B = torch.empty(0).cuda()
        tcls_w = torch.empty(0).cuda()
        tcls_b = torch.empty(0).cuda()
        i_qA = 0
        i_qB = 0
        i_vA = 0
        i_vB = 0
        i_cls_w = 0
        i_cls_b = 0
        for n, p in self.net.named_parameters():
            if p.requires_grad:
                if "proj_q" in n:
                    if "w_a" in n:
                        if i_qA < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tq_A = torch.cat((tq_A, p_), dim=1)
                            i_qA += 1
                            if i_qA == self.num_particles:   
                                q_A = torch.cat((q_A, tq_A), dim=0)
                                tq_A = torch.empty(0).cuda()
                                i_qA = 0
                    elif "w_b" in n:
                        if i_qB < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tq_B = torch.cat((tq_B, p_), dim=1)
                            i_qB += 1
                            if i_qB == self.num_particles:   
                                q_B = torch.cat((q_B, tq_B), dim=0)
                                tq_B = torch.empty(0).cuda()
                                i_qB = 0
                elif "proj_v" in n:
                    if "w_a" in n:
                        if i_vA < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tv_A = torch.cat((tv_A, p_), dim=1)
                            i_vA += 1
                            if i_vA == self.num_particles:   
                                v_A = torch.cat((v_A, tv_A), dim=0)
                                tv_A = torch.empty(0).cuda()
                                i_vA = 0
                    elif "w_b" in n:
                        if i_vB < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tv_B = torch.cat((tv_B, p_), dim=1)
                            i_vB += 1
                            if i_vB == self.num_particles:   
                                v_B = torch.cat((v_B, tv_B), dim=0)
                                tv_B = torch.empty(0).cuda()
                                i_vB = 0                 
                elif "fc" in n:
                    if "weight" in n:
                        if i_cls_w < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tcls_w = torch.cat((tcls_w, p_), dim=1)
                            i_cls_w += 1
                            if i_cls_w == self.num_particles:   
                                cls_w = torch.cat((cls_w, tcls_w), dim=0)
                                tcls_w = torch.empty(0).cuda()
                                i_cls_w = 0
                    elif "bias" in n:
                        if i_cls_b < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tcls_b = torch.cat((tcls_b, p_), dim=1)
                            i_cls_b += 1
                            if i_cls_b == self.num_particles:   
                                cls_b = torch.cat((cls_b, tcls_b), dim=0)
                                tcls_b = torch.empty(0).cuda()
                                i_cls_b = 0
                            
        return q_A, q_B, v_A, v_B, cls_w, cls_b
    
    def get_grad1(self): #for LoRA
        
        q_A = torch.empty(0).cuda()
        q_B = torch.empty(0).cuda()
        v_A = torch.empty(0).cuda()
        v_B = torch.empty(0).cuda()
        cls_w = torch.empty(0).cuda()
        cls_b = torch.empty(0).cuda()
        tq_A = torch.empty(0).cuda()
        tq_B = torch.empty(0).cuda()
        tv_A = torch.empty(0).cuda()
        tv_B = torch.empty(0).cuda()
        tcls_w = torch.empty(0).cuda()
        tcls_b = torch.empty(0).cuda()
        i_qA = 0
        i_qB = 0
        i_vA = 0
        i_vB = 0
        i_cls_w = 0
        i_cls_b = 0
        for n, p in self.net.named_parameters():
            if p.requires_grad:
                if "proj_q" in n:
                    if "w_a" in n:
                        if i_qA < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tq_A = torch.cat((tq_A, p_), dim=1)
                            i_qA += 1
                            if i_qA == self.num_particles:   
                                q_A = torch.cat((q_A, tq_A), dim=0)
                                tq_A = torch.empty(0).cuda()
                                i_qA = 0
                    elif "w_b" in n:
                        if i_qB < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tq_B = torch.cat((tq_B, p_), dim=1)
                            i_qB += 1
                            if i_qB == self.num_particles:   
                                q_B = torch.cat((q_B, tq_B), dim=0)
                                tq_B = torch.empty(0).cuda()
                                i_qB = 0
                elif "proj_v" in n:
                    if "w_a" in n:
                        if i_vA < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tv_A = torch.cat((tv_A, p_), dim=1)
                            i_vA += 1
                            if i_vA == self.num_particles:   
                                v_A = torch.cat((v_A, tv_A), dim=0)
                                tv_A = torch.empty(0).cuda()
                                i_vA = 0
                    elif "w_b" in n:
                        if i_vB < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tv_B = torch.cat((tv_B, p_), dim=1)
                            i_vB += 1
                            if i_vB == self.num_particles:   
                                v_B = torch.cat((v_B, tv_B), dim=0)
                                tv_B = torch.empty(0).cuda()
                                i_vB = 0
                                
                elif 'fc' in n:
                    if 'weight' in n:
                        if i_cls_w < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tcls_w = torch.cat((tcls_w, p_), dim=1)
                            i_cls_w += 1
                            if i_cls_w == self.num_particles:   
                                cls_w = torch.cat((cls_w, tcls_w), dim=0)
                                tcls_w = torch.empty(0).cuda()
                                i_cls_w = 0
                    elif 'bias' in n:
                        if i_cls_b < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tcls_b = torch.cat((tcls_b, p_), dim=1)
                            i_cls_b += 1
                            if i_cls_b == self.num_particles:   
                                cls_b = torch.cat((cls_b, tcls_b), dim=0)
                                tcls_b = torch.empty(0).cuda()
                                i_cls_b = 0
                    
        return q_A, q_B, v_A, v_B, cls_w, cls_b
    
    def kernel_func(self, q_A, q_B, v_A, v_B, clsW, clsB):

        q_A.requires_grad = True
        q_B.requires_grad = True
        v_A.requires_grad = True
        v_B.requires_grad = True
        clsW.requires_grad = True
        clsB.requires_grad = True
        
        kernel_qA = self.K(q_A)
        self.train_module.manual_backward(kernel_qA.sum())
        q_A_grad = q_A.grad
        
        kernel_qB = self.K(q_B)
        self.train_module.manual_backward(kernel_qB.sum())
        q_B_grad = q_B.grad
        
        kernel_vA = self.K(v_A)
        self.train_module.manual_backward(kernel_vA.sum())
        v_A_grad = v_A.grad
        
        kernel_vB = self.K(v_B)
        self.train_module.manual_backward(kernel_vB.sum())
        v_B_grad = v_B.grad
        
        kernel_clsW = self.K(clsW)
        self.train_module.manual_backward(kernel_clsW.sum())
        clsW_gradK = clsW.grad
        
        kernel_clsB = self.K(clsB)
        self.train_module.manual_backward(kernel_clsB.sum())
        clsB_gradK = clsB.grad
        
        return kernel_qA, kernel_qB, kernel_vA, kernel_vB, kernel_clsW, kernel_clsB, q_A_grad, q_B_grad, v_A_grad, v_B_grad, clsW_gradK, clsB_gradK
    
    def score_func(self):
        q_A_grad, q_B_grad, v_A_grad, v_B_grad, clsW_grad, clsB_grad = self.get_grad1() 
        
        self.zero_grad()
        q_A, q_B, v_A, v_B, clsW, clsB = self.get_learnable_block()
        q_A, q_B, v_A, v_B, clsW, clsB = q_A.clone().detach().requires_grad_(True), q_B.clone().detach().requires_grad_(True), v_A.clone().detach().requires_grad_(True), v_B.clone().detach().requires_grad_(True),  clsW.clone().detach().requires_grad_(True), clsB.clone().detach().requires_grad_(True)
        
        if q_A.shape[0] > 0:
            kernel_qA, kernel_qB, kernel_vA, kernel_vB, kernel_clsW, kernel_clsB, q_A_gradK, q_B_gradK, v_A_gradK, v_B_gradK, clsW_gradK, clsB_gradK = self.kernel_func(q_A, q_B, v_A, v_B, clsW, clsB) #self.K(self.X, self.X.detach())
        else:
            kernel_qA, kernel_qB, kernel_vA, kernel_vB, kernel_clsW, kernel_clsB, q_A_gradK, q_B_gradK, v_A_gradK, v_B_gradK, clsW_gradK, clsB_gradK = torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), 0, 0, 0, 0, 0, 0
        
        grad_qA = (-kernel_qA.detach().matmul(q_A_grad) + q_A_gradK) / self.num_particles
        
        grad_qB = (-kernel_qB.detach().matmul(q_B_grad) + q_B_gradK) / self.num_particles

        grad_vA = (-kernel_vA.detach().matmul(v_A_grad) + v_A_gradK) / self.num_particles

        grad_vB = (-kernel_vB.detach().matmul(v_B_grad) + v_B_gradK) / self.num_particles
        
        grad_clsW = (-kernel_clsW.detach().matmul(clsW_grad) + clsW_gradK) / self.num_particles
        
        grad_clsB = (-kernel_clsB.detach().matmul(clsB_grad) + clsB_gradK) / self.num_particles
        
        return grad_qA, grad_qB, grad_vA, grad_vB, grad_clsW, grad_clsB

    def step_(self):
        
        q_A_grad, q_B_grad, v_A_grad, v_B_grad, cls_grad_w, cls_grad_b = self.score_func()
        
        updated_n = []
        
        for net_id in range(self.num_particles):
            for layer_id in range(12):   
                for n, p in self.net.lora_vit.named_parameters():
                    
                    if p.requires_grad and n not in updated_n: 
                    
                        if f'blocks.{str(layer_id)}' in n:
                            if "proj_q" in n:
                                if f"w_a.layer.{net_id}" in n:
                                    updated_n.append(n)
                                    temp_w = p.data
                                    p.data = temp_w + self.lr * q_A_grad[layer_id][net_id].view(p.data.shape)
                                elif f"w_b.layer.{net_id}" in n:
                                    updated_n.append(n)
                                    temp_w = p.data
                                    p.data = temp_w + self.lr * q_B_grad[layer_id][net_id].view(p.data.shape)
                            elif "proj_v" in n:
                                if f"w_a.layer.{net_id}" in n:
                                    updated_n.append(n)
                                    temp_w = p.data
                                    p.data = temp_w + self.lr * v_A_grad[layer_id][net_id].view(p.data.shape)
                                elif f"w_b.layer.{net_id}" in n:
                                    updated_n.append(n)
                                    temp_w = p.data
                                    p.data = temp_w + self.lr * v_B_grad[layer_id][net_id].view(p.data.shape)
                                    
                        elif 'fc' in n:
                            if 'weight' in n:
                                if f"layer.{net_id}" in n:
                                    updated_n.append(n)
                                    temp_w = p.data
                                    p.data = temp_w + self.lr * cls_grad_w[layer_id][net_id].view(p.data.shape)
                            elif 'bias' in n and f"layer.{net_id}" in n:
                                    updated_n.append(n)
                                    temp_w = p.data
                                    p.data = temp_w + self.lr * cls_grad_b[layer_id][net_id].view(p.data.shape)
        
class FHBI(torch.optim.Adam):
    def __init__(self, param, rho, sigma, lr, betas, weight_decay, num_particles, train_module, net, base_optimizer_name, decoupled_weight_decay: bool = True):
        super(FHBI, self).__init__(param, lr, betas, weight_decay)
        self.K = RBF(sigma)
        self.net = net
        self.num_particles = num_particles
        self.lr = lr
        self.rho = rho
        self.weight_decay = weight_decay
        self.betas = betas
        self.train_module = train_module
        self.sigma = sigma
        self.base_optimizer_name = base_optimizer_name
        self.decoupled_weight_decay = decoupled_weight_decay 
        
    def get_learnable_block(self): #for LoRA        
        q_A = torch.empty(0).cuda()
        q_B = torch.empty(0).cuda()
        v_A = torch.empty(0).cuda()
        v_B = torch.empty(0).cuda()
        cls_w = torch.empty(0).cuda()
        cls_b = torch.empty(0).cuda()
        tq_A = torch.empty(0).cuda()
        tq_B = torch.empty(0).cuda()
        tv_A = torch.empty(0).cuda()
        tv_B = torch.empty(0).cuda()
        tcls_w = torch.empty(0).cuda()
        tcls_b = torch.empty(0).cuda()
        i_qA = 0
        i_qB = 0
        i_vA = 0
        i_vB = 0
        i_cls_w = 0
        i_cls_b = 0

        for n, p in self.net.named_parameters():
            if p.requires_grad:
                if "proj_q" in n:
                    if "w_a" in n:
                        if i_qA < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tq_A = torch.cat((tq_A, p_), dim=1)
                            i_qA += 1
                            if i_qA == self.num_particles:   
                                q_A = torch.cat((q_A, tq_A), dim=0)
                                tq_A = torch.empty(0).cuda()
                                i_qA = 0
                    elif "w_b" in n:
                        if i_qB < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tq_B = torch.cat((tq_B, p_), dim=1)
                            i_qB += 1
                            if i_qB == self.num_particles:   
                                q_B = torch.cat((q_B, tq_B), dim=0)
                                tq_B = torch.empty(0).cuda()
                                i_qB = 0
                elif "proj_v" in n:
                    if "w_a" in n:
                        if i_vA < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tv_A = torch.cat((tv_A, p_), dim=1)
                            i_vA += 1
                            if i_vA == self.num_particles:   
                                v_A = torch.cat((v_A, tv_A), dim=0)
                                tv_A = torch.empty(0).cuda()
                                i_vA = 0
                    elif "w_b" in n:
                        if i_vB < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tv_B = torch.cat((tv_B, p_), dim=1)
                            i_vB += 1
                            if i_vB == self.num_particles:   
                                v_B = torch.cat((v_B, tv_B), dim=0)
                                tv_B = torch.empty(0).cuda()
                                i_vB = 0                 
                elif "fc" in n:
                    if "weight" in n:
                        if i_cls_w < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tcls_w = torch.cat((tcls_w, p_), dim=1)
                            i_cls_w += 1
                            if i_cls_w == self.num_particles:   
                                cls_w = torch.cat((cls_w, tcls_w), dim=0)
                                tcls_w = torch.empty(0).cuda()
                                i_cls_w = 0
                    elif "bias" in n:
                        if i_cls_b < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tcls_b = torch.cat((tcls_b, p_), dim=1)
                            i_cls_b += 1
                            if i_cls_b == self.num_particles:   
                                cls_b = torch.cat((cls_b, tcls_b), dim=0)
                                tcls_b = torch.empty(0).cuda()
                                i_cls_b = 0
        params = [q_A, q_B, v_A, v_B, cls_w, cls_b]
        return q_A, q_B, v_A, v_B, cls_w, cls_b
    
    def get_grad1(self): #for LoRA
        q_A = torch.empty(0).cuda()
        q_B = torch.empty(0).cuda()
        v_A = torch.empty(0).cuda()
        v_B = torch.empty(0).cuda()
        cls_w = torch.empty(0).cuda()
        cls_b = torch.empty(0).cuda()
        tq_A = torch.empty(0).cuda()
        tq_B = torch.empty(0).cuda()
        tv_A = torch.empty(0).cuda()
        tv_B = torch.empty(0).cuda()
        tcls_w = torch.empty(0).cuda()
        tcls_b = torch.empty(0).cuda()
        i_qA = 0
        i_qB = 0
        i_vA = 0
        i_vB = 0
        i_cls_w = 0
        i_cls_b = 0
        for n, p in self.net.named_parameters():
            if p.requires_grad:
                if "proj_q" in n:
                    if "w_a" in n:
                        if i_qA < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tq_A = torch.cat((tq_A, p_), dim=1)
                            i_qA += 1
                            if i_qA == self.num_particles:   
                                q_A = torch.cat((q_A, tq_A), dim=0)
                                tq_A = torch.empty(0).cuda()
                                i_qA = 0
                    elif "w_b" in n:
                        if i_qB < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tq_B = torch.cat((tq_B, p_), dim=1)
                            i_qB += 1
                            if i_qB == self.num_particles:   
                                q_B = torch.cat((q_B, tq_B), dim=0)
                                tq_B = torch.empty(0).cuda()
                                i_qB = 0
                elif "proj_v" in n:
                    if "w_a" in n:
                        if i_vA < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tv_A = torch.cat((tv_A, p_), dim=1)
                            i_vA += 1
                            if i_vA == self.num_particles:   
                                v_A = torch.cat((v_A, tv_A), dim=0)
                                tv_A = torch.empty(0).cuda()
                                i_vA = 0
                    elif "w_b" in n:
                        if i_vB < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tv_B = torch.cat((tv_B, p_), dim=1)
                            i_vB += 1
                            if i_vB == self.num_particles:   
                                v_B = torch.cat((v_B, tv_B), dim=0)
                                tv_B = torch.empty(0).cuda()
                                i_vB = 0
                                
                elif 'fc' in n:
                    if 'weight' in n:
                        if i_cls_w < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tcls_w = torch.cat((tcls_w, p_), dim=1)
                            i_cls_w += 1
                            if i_cls_w == self.num_particles:   
                                cls_w = torch.cat((cls_w, tcls_w), dim=0)
                                tcls_w = torch.empty(0).cuda()
                                i_cls_w = 0
                    elif 'bias' in n:
                        if i_cls_b < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tcls_b = torch.cat((tcls_b, p_), dim=1)
                            i_cls_b += 1
                            if i_cls_b == self.num_particles:   
                                cls_b = torch.cat((cls_b, tcls_b), dim=0)
                                tcls_b = torch.empty(0).cuda()
                                i_cls_b = 0
                    
        return q_A, q_B, v_A, v_B, cls_w, cls_b
    
    def kernel_func(self, q_A, q_B, v_A, v_B, clsW, clsB):

        q_A.requires_grad = True
        q_B.requires_grad = True
        v_A.requires_grad = True
        v_B.requires_grad = True
        clsW.requires_grad = True
        clsB.requires_grad = True
        
        kernel_qA = self.K(q_A)
        self.train_module.manual_backward(kernel_qA.sum())
        q_A_grad = q_A.grad
        
        kernel_qB = self.K(q_B)
        self.train_module.manual_backward(kernel_qB.sum())
        q_B_grad = q_B.grad
        
        kernel_vA = self.K(v_A)
        self.train_module.manual_backward(kernel_vA.sum())
        v_A_grad = v_A.grad
        
        kernel_vB = self.K(v_B)
        self.train_module.manual_backward(kernel_vB.sum())
        v_B_grad = v_B.grad
        
        kernel_clsW = self.K(clsW)
        self.train_module.manual_backward(kernel_clsW.sum())
        clsW_gradK = clsW.grad
        
        kernel_clsB = self.K(clsB)
        self.train_module.manual_backward(kernel_clsB.sum())
        clsB_gradK = clsB.grad
        
        return kernel_qA, kernel_qB, kernel_vA, kernel_vB, kernel_clsW, kernel_clsB, q_A_grad, q_B_grad, v_A_grad, v_B_grad, clsW_gradK, clsB_gradK
        
    def step1(self):
        q_A_grad, q_B_grad, v_A_grad, v_B_grad, clsW_grad, clsB_grad = self.get_grad1() #dlog_prob(X)'
        
        # get kerr
        self.zero_grad()
        q_A, q_B, v_A, v_B, clsW, clsB = self.get_learnable_block()
        q_A, q_B, v_A, v_B, clsW, clsB = q_A.clone().detach().requires_grad_(True), q_B.clone().detach().requires_grad_(True), v_A.clone().detach().requires_grad_(True), v_B.clone().detach().requires_grad_(True),  clsW.clone().detach().requires_grad_(True), clsB.clone().detach().requires_grad_(True)
        org_weight_tuple = (q_A, q_B, v_A, v_B, clsW, clsB)
        
        if q_A.shape[0] > 0:
            kernel_qA, kernel_qB, kernel_vA, kernel_vB, kernel_clsW, kernel_clsB, q_A_gradK, q_B_gradK, v_A_gradK, v_B_gradK, clsW_gradK, clsB_gradK = self.kernel_func(q_A, q_B, v_A, v_B, clsW, clsB) #self.K(self.X, self.X.detach())
        else:
            kernel_qA, kernel_qB, kernel_vA, kernel_vB, kernel_clsW, kernel_clsB, q_A_gradK, q_B_gradK, v_A_gradK, v_B_gradK, clsW_gradK, clsB_gradK = torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), 0, 0, 0, 0, 0, 0
    
        kernel_tuple = (kernel_qA, kernel_qB, kernel_vA, kernel_vB, kernel_clsW, kernel_clsB, q_A_gradK, q_B_gradK, v_A_gradK, v_B_gradK, clsW_gradK, clsB_gradK)
        
        updated_n = []
        
        for net_id in range(self.num_particles):
            for layer_id in range(12):   
                for n, p in self.net.lora_vit.named_parameters():
                    
                    if p.requires_grad and n not in updated_n: 
                        
                        if f'blocks.{str(layer_id)}' in n:
                            if "proj_q" in n:
                                if f"w_a.layer.{net_id}" in n:
                                    updated_n.append(n)
                                    grad_n = torch.nn.functional.normalize(q_A_grad[layer_id][net_id],  p=2, dim=0)
                                    p.data = p.data + self.rho * grad_n.view(p.data.shape)
                                elif f"w_b.layer.{net_id}" in n:
                                    updated_n.append(n)
                                    grad_n = torch.nn.functional.normalize(q_B_grad[layer_id][net_id],  p=2, dim=0)
                                    p.data = p.data + self.rho * grad_n.view(p.data.shape)
                            elif "proj_v" in n:
                                if f"w_a.layer.{net_id}" in n:
                                    updated_n.append(n)
                                    grad_n = torch.nn.functional.normalize(v_A_grad[layer_id][net_id],  p=2, dim=0)
                                    p.data = p.data + self.rho * grad_n.view(p.data.shape)
                                elif f"w_b.layer.{net_id}" in n:
                                    updated_n.append(n)
                                    grad_n = torch.nn.functional.normalize(v_B_grad[layer_id][net_id],  p=2, dim=0)
                                    p.data = p.data + self.rho * grad_n.view(p.data.shape)
                                    
                        elif 'fc' in n:
                            if 'weight' in n:
                                if f"layer.{net_id}" in n:
                                    updated_n.append(n)
                                    grad_n = torch.nn.functional.normalize(clsW_grad[layer_id][net_id],  p=2, dim=0)
                                    p.data = p.data + self.rho * grad_n.view(p.data.shape)
                            elif 'bias' in n and f"layer.{net_id}" in n:
                                    updated_n.append(n)
                                    grad_n = torch.nn.functional.normalize(clsB_grad[layer_id][net_id],  p=2, dim=0)
                                    p.data = p.data + self.rho * grad_n.view(p.data.shape)
                                    
        return org_weight_tuple, kernel_tuple
    
    def step2(self, org_weight_tuple, kernel_tuple, zero_grad=True):
        q_A_grad, q_B_grad, v_A_grad, v_B_grad, clsW_grad, clsB_grad = self.get_grad1() #dlog_prob(X)'
        q_A, q_B, v_A, v_B, clsW, clsB = org_weight_tuple
        kernel_qA, kernel_qB, kernel_vA, kernel_vB, kernel_clsW, kernel_clsB, q_A_gradK, q_B_gradK, v_A_gradK, v_B_gradK, clsW_gradK, clsB_gradK = kernel_tuple
        
        #compute score func:
        grad_qA = (-kernel_qA.detach().matmul(q_A_grad) + q_A_gradK) / self.num_particles
        grad_qB = (-kernel_qB.detach().matmul(q_B_grad) + q_B_gradK) / self.num_particles
        grad_vA = (-kernel_vA.detach().matmul(v_A_grad) + v_A_gradK) / self.num_particles
        grad_vB = (-kernel_vB.detach().matmul(v_B_grad) + v_B_gradK) / self.num_particles
        grad_clsW = (-kernel_clsW.detach().matmul(clsW_grad) + clsW_gradK) / self.num_particles
        grad_clsB = (-kernel_clsB.detach().matmul(clsB_grad) + clsB_gradK) / self.num_particles
        
        #update weight:
        updated_n = []
        if self.base_optimizer_name == "sgd":
            for net_id in range(self.num_particles):
                for layer_id in range(12):   
                    for n, p in self.net.lora_vit.named_parameters():
                        if p.requires_grad and n not in updated_n: 

                            if f'blocks.{str(layer_id)}' in n:
                                if "proj_q" in n:
                                    if f"w_a.layer.{net_id}" in n:
                                        updated_n.append(n)
                                        print("name:", n)
                                        p_grad = -self.lr * grad_qA[layer_id][net_id].view(p.data.shape)
                                        p.data = q_A[layer_id][net_id].view(p.data.shape) - self.lr * p_grad
                                    elif f"w_b.layer.{net_id}" in n:
                                        updated_n.append(n)
                                        p_grad = -grad_qB[layer_id][net_id].view(p.data.shape)
                                        p.data = q_B[layer_id][net_id].view(p.data.shape) - self.lr * p_grad
                                elif "proj_v" in n:
                                    if f"w_a.layer.{net_id}" in n:
                                        updated_n.append(n)
                                        p_grad = -grad_vA[layer_id][net_id].view(p.data.shape)
                                        p.data = v_A[layer_id][net_id].view(p.data.shape) - self.lr * p_grad
                                    elif f"w_b.layer.{net_id}" in n:
                                        updated_n.append(n)
                                        p_grad = -grad_vB[layer_id][net_id].view(p.data.shape)
                                        p.data = v_B[layer_id][net_id].view(p.data.shape) - self.lr * p_grad
                                        
                            elif 'fc' in n:
                                if 'weight' in n:
                                    if f"layer.{net_id}" in n:
                                        updated_n.append(n)
                                        p_grad = -grad_clsW[layer_id][net_id].view(p.data.shape)
                                        p.data = clsW[layer_id][net_id].view(p.data.shape) - self.lr * p_grad
                                elif 'bias' in n and f"layer.{net_id}" in n:
                                        updated_n.append(n)
                                        p_grad = -grad_clsB[layer_id][net_id].view(p.data.shape)
                                        p.data = clsB[layer_id][net_id].view(p.data.shape) - self.lr * p_grad

        elif self.base_optimizer_name == "adamw": # Uses AdamW as the base optimizer
            # Based on PyTorch implementation: https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py#L323
            eps = 1e-6
            beta1, beta2 = self.betas[0], self.betas[1]
            for net_id in range(self.num_particles):
                for layer_id in range(12):   
                    for n, p in self.net.lora_vit.named_parameters():
                        if p.requires_grad and n not in updated_n:
                            if f'blocks.{str(layer_id)}' in n:
                                if "proj_q" in n:
                                    if f"w_a.layer.{net_id}" in n:
                                        updated_n.append(n)
                                        grad = -grad_qA[layer_id][net_id].view(p.data.shape)
                                        state = self.state[p]
                                        if len(state) == 0:
                                            state['step'] = 0
                                            state['exp_avg'] = torch.zeros_like(p.data)
                                            state['exp_avg_sq'] = torch.zeros_like(p.data)
                                        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                                        state['step'] += 1  #t = t+1
                                        p.data.mul_(1 - self.lr * self.weight_decay) 
                                        exp_avg.mul_(beta1).add_(grad, alpha= 1 - beta1) 
                                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                                        exp_avg_hat = exp_avg / (1 - beta1 ** state['step'])
                                        exp_avg_sq_hat = exp_avg_sq / (1 - beta2 ** state['step'] ) 
                                        denom = exp_avg_sq_hat.sqrt().add_(eps)  
                                        p.data.addcdiv_(exp_avg_hat, denom, value = -self.lr) 
                                    elif f"w_b.layer.{net_id}" in n:
                                        updated_n.append(n)
                                        grad = -grad_qB[layer_id][net_id].view(p.data.shape)
                                        state = self.state[p]
                                        if len(state) == 0:
                                            state['step'] = 0
                                            state['exp_avg'] = torch.zeros_like(p.data)
                                            state['exp_avg_sq'] = torch.zeros_like(p.data)
                                        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                                        state['step'] += 1  #t = t+1
                                        p.data.mul_(1 - self.lr * self.weight_decay) 
                                        exp_avg.mul_(beta1).add_(grad, alpha= 1 - beta1) 
                                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) 
                                        exp_avg_hat = exp_avg / (1 - beta1 ** state['step']) 
                                        exp_avg_sq_hat = exp_avg_sq / (1 - beta2 ** state['step'] ) 
                                        denom = exp_avg_sq_hat.sqrt().add_(eps) 
                                        p.data.addcdiv_(exp_avg_hat, denom, value = -self.lr) 
                                elif "proj_v" in n:
                                    if f"w_a.layer.{net_id}" in n:
                                        updated_n.append(n)
                                        grad = -grad_vA[layer_id][net_id].view(p.data.shape)
                                        state = self.state[p]
                                        if len(state) == 0:
                                            state['step'] = 0
                                            state['exp_avg'] = torch.zeros_like(p.data)
                                            state['exp_avg_sq'] = torch.zeros_like(p.data)
                                        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                                        state['step'] += 1  #t = t+1
                                        p.data.mul_(1 - self.lr * self.weight_decay)
                                        exp_avg.mul_(beta1).add_(grad, alpha= 1 - beta1) 
                                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                                        exp_avg_hat = exp_avg / (1 - beta1 ** state['step']) 
                                        exp_avg_sq_hat = exp_avg_sq / (1 - beta2 ** state['step'] )
                                        denom = exp_avg_sq_hat.sqrt().add_(eps)  
                                        p.data.addcdiv_(exp_avg_hat, denom, value = -self.lr) 
                                    elif f"w_b.layer.{net_id}" in n:
                                        updated_n.append(n)
                                        grad = -grad_vB[layer_id][net_id].view(p.data.shape)
                                        state = self.state[p]
                                        if len(state) == 0:
                                            state['step'] = 0
                                            state['exp_avg'] = torch.zeros_like(p.data)
                                            state['exp_avg_sq'] = torch.zeros_like(p.data)
                                        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                                        state['step'] += 1  #t = t+1
                                        p.data.mul_(1 - self.lr * self.weight_decay) 
                                        exp_avg.mul_(beta1).add_(grad, alpha= 1 - beta1) 
                                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) 
                                        exp_avg_hat = exp_avg / (1 - beta1 ** state['step'])
                                        exp_avg_sq_hat = exp_avg_sq / (1 - beta2 ** state['step'] ) 
                                        denom = exp_avg_sq_hat.sqrt().add_(eps) 
                                        p.data.addcdiv_(exp_avg_hat, denom, value = -self.lr) 
                            elif 'fc' in n:
                                if 'weight' in n:
                                    if f"layer.{net_id}" in n:
                                        updated_n.append(n)
                                        grad = -grad_clsW[layer_id][net_id].view(p.data.shape)
                                        state = self.state[p]
                                        if len(state) == 0:
                                            state['step'] = 0
                                            state['exp_avg'] = torch.zeros_like(p.data)
                                            state['exp_avg_sq'] = torch.zeros_like(p.data)
                                        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                                        state['step'] += 1  #t = t+1
                                        p.data.mul_(1 - self.lr * self.weight_decay) 
                                        exp_avg.mul_(beta1).add_(grad, alpha= 1 - beta1)
                                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) 
                                        exp_avg_hat = exp_avg / (1 - beta1 ** state['step']) 
                                        exp_avg_sq_hat = exp_avg_sq / (1 - beta2 ** state['step'] ) 
                                        denom = exp_avg_sq_hat.sqrt().add_(eps) 
                                        p.data.addcdiv_(exp_avg_hat, denom, value = -self.lr) 
                                elif 'bias' in n and f"layer.{net_id}" in n:
                                        updated_n.append(n)
                                        grad = -grad_clsB[layer_id][net_id].view(p.data.shape)
                                        state = self.state[p]
                                        if len(state) == 0:
                                            state['step'] = 0
                                            state['exp_avg'] = torch.zeros_like(p.data)
                                            state['exp_avg_sq'] = torch.zeros_like(p.data)
                                        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                                        state['step'] += 1  #t = t+1
                                        p.data.mul_(1 - self.lr * self.weight_decay) 
                                        exp_avg.mul_(beta1).add_(grad, alpha= 1 - beta1) 
                                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) 
                                        exp_avg_hat = exp_avg / (1 - beta1 ** state['step']) 
                                        exp_avg_sq_hat = exp_avg_sq / (1 - beta2 ** state['step'] ) 
                                        denom = exp_avg_sq_hat.sqrt().add_(eps)  
                                        p.data.addcdiv_(exp_avg_hat, denom, value = -self.lr) 