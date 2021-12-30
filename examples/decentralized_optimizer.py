import math

import bluefog.torch as bf
import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class DSGDOptimizer(torch.optim.SGD):
    def __init__(self, params, model, nu=0.01, communication_type=bf.CommunicationType.neighbor_allreduce,
                 **kwargs):
        super(self.__class__, self).__init__(params, **kwargs)

        self.nu = nu 
        self.enable_topo_check = True
        self.self_weight = None
        self.src_weights = None
        self.dst_weights = None
        self._communication_type = communication_type

        self._parameter_names = {v: k for k, v in model.named_parameters()}
        self._name_parameters = {k: v for k, v in model.named_parameters()}
        self._handles = {}
        self.adjust_lr_by_stddev = True

        if bf.size() > 1:
            self._register_hooks()

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group["params"]:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    p.register_hook(self._make_hook(p, param_group)) 

    def _make_hook(self, p, param_group):
        def hook(grad): 

            # w^{k+1/2} = w^k - lr * g^k
            self.sgd_step(p, grad.data, param_group)

            # w^{k+1} = A w^{k}
            if self._communication_type == bf.CommunicationType.allreduce:
                handle = self._allreduce_data_async(p)
            elif self._communication_type == bf.CommunicationType.neighbor_allreduce:
                handle = self._neighbor_allreduce_data_async(p)
            elif self._communication_type == bf.CommunicationType.empty:
                handle = None
            else:
                raise ValueError("Unsuppported CommunicationType encountered.")
            self._handles[p] = handle

        return hook

    def sgd_step(self, p, grad, param_group):
        """Parameter-wise of torch.optim.SGD.step."""
        weight_decay = param_group['weight_decay']
        momentum = param_group['momentum']
        dampening = param_group['dampening']
        nesterov = param_group['nesterov']
        lr = param_group['lr']
        d_p = grad
        if weight_decay != 0:
            d_p.add_(p.data, alpha=weight_decay)
        if momentum != 0:
            param_state = self.state[p]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.clone(
                    d_p).detach()
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        p.data.add_(d_p, alpha=-lr)

    def _neighbor_allreduce_data_async(self, p):
        name = self._parameter_names.get(p)
        handle = bf.neighbor_allreduce_nonblocking(p.data, name=name, self_weight=self.self_weight,
                                                   src_weights=self.src_weights,
                                                   dst_weights=self.dst_weights,
                                                   enable_topo_check=self.enable_topo_check)
        return handle

    def _allreduce_data_async(self, p):
        name = self._parameter_names.get(p)
        handle = bf.allreduce_nonblocking(p.data, average=True, name=name)
        return handle

    def flattern_stddev(self):
        flat_std = []
        for param_group in self.param_groups:
            for p in param_group["params"]:
                flat_std.append(self.state[p]['exp_avg_sq'])
        return _flatten_dense_tensors(flat_std)
        
    def step(self, closure=None):
        
        with torch.no_grad():
            for p, handle in self._handles.items():
                if handle is not None:
                    output = bf.synchronize(handle)
                    p.set_(output)
        self._handles.clear()

        # with torch.no_grad():
        #     for param_group in self.param_groups:
        #         for p in param_group["params"]:
        #             if p not in self._handles:
        #                 # raise ValueError('Cannot find parameter p in handler')
        #                 self.sgd_step(p, p.grad, param_group)
        #                 continue

        #             with torch.no_grad():
        #                 self.sgd_step(p, p.grad, param_group)
        # self._handles.clear()

    def zero_grad(self):
        if self._handles:
            raise AssertionError(
                "optimizer.zero_grad() was called after loss.backward() "
                "but before optimizer.step() or optimizer.synchronize(). "
                "This is prohibited as it can cause a race condition."
            )
        return super(self.__class__, self).zero_grad()
