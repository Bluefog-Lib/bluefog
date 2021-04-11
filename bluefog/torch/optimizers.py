# Modifications copyright (C) 2020 Bluefog Team. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from contextlib import contextmanager
from enum import Enum
import functools
import itertools
import math
import os
import warnings

import torch
import bluefog.torch as bf

class CommunicationType(Enum):
    neighbor_allreduce = "neighbor.allreduce"
    hierarchical_neighbor_allreduce = "hierarchical.neighbor.allreduce"
    allreduce = "allreduce"
    empty = "empty"

# TODO(hanbinhu): Add URL for FAQ page
_warning_message_num_step_per_communication = (
    "Unexpected behavior:\n"
    "  After num_steps_per_communication times of forward computation `y=model(x)` are called,\n"
    "  an optimizer step() function must be called.\n"
    "  It does not matter how many step() functions are called in between.\n"
    "  Please adjust num_step_per_communication to update model parameters locally.\n"
    "  More information can be found in the FAQ page.\n"
)
_warning_message_backward_pass_per_step = (
    "Unexpected behavior:\n"
    "  After num_steps_per_communication times of backward computation `loss.backward()` are called,\n"
    "  an optimizer step() function must be called.\n"
    "  It does not matter how many step() functions are called in between.\n"
    "  Please adjust num_steps_per_communication to accumulate gradients locally.\n"
    "  More information can be found in the FAQ page.\n"
)

#pylint: disable=unused-argument
def _named_leaf_module(module, parent_name=None):
    """Yield an iterator over all leaf modules."""
    if not list(module.named_children()):
        yield (parent_name, module)
    for name, ch_module in module.named_children():
        full_name = (parent_name + '.' + name if parent_name else name)
        yield from _named_leaf_module(ch_module, full_name)


def _find_duplicates(lst):
    seen = set()
    dups = set()
    for el in lst:
        if el in seen:
            dups.add(el)
        seen.add(el)
    return dups


def _check_named_parameters(optimizer, model):
    if isinstance(model, torch.nn.Module):
        _models = [model]
    if isinstance(model, list):
        for m in model:
            assert isinstance(m, torch.nn.Module)
        _models = model

    named_parameters = list(itertools.chain(
        *[m.named_parameters() for m in _models]))

    # make sure that named_parameters are tuples
    if any([not isinstance(p, tuple) for p in named_parameters]):
        raise ValueError(
            "named_parameters should be a sequence of "
            "tuples (name, parameter), usually produced by "
            "model.named_parameters()."
        )

    dups = _find_duplicates([k for k, _ in named_parameters])
    if dups:
        raise ValueError(
            "Parameter names in named_parameters must be unique. "
            "Found duplicates: %s" % ", ".join(dups)
        )

    all_param_ids = {
        id(v) for param_group in optimizer.param_groups for v in param_group["params"]
    }
    named_param_ids = {id(v) for k, v in named_parameters}
    unnamed_param_ids = all_param_ids - named_param_ids
    if unnamed_param_ids:
        raise ValueError(
            "Named parameters provided by model are mismatch with the parameters"
            "handled by optimizer. Python object ids: "
            "%s" % ", ".join(str(id) for id in unnamed_param_ids)
        )
    return named_parameters, _models


def _register_timeline(optimizer, models, parameter_names, parent_name=None):
    def _timeline_hook(module, *unused):
        for name, _ in module.named_parameters():
            full_name = parent_name+'.'+name if parent_name else name
            bf.timeline_start_activity(
                full_name, activity_name="GRADIENT COMPT.")
    backward_hook_handles = []
    for model in models:
        backward_hook_handles.append(
            model.register_backward_hook(_timeline_hook))

    def _make_backward_end_timeline_hook(name):
        def hook(*ignore):
            bf.timeline_end_activity(name)
        return hook

    backward_end_hook_handles = []
    for param_group in optimizer.param_groups:
        for p in param_group["params"]:
            if p.requires_grad:
                name = parameter_names.get(p)
                full_name = parent_name+'.'+name if parent_name else name
                h = p.register_hook(_make_backward_end_timeline_hook(full_name))
                backward_end_hook_handles.append(h)

    def _timeline_forward_pre_hook(module, *unused):
        for name, _ in module.named_parameters():
            full_name = parent_name+'.'+name if parent_name else name
            bf.timeline_start_activity(full_name, activity_name="FORWARD")

    pre_forward_hook_handles = []
    for model in models:
        pre_forward_hook_handles.append(model.register_forward_pre_hook(
            _timeline_forward_pre_hook))

    def _make_timeline_forward_end_hook(parent_name):
        def _timeline_forward_end_hook(module, *unused):
            for name, _ in module.named_parameters():
                full_name = parent_name+'.'+name if parent_name else name
                bf.timeline_end_activity(full_name)
        return _timeline_forward_end_hook

    forward_end_hook_handles = []
    for model in models:
        for name, layer in _named_leaf_module(model):
            full_name = parent_name+'.'+name if parent_name else name
            handle = layer.register_forward_hook(
                _make_timeline_forward_end_hook(full_name))
            forward_end_hook_handles.append(handle)

    return [*backward_hook_handles, *backward_end_hook_handles,
            *pre_forward_hook_handles, *forward_end_hook_handles]


class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, model, backward_passes_per_step=1):
        super(self.__class__, self).__init__(params)

        named_parameters, models = _check_named_parameters(self, model)
        self._models = models
        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._synchronized = False
        self._should_synchronize = True
        self._timeline_hook_handles = []
        self._use_timeline = False
        self._backward_passes_per_step = backward_passes_per_step
        self._allreduce_delay = {v: self._backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        self._error_encountered = False
        if os.getenv('BLUEFOG_TIMELINE'):
            self.turn_on_timeline()
        if bf.size() > 1:
            self._register_hooks()

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group["params"]:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _make_hook(self, p):
        def hook(*ignore):
            assert not p.grad.requires_grad
            if self._allreduce_delay[p] <= 0:
                if not self._error_encountered:
                    warnings.warn(_warning_message_backward_pass_per_step)
                    self._error_encountered = True
            self._allreduce_delay[p] -= 1
            if self._allreduce_delay[p] == 0:
                handle = self._allreduce_grad_async(p)
                self._handles[p] = handle

        return hook

    def _allreduce_grad_async(self, p):
        name = self._parameter_names.get(p)
        handle = bf.allreduce_nonblocking(
            p.grad, average=True, name=name
        )
        return handle

    def turn_on_timeline(self):
        handles = _register_timeline(
            self, self._models, self._parameter_names, 'allreduce')
        self._timeline_hook_handles.extend(handles)
        self._use_timeline = True

    def turn_off_timeline(self):
        for hook in self._timeline_hook_handles:
            hook.remove()
        self._timeline_hook_handles.clear()
        self._use_timeline = False

    def synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle = self._allreduce_grad_async(p)
            self._handles[p] = handle

        for p, handle in self._handles.items():
            if handle is None:
                handle = self._allreduce_grad_async(p)
                self._handles[p] = handle

        for p, handle in self._handles.items():
            output = bf.synchronize(handle)
            self._allreduce_delay[p] = self._backward_passes_per_step
            p.grad.set_(output)
        self._handles.clear()

        self._synchronized = True

    @contextmanager
    def skip_synchronize(self):
        """
        A context manager used to specify that optimizer.step() should
        not perform synchronization.

        It's typically used in a following pattern:

        .. code-block:: python

            optimizer.synchronize()
            with optimizer.skip_synchronize():
                optimizer.step()
        """
        self._should_synchronize = False
        try:
            yield
        finally:
            self._should_synchronize = True

    def step(self, closure=None):
        if self._should_synchronize:
            if self._synchronized:
                warnings.warn(
                    "optimizer.step() called without "
                    "optimizer.skip_synchronize() context after "
                    "optimizer.synchronize(). This can cause training "
                    "slowdown. You may want to consider using "
                    "optimizer.skip_synchronize() context if you use "
                    "optimizer.synchronize() in your code."
                )
            self.synchronize()
        self._synchronized = False
        return super(self.__class__, self).step(closure)

    def zero_grad(self):
        if self._handles:
            raise AssertionError(
                "optimizer.zero_grad() was called after loss.backward() "
                "but before optimizer.step() or optimizer.synchronize(). "
                "This is prohibited as it can cause a race condition."
            )
        return super(self.__class__, self).zero_grad()


class _DistributedReduceOptimizer(torch.optim.Optimizer):
    """ A distributed optimizer wrapper over torch optimizer.

    Arguments:
        reduce_type: either to be "allreduce" or "neighbor_allreduce" to decide the reducing
                     method for communication.

    Note: Unlike the _DistributedOptimizer class that registers hook for each named parameters,
    triggers the allreduce_nonblocking after gradient computation is finished, and updates the
    parameters at last step function. We will trigger the win_put ops that puts the weights
    to its neighbor and compute average of iterates instead of gradient, i.e. combine-then-adapt
    (CTA) strategy. In theory, adapt-then-combine has superior performance, but it is much more
    difficult to be implemented.

    By equation, there are three styles (w --- iterates, i --- iteration, k --- agent number)
        w_{i+1, k} = w_{i, k} - lr * Global_Average( grad(w_{i,k}) )
    Consensus Style:
        w_{i+1, k} = Neighbor_Average(w_{i, k}) - lr * local_grad(w_{i, k})
    CTA style:
        w_{i+1, k} = Neighbor_Average(w_{i, k}) - lr * local_grad(Neighbor_Average(w_{i, k}))
    ATC style:
        w_{i+1, k} = Neighbor_Average( w_{i, k} - lr * local_grad(w_{i, k}) )
    """

    def __init__(self, params, model, communication_type, num_steps_per_communication=1):
        super(self.__class__, self).__init__(params)

        named_parameters, models = _check_named_parameters(self, model)
        # knobs for neighbor communication behavior
        self.self_weight = None
        self.src_weights = None
        self.dst_weights = None
        self.neighbor_machine_weights = None
        self.send_neighbor_machines = None
        self.enable_topo_check = False

        self._models = models
        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self._name_parameters = {k: v for k, v in sorted(named_parameters)}
        self._handles = {}
        self._requires_update = set()
        self._synchronized = False
        self._should_synchronize = True
        self._timeline_hook_handles = []
        self._use_timeline = False
        self._error_encountered = False
        self._num_steps_per_communication = num_steps_per_communication
        assert isinstance(communication_type, CommunicationType)
        self._communication_type = communication_type

        self._reduce_delay = {v: self._num_steps_per_communication
                              for _, v in sorted(named_parameters)}
        if os.getenv('BLUEFOG_TIMELINE'):
            self.turn_on_timeline()
        if bf.size() > 1:
            self._register_hooks()

    def _register_hooks(self):
        for model in self._models:
            # The hook is added at model level instead of layer level, as it avoids triggering
            # the hook function of the same layer multiple times in case the layer is called 
            # several times during the forward computation of the model.
            model.register_forward_hook(self._make_hook())
            self._requires_update.update(dict(model.named_parameters()).values())

    def _make_hook(self):
        def hook(model, *unused):
            for parent_name, layer in _named_leaf_module(model):
                for name, p in layer.named_parameters():
                    if not layer.training:
                        continue
                    if self._name_parameters.get(parent_name+'.'+name, None) is None:
                        # Some case like encoder-decode, which shared the same weights.
                        continue
                    if self._use_timeline:
                        # End forward computation timeline
                        bf.timeline_end_activity(parent_name+'.'+name)
                    if p.requires_grad:
                        if self._reduce_delay[p] <= 0:
                            if not self._error_encountered:
                                warnings.warn(_warning_message_num_step_per_communication)
                                self._error_encountered = True
                        self._reduce_delay[p] -= 1
                        if self._reduce_delay[p] == 0:
                            if self._communication_type == CommunicationType.allreduce:
                                handle = self._allreduce_data_async(p)
                            elif self._communication_type == CommunicationType.neighbor_allreduce:
                                handle = self._neighbor_allreduce_data_async(p)
                            elif self._communication_type == CommunicationType.hierarchical_neighbor_allreduce:
                                handle = self._hierarchical_neighbor_allreduce_data_async(p)
                            elif self._communication_type == CommunicationType.empty:
                                handle = None
                            else:
                                raise ValueError("Unsuppported CommunicationType encountered.")
                            self._handles[p] = handle
        return hook

    def _neighbor_allreduce_data_async(self, p):
        name = self._parameter_names.get(p)
        handle = bf.neighbor_allreduce_nonblocking(p.data, name=name, self_weight=self.self_weight,
                                                   src_weights=self.src_weights,
                                                   dst_weights=self.dst_weights,
                                                   enable_topo_check=self.enable_topo_check)
        return handle

    def _hierarchical_neighbor_allreduce_data_async(self, p):
        name = self._parameter_names.get(p)
        handle = bf.hierarchical_neighbor_allreduce_nonblocking(
            p.data, name=name, self_weight=self.self_weight,
            neighbor_machine_weights=self.neighbor_machine_weights,
            send_neighbor_machines=self.send_neighbor_machines,
            enable_topo_check=self.enable_topo_check)
        return handle

    def _allreduce_data_async(self, p):
        name = self._parameter_names.get(p)
        handle = bf.allreduce_nonblocking(p.data, average=True, name=name)
        return handle

    def turn_on_timeline(self):
        handles = _register_timeline(
            self, self._models, self._parameter_names, self._communication_type.value)
        self._timeline_hook_handles.extend(handles)
        self._use_timeline = True

    def turn_off_timeline(self):
        for hook in self._timeline_hook_handles:
            hook.remove()
        self._timeline_hook_handles.clear()
        self._use_timeline = False

    @property
    def communication_type(self):
        return self._communication_type

    @communication_type.setter
    def communication_type(self, value):
        assert isinstance(value, CommunicationType)
        self._communication_type = value

    def synchronize(self):
        with torch.no_grad():
            for p, handle in self._handles.items():
                if handle is not None:
                    output = bf.synchronize(handle)
                    p.set_(output)
                self._reduce_delay[p] = self._num_steps_per_communication
        self._handles.clear()

        self._synchronized = True

    @contextmanager
    def skip_synchronize(self):
        """
        A context manager used to specify that optimizer.step() should
        not perform synchronization.

        It's typically used in a following pattern:

        .. code-block:: python

            optimizer.synchronize()
            with optimizer.skip_synchronize():
                optimizer.step()
        """
        self._should_synchronize = False
        try:
            yield
        finally:
            self._should_synchronize = True

    def step(self, closure=None):
        # consensus style is the easist way to implement it.
        if self._should_synchronize:
            if self._synchronized:
                warnings.warn(
                    "optimizer.step() called without "
                    "optimizer.skip_synchronize() context after "
                    "optimizer.synchronize(). This can cause training "
                    "slowdown. You may want to consider using "
                    "optimizer.skip_synchronize() context if you use "
                    "optimizer.synchronize() in your code."
                )
            self.synchronize()
        self._synchronized = False
        return super(self.__class__, self).step(closure)


class _DistributedAdaptThenCombineOptimizer(torch.optim.Optimizer):
    def __init__(self, params, model, communication_type, backward_passes_per_step=1):
        super(self.__class__, self).__init__(params)

        named_parameters, models = _check_named_parameters(self, model)
        # knobs for neighbor communication behavior
        self.self_weight = None
        self.src_weights = None
        self.dst_weights = None
        self.neighbor_machine_weights = None
        self.send_neighbor_machines = None
        self.enable_topo_check = False

        self._models = models
        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self._handles = {}
        self._requires_update = set()
        self._synchronized = False
        self._should_synchronize = True
        self._timeline_hook_handles = []
        self._use_timeline = False
        self._error_encountered = False
        self._backward_passes_per_step = backward_passes_per_step
        assert isinstance(communication_type, CommunicationType)
        self._communication_type = communication_type

        if isinstance(self, torch.optim.SGD):
            self._step_func = self._sgd_step
        elif isinstance(self, torch.optim.Adam):
            self._step_func = self._adam_step
        elif isinstance(self, torch.optim.RMSprop):
            self._step_func = self._rmsprop_step
        elif isinstance(self, torch.optim.Adagrad):
            self._step_func = self._adagrad_step
        elif isinstance(self, torch.optim.Adadelta):
            self._step_func = self._adadelta_step
        else:
            self._step_func = None  # Need user to register their own step function.

        self._reduce_delay = {v: self._backward_passes_per_step
                              for _, v in sorted(named_parameters)}
        self._step_delay = {v: self._backward_passes_per_step
                            for _, v in sorted(named_parameters)}
        if os.getenv('BLUEFOG_TIMELINE'):
            self.turn_on_timeline()
        if bf.size() > 1:
            self._register_hooks()

    @property
    def communication_type(self):
        return self._communication_type

    @communication_type.setter
    def communication_type(self, value):
        assert isinstance(value, CommunicationType)
        self._communication_type = value

    def register_step_function(self, step_func):
        """Register the step function.

        Args:
            step_func: The signature should be:
                   func(self, parameter, gradient, parameter_group) -> None
                Where self refer to the instance of this distributed optimizer. A common
                usage is to get the stat of parameter through self.state[p].
                Note, it has to be paramter-wise and parameter_group is the one that the
                standard torch.optimizer provided, which can store the auxuilary information
                or state of optimizer like learning_rate, weight_decay, etc.
        """
        self._step_func = functools.partial(step_func, self)

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group["params"]:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p.register_hook(self._make_hook(p, param_group))

    def _make_hook(self, p, param_group):
        def hook(grad):
            # run the step first:
            if self._step_func is None:
                raise ValueError(
                    "We don't have default implementation for the optimizer you provided. "
                    "However, you can register you own step to this class. The signature "
                    "should be parameter-wise tep func:\n"
                    "   func(parameter, gradient, parameter_group) -> None\n")
            if self._step_delay[p] <= 0:
                if not self._error_encountered:
                    warnings.warn(_warning_message_num_step_per_communication)
                    self._error_encountered = True
            self._step_delay[p] -= 1
            if self._step_delay[p] == 0:
                self._step_func(p, grad.data, param_group)

            if self._reduce_delay[p] <= 0:
                if not self._error_encountered:
                    warnings.warn(_warning_message_num_step_per_communication)
                    self._error_encountered = True
            self._reduce_delay[p] -= 1
            if self._reduce_delay[p] == 0:
                if self._communication_type == CommunicationType.allreduce:
                    handle = self._allreduce_data_async(p)
                elif self._communication_type == CommunicationType.neighbor_allreduce:
                    handle = self._neighbor_allreduce_data_async(p)
                elif self._communication_type == CommunicationType.hierarchical_neighbor_allreduce:
                    handle = self._hierarchical_neighbor_allreduce_data_async(p)
                elif self._communication_type == CommunicationType.empty:
                    handle = None
                else:
                    raise ValueError("Unsuppported CommunicationType encountered.")
                self._handles[p] = handle

        return hook

    def _sgd_step(self, p, grad, param_group):
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
                buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        p.data.add_(d_p, alpha=-lr)

    def _adam_step(self, p, grad, param_group):
        """Parameter-wise of torch.optim.Adam.step."""
        if grad.is_sparse:
            raise RuntimeError(
                'Adam does not support sparse gradients, please consider SparseAdam instead')
        amsgrad = param_group['amsgrad']
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = param_group['betas']

        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        if param_group['weight_decay'] != 0:
            grad.add_(p.data, alpha=param_group['weight_decay'])

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(param_group['eps'])
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(param_group['eps'])

        step_size = param_group['lr'] / bias_correction1
        p.data.addcdiv_(-step_size, exp_avg, denom)

    def _rmsprop_step(self, p, grad, param_group):
        if grad.is_sparse:
            raise RuntimeError('RMSprop does not support sparse gradients')
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['square_avg'] = torch.zeros_like(p.data)
            if param_group['momentum'] > 0:
                state['momentum_buffer'] = torch.zeros_like(p.data)
            if param_group['centered']:
                state['grad_avg'] = torch.zeros_like(p.data)

        square_avg = state['square_avg']
        alpha = param_group['alpha']

        state['step'] += 1
        if param_group['weight_decay'] != 0:
            grad = grad.add(p.data, alpha=param_group['weight_decay'])

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

        if param_group['centered']:
            grad_avg = state['grad_avg']
            grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(param_group['eps'])
        else:
            avg = square_avg.sqrt().add_(param_group['eps'])

        if param_group['momentum'] > 0:
            buf = state['momentum_buffer']
            buf.mul_(param_group['momentum']).addcdiv_(grad, avg)
            p.data.add_(buf, alpha=-param_group['lr'])
        else:
            p.data.addcdiv_(grad, avg, alpha=-param_group['lr'])

    def _adagrad_step(self, p, grad, param_group):
        state = self.state[p]
        state['step'] += 1

        if param_group['weight_decay'] != 0:
            if grad.is_sparse:
                raise RuntimeError("weight_decay option is not compatible with sparse gradients")
            grad = grad.add(p.data, alpha=param_group['weight_decay'])

        clr = param_group['lr'] / (1 + (state['step'] - 1) * param_group['lr_decay'])

        if grad.is_sparse:
            grad = grad.coalesce()  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()
            size = grad.size()

            def make_sparse(values):
                constructor = grad.new
                if grad_indices.dim() == 0 or values.dim() == 0:
                    return constructor().resize_as_(grad)
                return constructor(grad_indices, values, size)
            state['sum'].add_(make_sparse(grad_values.pow(2)))
            std = state['sum'].sparse_mask(grad)
            std_values = std._values().sqrt_().add_(param_group['eps'])
            p.data.add_(make_sparse(grad_values / std_values), alpha=-clr)
        else:
            state['sum'].addcmul_(1, grad, grad)
            std = state['sum'].sqrt().add_(param_group['eps'])
            p.data.addcdiv_(grad, std, value=-clr)

    def _adadelta_step(self, p, grad, param_group):
        if grad.is_sparse:
            raise RuntimeError('Adadelta does not support sparse gradients')
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['square_avg'] = torch.zeros_like(p.data)
            state['acc_delta'] = torch.zeros_like(p.data)

        square_avg, acc_delta = state['square_avg'], state['acc_delta']
        rho, eps = param_group['rho'], param_group['eps']

        state['step'] += 1

        if param_group['weight_decay'] != 0:
            grad = grad.add(p.data, alpha=param_group['weight_decay'])

        square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
        std = square_avg.add(eps).sqrt_()
        delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
        p.data.add_(delta, alpha=-param_group['lr'])
        acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)

    def _neighbor_allreduce_data_async(self, p):
        name = self._parameter_names.get(p)
        handle = bf.neighbor_allreduce_nonblocking(p.data, name=name, self_weight=self.self_weight,
                                                   src_weights=self.src_weights,
                                                   dst_weights=self.dst_weights,
                                                   enable_topo_check=self.enable_topo_check)
        return handle

    def _hierarchical_neighbor_allreduce_data_async(self, p):
        name = self._parameter_names.get(p)
        handle = bf.hierarchical_neighbor_allreduce_nonblocking(
            p.data, name=name, self_weight=self.self_weight,
            neighbor_machine_weights=self.neighbor_machine_weights,
            send_neighbor_machines=self.send_neighbor_machines,
            enable_topo_check=self.enable_topo_check)
        return handle

    def _allreduce_data_async(self, p):
        name = self._parameter_names.get(p)
        handle = bf.allreduce_nonblocking(p.data, average=True, name=name)
        return handle

    def turn_on_timeline(self):
        handles = _register_timeline(
            self, self._models, self._parameter_names, self._communication_type.value)
        self._timeline_hook_handles.extend(handles)
        self._use_timeline = True

    def turn_off_timeline(self):
        for hook in self._timeline_hook_handles:
            hook.remove()
        self._timeline_hook_handles.clear()
        self._use_timeline = False

    def synchronize(self):
        with torch.no_grad():
            for p, handle in self._handles.items():
                if handle is not None:
                    output = bf.synchronize(handle)
                    p.set_(output)
                self._reduce_delay[p] = self._backward_passes_per_step
        self._handles.clear()

        self._synchronized = True

    def step(self, closure=None):
         # Step 0 is called for parameter initialization after parameter broadcast
        if bf.size() > 1 and self._handles:
            loss = None
            if closure is not None:
                loss = closure()

            if all(v != 0 for k, v in self._step_delay.items()):
                # This corresponding to case 2 multiple local updates.
                super(self.__class__, self).step()
            elif all(v == 0 for k, v in self._step_delay.items()):
                self._step_delay = self._step_delay.fromkeys(
                    self._step_delay, self._backward_passes_per_step)
            else:
                raise ValueError(
                    "We do not support partial step update in ATC yet.")

            self.synchronize()
            # TODO(ybc) Figure out a better and more robust way to do sync in ATC.
            # Note, tere self. _synchronized just turns from true to false immediately.
            self._synchronized = False

            return loss
        else:
            # Optimizer.step() might be triggered when user calls broadcast_optimizer_state()
            super(self.__class__, self).step(closure)

    def zero_grad(self):
        if self._handles:
            raise AssertionError(
                "optimizer.zero_grad() was called after loss.backward() "
                "but before optimizer.step() or optimizer.synchronize(). "
                "This is prohibited as it can cause a race condition."
            )
        return super(self.__class__, self).zero_grad()


class _DistributedWinOptimizer(torch.optim.Optimizer):

    def __init__(self, params, model, num_steps_per_communication, window_prefix, pull_style):
        super(self.__class__, self).__init__(params)

        if pull_style:
            self.src_weights = None # use to control the behavior of win_get dynamically.
        else:
            self.dst_weights = None # use to control the behavior of win_put dynamically.
        self.force_barrier = False
        self.window_prefix = window_prefix+'.' if window_prefix is not None else ''

        named_parameters, models = _check_named_parameters(self, model)
        self._models = models
        self._pull_style = pull_style
        self._parameter_names = {v: self.window_prefix+k for k, v in sorted(named_parameters)}
        self._handles = {}  # store parameter -> handle
        self._synchronized = False
        self._should_synchronize = True
        self._use_timeline = False
        self._error_encountered = False
        self._num_steps_per_communication = num_steps_per_communication
        self._bluefog_delay = {v: self._num_steps_per_communication
                               for _, v in sorted(named_parameters)}
        self._timeline_hook_handles = []
        if os.getenv('BLUEFOG_TIMELINE'):
            self.turn_on_timeline()
        if bf.size() > 1:
            self._register_window()
            self._register_hooks()

    def __del__(self):
        self.unregister_window()

    def _register_hooks(self):
        for model in self._models:
            # The hook is added at model level instead of layer level, as it avoids triggering
            # the hook function of the same layer multiple times in case the layer is called 
            # several times during the forward computation of the model.
            if self._pull_style:
                hook = self._make_get_hook()
            else:
                hook = self._make_put_hook()
            model.register_forward_hook(hook)

    def _make_put_hook(self):
        def hook(model, *unused):
            for parent_name, layer in _named_leaf_module(model):
                for name, p in layer.named_parameters():
                    if self._use_timeline:
                        # End forward computation timeline
                        bf.timeline_end_activity(self.window_prefix+parent_name+'.'+name)
                    if not layer.training:
                        continue
                    if p.requires_grad:
                        if self._bluefog_delay[p] <= 0:
                            if not self._error_encountered:
                                warnings.warn(_warning_message_num_step_per_communication)
                                self._error_encountered = True
                        self._bluefog_delay[p] -= 1
                        if self._bluefog_delay[p] == 0:
                            handle = bf.win_put_nonblocking(
                                tensor=p.data, name=self.window_prefix+parent_name+'.'+name,
                                dst_weights=self.dst_weights, require_mutex=False)
                            self._handles[p] = handle
        return hook

    def _make_get_hook(self):
        def hook(model, *unused):
            for parent_name, layer in _named_leaf_module(model):
                for name, p in layer.named_parameters():
                    if self._use_timeline:
                        # End forward computation timeline
                        bf.timeline_end_activity(self.window_prefix+parent_name+'.'+name)
                    if not layer.training:
                        continue
                    if p.requires_grad:
                        if self._bluefog_delay[p] <= 0:
                            if not self._error_encountered:
                                warnings.warn(_warning_message_num_step_per_communication)
                                self._error_encountered = True
                        self._bluefog_delay[p] -= 1
                        if self._bluefog_delay[p] == 0:
                            handle = bf.win_get_nonblocking(
                                name=self.window_prefix+parent_name+'.'+name,
                                src_weights=self.src_weights, require_mutex=True)
                            self._handles[p] = handle
        return hook

    def _register_window(self):
        if bf.size() <= 1:
            return
        for param_group in self.param_groups:
            for p in param_group["params"]:
                name = self._parameter_names.get(p)
                if name is None:
                    raise KeyError(
                        "Cannot find parameter {} in the _parameter_names dictionary".format(name))
                if not bf.win_create(p.data, name):
                    raise ValueError(
                        "Cannot allocate MPI window for the parameter {}".format(name))

    def unregister_window(self):
        ''' Unregister MPI Window objects for the optimizer manually.
        '''
        if bf.size() <= 1:
            return
        for param_group in self.param_groups:
            for p in param_group["params"]:
                name = self._parameter_names.get(p)
                if name is None:
                    raise KeyError(
                        "Cannot find parameter {} in the _parameter_names dictionary".format(name))
                if name in bf.get_current_created_window_names():
                    if not bf.win_free(name):
                        raise ValueError(
                            "Cannot free MPI window for the parameter {}".format(name))

    def turn_on_timeline(self):
        handles = _register_timeline(self, self._models, self._parameter_names)
        self._timeline_hook_handles.extend(handles)
        self._use_timeline = True

    def turn_off_timeline(self):
        for hook in self._timeline_hook_handles:
            hook.remove()
        self._timeline_hook_handles.clear()
        self._use_timeline = False

    @contextmanager
    def skip_synchronize(self):
        """
        A context manager used to specify that optimizer.step() should
        not perform synchronization.

        It's typically used in a following pattern:

        .. code-block:: python

            optimizer.synchronize()
            with optimizer.skip_synchronize():
                optimizer.step()
        """
        self._should_synchronize = False
        try:
            yield
        finally:
            self._should_synchronize = True

    def synchronize(self):
        # Here synchronize just to make sure win_put ops is finished
        # in one iteration.
        with torch.no_grad():
            for p, handle in self._handles.items():
                _ = bf.win_wait(handle)
                name = self._parameter_names.get(p)
                self._bluefog_delay[p] = self._num_steps_per_communication
                # Update p to the average of neighbors.
                p.set_(bf.win_update(name=name, require_mutex=True))

        self._handles.clear()
        self._synchronized = True

    def step(self, closure=None):
        if self.force_barrier:
            bf.barrier()
        # some validation here?
        if self._should_synchronize:
            if self._synchronized:
                warnings.warn(
                    "optimizer.step() called without "
                    "optimizer.skip_synchronize() context after "
                    "optimizer.synchronize(). This can cause training "
                    "slowdown. You may want to consider using "
                    "optimizer.skip_synchronize() context if you use "
                    "optimizer.synchronize() in your code."
                )
            self.synchronize()
        self._synchronized = False
        return super(self.__class__, self).step(closure)


class _DistributedPushSumOptimizer(torch.optim.Optimizer):

    def __init__(self, params, model, num_steps_per_communication):
        super(self.__class__, self).__init__(params)

        # use to control the behavior of win_accumulate dynamically.
        outdegree = len(bf.out_neighbor_ranks())
        self.dst_weights = {rank: 1.0 / (outdegree + 1)
                            for rank in bf.out_neighbor_ranks()}
        self.self_weight = 1.0 / (outdegree + 1)
        self.force_barrier = True

        named_parameters, models = _check_named_parameters(self, model)
        self._models = models
        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self._handles = {}  # store parameter -> handle
        self._named_ps_weights = {}
        self._named_extension_parameters = {}
        self._synchronized = False
        self._should_synchronize = True
        self._use_timeline = False
        self._error_encountered = False
        self._num_steps_per_communication = num_steps_per_communication
        self._pushsum_delay = {v: self._num_steps_per_communication
                               for _, v in sorted(named_parameters)}
        self._timeline_hook_handles = []
        if bf.size() > 1:
            self._register_window()
            self._register_hooks()

    @torch.no_grad()
    def _register_window(self):
        for param_group in self.param_groups:
            for p in param_group["params"]:
                name = self._parameter_names.get(p)
                if name is None:
                    raise KeyError(
                        "Cannot find parameter {} in the _parameter_names dictionary".format(name))

                ps_weights = torch.Tensor([1.0]).to(p.data.dtype).to(p.data.device)
                self._named_ps_weights[name] = ps_weights
                # If do not modify in the C level, it is inevitable to copy
                # the parameter once in the cat ops.
                extended_parameter = torch.cat((p.data.view(-1), ps_weights), 0)
                self._named_extension_parameters[name] = extended_parameter
                if not bf.win_create(extended_parameter, name, zero_init=True):
                    raise ValueError(
                        "Cannot allocate MPI window for the parameter {}".format(name))

    def _register_hooks(self):
        for model in self._models:
            # The hook is added at model level instead of layer level, as it avoids triggering
            # the hook function of the same layer multiple times in case the layer is called 
            # several times during the forward computation of the model.
            model.register_forward_hook(self._make_hook())

    def _make_hook(self):
        def hook(model, *unused):
            for parent_name, layer in _named_leaf_module(model):
                for name, p in layer.named_parameters():
                    full_name = parent_name+'.'+name
                    if self._use_timeline:
                        # End forward computation timeline
                        bf.timeline_end_activity(full_name)
                    if not layer.training:
                        continue
                    if p.requires_grad:
                        if self._pushsum_delay[p] <= 0:
                            if not self._error_encountered:
                                warnings.warn(_warning_message_num_step_per_communication)
                                self._error_encountered = True
                        self._pushsum_delay[p] -= 1
                        if self._pushsum_delay[p] == 0:
                            ps_weights = self._named_ps_weights[full_name]
                            extended_parameter = torch.cat((p.data.view(-1), ps_weights), 0)
                            self._named_extension_parameters[name] = extended_parameter
                            handle = bf.win_accumulate_nonblocking(
                                tensor=extended_parameter, name=full_name,
                                dst_weights=self.dst_weights,
                                require_mutex=True)
                            self._handles[p] = handle
        return hook

    def turn_on_timeline(self):
        handles = _register_timeline(self, self._models, self._parameter_names)
        self._timeline_hook_handles.extend(handles)
        self._use_timeline = True

    def turn_off_timeline(self):
        for hook in self._timeline_hook_handles:
            hook.remove()
        self._timeline_hook_handles.clear()
        self._use_timeline = False

    @contextmanager
    def skip_synchronize(self):
        """
        A context manager used to specify that optimizer.step() should
        not perform synchronization.

        It's typically used in a following pattern:

        .. code-block:: python

            optimizer.synchronize()
            with optimizer.skip_synchronize():
                optimizer.step()
        """
        self._should_synchronize = False
        try:
            yield
        finally:
            self._should_synchronize = True

    def synchronize(self):
        # Here synchronize just to make sure win_put ops is finished
        # in one iteration.
        with torch.no_grad():
            for p, handle in self._handles.items():
                _ = bf.win_wait(handle)
                name = self._parameter_names.get(p)
                self._pushsum_delay[p] = self._num_steps_per_communication
                extended_parameter = self._named_extension_parameters[name]
                extended_parameter.mul_(self.self_weight)
                # Last dimension is the push_sum weights and we want parameter / weight
                extended_parameter = bf.win_update_then_collect(name=name)
                corrected_parameter = (
                    extended_parameter[:-1] / extended_parameter[-1]).reshape(p.shape)
                # Update p to the average of neighbors.
                p.set_(corrected_parameter)

        self._handles.clear()
        self._synchronized = True

    def step(self, closure=None):
        if self.force_barrier:
            bf.barrier()
        # some validation here?
        if self._should_synchronize:
            if self._synchronized:
                warnings.warn(
                    "optimizer.step() called without "
                    "optimizer.skip_synchronize() context after "
                    "optimizer.synchronize(). This can cause training "
                    "slowdown. You may want to consider using "
                    "optimizer.skip_synchronize() context if you use "
                    "optimizer.synchronize() in your code."
                )
            self.synchronize()

        self._synchronized = False
        return super(self.__class__, self).step(closure)


def DistributedPushSumOptimizer(optimizer, model,
                                num_steps_per_communication=1):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer through
    win_accumulate ops to implement the gradient push algorithm.

    Returned optimizer has two extra parameters `self_weight` and `neighbor_weights`.
    Set self_weight as some scalar and dst_weights dictionary as {rank: scaling} differently
    per iteration to achieve win_put over dynamic graph behavior.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.
        num_steps_per_communication: Number of expected model forward function calls before each
                                     communication. This allows local model parameter updates
                                     per num_steps_per_communication before reducing them over
                                     distributed computation resources.

    Example for two scenarios to use num_steps_per_communication:
        Scenario 1) Local accumulation of gradient without update model.
                    (Used in large batch size or large model cases)
        >>> opt = bf.DistributedPushSumOptimizer(optimizer, model, num_steps_per_communication=J)
        >>> opt.zero_grad()
        >>> for j in range(J):
        >>>     output = model(data_batch_i)
        >>>     loss = ...
        >>>     loss.backward()
        >>> opt.step()  # PushSum reducing happens here
        Scenario 2) Local updating the model. (Used in case that decreasing the communication).
        >>> opt = bf.DistributedPushSumOptimizer(optimizer, model, num_steps_per_communication=J)
        >>> for j in range(J):
        >>>     output = model(data_batch_i)
        >>>     loss = ...
        >>>     opt.zero_grad()
        >>>     loss.backward()
        >>>     opt.step()  # PushSum reducing happens at the last iteration
    """
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedPushSumOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model, num_steps_per_communication)


def DistributedPullGetOptimizer(optimizer, model,
                                num_steps_per_communication=1):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer with
    pull model average through bf.win_get ops.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.
        num_steps_per_communication: Number of expected model forward function calls before each
                                     communication. This allows local model parameter updates
                                     per num_steps_per_communication before reducing them over
                                     distributed computation resources.

    Example for two scenarios to use num_steps_per_communication:
        Scenario 1) Local accumulation of gradient without update model.
                    (Used in large batch size or large model cases)
        >>> opt = bf.DistributedPullGetOptimizer(optimizer, model, num_steps_per_communication=J)
        >>> opt.zero_grad()
        >>> for j in range(J):
        >>>     output = model(data_batch_i)
        >>>     loss = ...
        >>>     loss.backward()
        >>> opt.step()  # PullGet communication happens here
        Scenario 2) Local updating the model. (Used in case that decreasing the communication).
        >>> opt = bf.DistributedPullGetOptimizer(optimizer, model, num_steps_per_communication=J)
        >>> for j in range(J):
        >>>     output = model(data_batch_i)
        >>>     loss = ...
        >>>     opt.zero_grad()
        >>>     loss.backward()
        >>>     opt.step()  # PullGet communication happens at the last iteration

    Returned optimizer has two extra parameters `src_weights` and `force_barrier`.
    Set src_weights dictionary as {rank: scaling} differently per iteration to achieve
    win_get over dynamic graph behavior. If force_barrier is True, a barrier function
    will put at `step()` to synchronous processes.
    """
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedWinOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model, num_steps_per_communication, pull_style=True)


def DistributedWinPutOptimizer(optimizer, model, num_steps_per_communication=1, window_prefix=None):
    """An distributed optimizer that wraps another torch.optim.Optimizer with
    pull model average through bf.win_put ops.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.
        num_steps_per_communication: Number of expected model forward function calls before each
                                     communication. This allows local model parameter updates
                                     per num_steps_per_communication before reducing them over
                                     distributed computation resources.
        window_prefix: A string to identify the unique DistributedWinPutOptimizer, which will be
                       applied as the prefix for window name.

    Returned optimizer has two extra parameters `dst_weights` and `force_barrier`.
    Set dst_weights dictionary as {rank: scaling} differently per iteration to achieve
    win_put over dynamic graph behavior. If force_barrier is True, a barrier function
    will put at `step()` to synchronous processes.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method.
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedWinOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model, num_steps_per_communication,
               window_prefix, pull_style=False)


def DistributedAllreduceOptimizer(optimizer, model,
                                  num_steps_per_communication=1):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer through allreduce ops.
    The communication for allreduce is applied on the parameters when forward propagation happens.

    .. warning::
        This API will be deprecated in the future.
        Use ``DistributedAdaptWithCombineOptimizer`` instead.
    """
    warnings.warn(
        "This API will be deprecated in next version. Please use new equivalent API:\n "
        "  DistributedAdaptWithCombineOptimizer(opt, model, bf.CommunicationType.allreduce)",
        PendingDeprecationWarning)
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with allreduce implementation.
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedReduceOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model,
               CommunicationType.allreduce, num_steps_per_communication)


def DistributedNeighborAllreduceOptimizer(optimizer, model,
                                          num_steps_per_communication=1):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer through
    neighbor_allreduce ops over parameters.

    .. warning::
        This API will be deprecated in the future.
        Use ``DistributedAdaptWithCombineOptimizer`` instead.
    """
    warnings.warn(
        "This API will be deprecated in next version. Please use new equivalent API:\n "
        "  DistributedAdaptWithCombineOptimizer(\n"
        "      opt, model, bf.CommunicationType.neighbor_allreduce)",
        PendingDeprecationWarning)
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with neighbor_allreduce implementation.
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedReduceOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model, CommunicationType.neighbor_allreduce,
               num_steps_per_communication)


def DistributedHierarchicalNeighborAllreduceOptimizer(optimizer, model,
                                                      num_steps_per_communication=1):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer through
    hierarchical_neighbor_allreduce ops over parameters.

    .. warning::
        This API will be deprecated in the future.
        Use ``DistributedAdaptWithCombineOptimizer`` instead.
    """
    warnings.warn(
        "This API will be deprecated in next version. Please use new equivalent API:\n "
        "  DistributedAdaptWithCombineOptimizer(\n"
        "      opt, model, bf.CommunicationType.hierarchical_neighbor_allreduce)",
        PendingDeprecationWarning)
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedReduceOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model, CommunicationType.hierarchical_neighbor_allreduce,
               num_steps_per_communication)


def DistributedGradientAllreduceOptimizer(optimizer, model,
                                          num_steps_per_communication=1):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer through allreduce ops.
    The communication happens when backward propagation happens, which is the same as Horovod.
    In addition, allreduce is applied on gradient instead of parameters.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.
        num_steps_per_communication: Number of expected backward function calls before each
                                     communication. This allows local model parameter updates
                                     per num_steps_per_communication before reducing them over
                                     distributed computation resources.

    Example for two scenarios to use num_steps_per_communication:

        Scenario 1) Local accumulation of gradient without update model.
                    (Used in large batch size or large model cases)

        >>> opt = bf.DistributedGradientAllreduceOptimizer(optimizer, model,
        >>>                                                num_steps_per_communication=J)
        >>> opt.zero_grad()
        >>> for j in range(J):
        >>>     output = model(data_batch_i)
        >>>     loss = ...
        >>>     loss.backward()
        >>> opt.step()  # Allreducing happens here

        Scenario 2) Local updating the model. (Used in case that decreasing the communication).

        >>> opt = bf.DistributedGradientAllreduceOptimizer(optimizer, model,
        >>>                                                num_steps_per_communication=J)
        >>> for j in range(J):
        >>>     output = model(data_batch_i)
        >>>     loss = ...
        >>>     opt.zero_grad()
        >>>     loss.backward()
        >>>     opt.step()  # Allreducing happens at the last iteration
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model, num_steps_per_communication)


def DistributedAdaptThenCombineOptimizer(optimizer, model,
                                         communication_type=CommunicationType.neighbor_allreduce,
                                         num_steps_per_communication=1):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer.
    The communication is applied on the parameters when backward propagation triggered and
    run the communication after parameter updated with gradient.

    In order to maximize the overlapping between communication and computation, we override
    the step() function in standard optimizer provided by PyTorch. Currenly, we support
    SGD, ADAM, AdaDelta, RMSProp, AdaGrad. If you don't use these, you need
    to register your own step function to the returned optimizer through \n

    >>> opt.register_step_function(step_func)

    where the signature should be func(self, parameter, gradient, parameter_group) -> None.
    Note, it has to be paramter-wise and parameter_group is the one that the standard
    torch.optimizer provided, which can store the auxuilary information or state of
    optimizer like learning_rate, weight_decay, etc.

    Returned optimizer has three extra parameters `self_weight`, `neighbor_weights` and
    `send_neighbors`, `neighbor_machine_weights` and `send_neighbor_machines` to control
    the behavior of hierarchical neighbor allreduce. Changing the values
    of these knobs to achieve dynamic topologies.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.
        communication_type: A enum type to determine use neighbor_allreduce, or allreduce, or
            hierarchical_neighbor_allreduce, or empty function as communcaiton behavior.
            Empty function just means no communication.
        num_steps_per_communication: Number of expected backward function calls before each
                                  communication. This allows local model parameter updates
                                  per num_steps_per_communication before reducing them over
                                  distributed computation resources.

    Example for two scenarios to use num_steps_per_communication:

        Scenario 1) Local accumulation of gradient without update model.
                    (Used in large batch size or large model cases)

        >>> opt = bf.DistributedAdaptWithCombineOptimizer(optimizer, model,
        >>>          communication_type=CommunicationType.neighbor_allreduce,
        >>>          num_steps_per_communication=J)
        >>> opt.zero_grad()
        >>> for j in range(J):
        >>>     output = model(data_batch_i)
        >>>     loss = ...
        >>>     loss.backward()
        >>> opt.step()  # Allreducing happens here

        Scenario 2) Local updating the model. (Used in case that decreasing the communication).

        >>> opt = bf.DistributedAdaptWithCombineOptimizer(optimizer, model,
        >>>          communication_type=CommunicationType.neighbor_allreduce,
        >>>          num_steps_per_communication=J)
        >>> for j in range(J):
        >>>     output = model(data_batch_i)
        >>>     loss = ...
        >>>     opt.zero_grad()
        >>>     loss.backward()
        >>>     opt.step()  # Allreducing happens at the last iteration
    """
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedAdaptThenCombineOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model, communication_type, num_steps_per_communication)


def DistributedAdaptWithCombineOptimizer(optimizer, model,
                                         communication_type=CommunicationType.neighbor_allreduce,
                                         num_steps_per_communication=1):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer.
    The communication is applied on the parameters when forward propagation triggered. Hence,
    communication is overlapped with both forward and backward phase. Unlike AdaptThenCombine,
    this dist-optimizer do not need to register customized step function.

    Returned optimizer has three extra parameters `self_weight`, `neighbor_weights` and
    `send_neighbors`, `neighbor_machine_weights` and `send_neighbor_machines` to control
    the behavior of hierarchical neighbor allreduce. Changing the values
    of these knobs to achieve dynamic topologies.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.
        communication_type: A enum type to determine use neighbor_allreduce, or allreduce, or
            hierarchical_neighbor_allreduce, or empty function as communcaiton behavior.
            Empty function just means no communication.
        num_steps_per_communication: Number of expected backward function calls before each
                                     communication. This allows local model parameter updates
                                     per num_steps_per_communication before reducing them over
                                     distributed computation resources.

    Example for two scenarios to use num_steps_per_communication:

        Scenario 1) Local accumulation of gradient without update model.
                    (Used in large batch size or large model cases)

        >>> opt = bf.DistributedAdaptWithCombineOptimizer(optimizer, model,
        >>>          communication_type=CommunicationType.neighbor_allreduce,
        >>>          num_steps_per_communication=J)
        >>> opt.zero_grad()
        >>> for j in range(J):
        >>>     output = model(data_batch_i)
        >>>     loss = ...
        >>>     loss.backward()
        >>> opt.step()  # Allreducing happens here

        Scenario 2) Local updating the model. (Used in case that decreasing the communication).

        >>> opt = bf.DistributedAdaptWithCombineOptimizer(optimizer, model,
        >>>          communication_type=CommunicationType.neighbor_allreduce,
        >>>          num_steps_per_communication=J)
        >>> for j in range(J):
        >>>     output = model(data_batch_i)
        >>>     loss = ...
        >>>     opt.zero_grad()
        >>>     loss.backward()
        >>>     opt.step()  # Allreducing happens at the last iteration
    """
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedReduceOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model, communication_type, num_steps_per_communication)
