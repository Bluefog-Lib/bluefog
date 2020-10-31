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
import itertools
import os
import warnings

import torch
import bluefog.torch as bf

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
                raise AssertionError(
                    "Unexpected behavior: backward computation were computed "
                    "more than num_steps_per_communication times before call "
                    "to step(). Adjust num_steps_per_communication to "
                    "accumulate gradients locally.")
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

    def __init__(self, params, model, reduce_type, num_steps_per_communication=1):
        super(self.__class__, self).__init__(params)

        named_parameters, models = _check_named_parameters(self, model)
        # knobs for neighbor communication behavior
        self.self_weight = None
        self.neighbor_weights = None
        self.send_neighbors = None
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
        self._num_steps_per_communication = num_steps_per_communication
        self._reduce_type_str = reduce_type
        # _reduce_method: 0 for allreduce, and 1 for neighbor_allreduce
        if self._reduce_type_str == "allreduce":
            self._reduce_method = 0
        elif self._reduce_type_str == "neighbor.allreduce":
            self._reduce_method = 1
        elif self._reduce_type_str == "hierarchical.neighbor.allreduce":
            self._reduce_method = 2
        else:
            raise ValueError("Unknown reduce type for internal class _DistributedReduceOptimizer")

        self._reduce_delay = {v: self._num_steps_per_communication
                              for _, v in sorted(named_parameters)}
        if os.getenv('BLUEFOG_TIMELINE'):
            self.turn_on_timeline()
        if bf.size() > 1:
            self._register_hooks()

    def _register_hooks(self):
        for model in self._models:
            for parent_name, layer in _named_leaf_module(model):
                layer.register_forward_hook(self._make_hook(parent_name))
                for _, p in layer.named_parameters():
                    self._requires_update.add(p)

    def _make_hook(self, parent_name):
        def hook(module, *unused):
            for name, p in module.named_parameters():
                if not module.training:
                    continue
                if self._use_timeline:
                    # End forward computation timeline
                    bf.timeline_end_activity(parent_name+'.'+name)
                if p.requires_grad:
                    if self._reduce_delay[p] <= 0:
                        raise AssertionError(
                            "Unexpected behavior: forward computation were computed "
                            "more than num_steps_per_communication times before call "
                            "to step(). Adjust num_steps_per_communication to "
                            "accumulate gradients locally.")
                    self._reduce_delay[p] -= 1
                    if self._reduce_delay[p] == 0:
                        if self._reduce_method == 0:
                            handle = self._allreduce_data_async(p)
                        elif self._reduce_method == 1:
                            handle = self._neighbor_allreduce_data_async(p)
                        elif self._reduce_method == 2:
                            handle = self._hierarchical_neighbor_allreduce_data_async(p)
                        elif self._reduce_method == -1:
                            handle = None
                        else:
                            raise ValueError(
                                "Unknown reduce method. Do not change _reduce_method manually.")
                        self._handles[p] = handle
        return hook

    def _neighbor_allreduce_data_async(self, p):
        name = self._parameter_names.get(p)
        handle = bf.neighbor_allreduce_nonblocking(p.data, name=name, self_weight=self.self_weight,
                                                   neighbor_weights=self.neighbor_weights,
                                                   send_neighbors=self.send_neighbors,
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
            self, self._models, self._parameter_names, self._reduce_type_str)
        self._timeline_hook_handles.extend(handles)
        self._use_timeline = True

    def turn_off_timeline(self):
        for hook in self._timeline_hook_handles:
            hook.remove()
        self._timeline_hook_handles.clear()
        self._use_timeline = False

    def use_allreduce_in_communication(self):
        self._reduce_method = 0

    def use_neighbor_allreduce_in_communication(self):
        self._reduce_method = 1

    def use_hierarchical_neighbor_allreduce_in_communication(self):
        self._reduce_method = 2

    def use_empty_function_in_communication(self):
        self._reduce_method = -1

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


class _DistributedWinOptimizer(torch.optim.Optimizer):

    def __init__(self, params, model, num_steps_per_communication, pull_style):
        super(self.__class__, self).__init__(params)

        if pull_style:
            self.src_weights = None # use to control the behavior of win_get dynamically.
        else:
            self.dst_weights = None # use to control the behavior of win_put dynamically.
        self.force_barrier = False

        named_parameters, models = _check_named_parameters(self, model)
        self._models = models
        self._pull_style = pull_style
        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self._handles = {}  # store parameter -> handle
        self._synchronized = False
        self._should_synchronize = True
        self._use_timeline = False
        self._num_steps_per_communication = num_steps_per_communication
        self._bluefog_delay = {v: self._num_steps_per_communication
                               for _, v in sorted(named_parameters)}
        self._timeline_hook_handles = []
        if os.getenv('BLUEFOG_TIMELINE'):
            self.turn_on_timeline()
        if bf.size() > 1:
            self._register_window()
            self._register_hooks()

    def _register_hooks(self):
        for model in self._models:
            for parent_name, layer in _named_leaf_module(model):
                if self._pull_style:
                    hook = self._make_get_hook(parent_name)
                else:
                    hook = self._make_put_hook(parent_name)
                layer.register_forward_hook(hook)

    def _make_put_hook(self, parent_name):
        def hook(module, *unused):
            for name, p in module.named_parameters():
                if self._use_timeline:
                    # End forward computation timeline
                    bf.timeline_end_activity(parent_name+'.'+name)
                if not module.training:
                    continue
                if p.requires_grad:
                    if self._bluefog_delay[p] <= 0:
                        raise AssertionError(
                            "Unexpected behavior: forward computation were computed "
                            "more than num_steps_per_communication times before call "
                            "to step(). Adjust num_steps_per_communication to "
                            "accumulate gradients locally.")
                    self._bluefog_delay[p] -= 1
                    if self._bluefog_delay[p] == 0:
                        handle = bf.win_put_nonblocking(
                            tensor=p.data, name=parent_name+'.'+name,
                            dst_weights=self.dst_weights, require_mutex=False)
                        self._handles[p] = handle
        return hook

    def _make_get_hook(self, parent_name):
        def hook(module, *unused):
            for name, p in module.named_parameters():
                if self._use_timeline:
                    # End forward computation timeline
                    bf.timeline_end_activity(parent_name+'.'+name)
                if not module.training:
                    continue
                if p.requires_grad:
                    if self._bluefog_delay[p] <= 0:
                        raise AssertionError(
                            "Unexpected behavior: forward computation were computed "
                            "more than num_steps_per_communication times before call "
                            "to step(). Adjust num_steps_per_communication to "
                            "accumulate gradients locally.")
                    self._bluefog_delay[p] -= 1
                    if self._bluefog_delay[p] == 0:
                        handle = bf.win_get_nonblocking(
                            name=parent_name+'.'+name, src_weights=self.src_weights,
                            require_mutex=True)
                        self._handles[p] = handle
        return hook

    def _register_window(self):
        for param_group in self.param_groups:
            for p in param_group["params"]:
                name = self._parameter_names.get(p)
                if name is None:
                    raise KeyError(
                        "Cannot find parameter {} in the _parameter_names dictionary".format(name))
                if not bf.win_create(p.data, name):
                    raise ValueError(
                        "Cannot allocate MPI window for the parameter {}".format(name))

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
            for parent_name, layer in _named_leaf_module(model):
                layer.register_forward_hook(self._make_hook(parent_name))

    def _make_hook(self, parent_name):
        def hook(module, *unused):
            for name, p in module.named_parameters():
                full_name = parent_name+'.'+name
                if self._use_timeline:
                    # End forward computation timeline
                    bf.timeline_end_activity(full_name)
                if not module.training:
                    continue
                if p.requires_grad:
                    if self._pushsum_delay[p] <= 0:
                        raise AssertionError(
                            "Unexpected behavior: forward computation were computed "
                            "more than num_steps_per_communication times before call "
                            "to step(). Adjust num_steps_per_communication to "
                            "accumulate gradients locally.")
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


def DistributedWinPutOptimizer(optimizer, model,
                               num_steps_per_communication=1):
    """An distributed optimizer that wraps another torch.optim.Optimizer with
    pull model average through bf.win_put ops.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.
        num_steps_per_communication: Number of expected model forward function calls before each
                                     communication. This allows local model parameter updates
                                     per num_steps_per_communication before reducing them over
                                     distributed computation resources.

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
    return cls(optimizer.param_groups, model, num_steps_per_communication, pull_style=False)


def DistributedAllreduceOptimizer(optimizer, model,
                                  num_steps_per_communication=1):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer through allreduce ops.
    The communication for allreduce is applied on the parameters when forward propagation happens.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.
        num_steps_per_communication: Number of expected model forward function calls before each
                                     communication. This allows local model parameter updates
                                     per num_steps_per_communication before reducing them over
                                     distributed computation resources.

    Example for two scenarios to use num_steps_per_communication.
        Scenario 1) Local accumulation of gradient without update model.
                    (Used in large batch size or large model cases)

        >>> opt = bf.DistributedAllreduceOptimizer(optimizer, model,
                                                   num_steps_per_communication=J)
        >>> opt.zero_grad()
        >>> for j in range(J):
        >>>     output = model(data_batch_i)
        >>>     loss = ...
        >>>     loss.backward()
        >>> opt.step()  # Allreducing happens here

        Scenario 2) Local updating the model. (Used in case that decreasing the communication).

        >>> opt = bf.DistributedAllreduceOptimizer(optimizer, model,
                                                   num_steps_per_communication=J)
        >>> for j in range(J):
        >>>     output = model(data_batch_i)
        >>>     loss = ...
        >>>     opt.zero_grad()
        >>>     loss.backward()
        >>>     opt.step()  # Allreducing happens at the last iteration
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with allreduce implementation.
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedReduceOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model, "allreduce", num_steps_per_communication)


def DistributedNeighborAllreduceOptimizer(optimizer, model,
                                          num_steps_per_communication=1):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer through
    neighbor_allreduce ops over parameters.

    Returned optimizer has three extra parameters `self_weight`, `neighbor_weights` and
    `send_neighbors` to control the behavior of neighbor allreduce. Changing the values
    of these knobs to achieve dynamic topologies.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.
        num_steps_per_communication: Number of expected model forward function calls before each
                                     communication. This allows local model parameter updates
                                     per num_steps_per_communication before reducing them over
                                     distributed computation resources.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with neighbor_allreduce implementation.
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedReduceOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model, "neighbor.allreduce", num_steps_per_communication)


def DistributedHierarchicalNeighborAllreduceOptimizer(optimizer, model,
                                                      num_steps_per_communication=1):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer through
    hierarchical_neighbor_allreduce ops over parameters.

    Returned optimizer has three extra parameters `self_weight`, `neighbor_machine_weights` and
    `send_neighbor_machines` to control the behavior of hierarchical neighbor allreduce. Changing
    the values of these knobs to achieve dynamic topologies.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.
        num_steps_per_communication: Number of expected model forward function calls before each
                                     communication. This allows local model parameter updates
                                     per num_steps_per_communication before reducing them over
                                     distributed computation resources.

    Warning:
        The processes within the same machine should provide the same `neighbor_machine_weights` and
        `send_neighbor_machines` to avoid unexpected behavior.

    Example for two scenarios to use num_steps_per_communication:
        Scenario 1) Local accumulation of gradient without update model.
                    (Used in large batch size or large model cases)

        >>> opt = bf.DistributedHierarchicalNeighborAllreduceOptimizer(
                        optimizer, model, num_steps_per_communication=J)
        >>> opt.zero_grad()
        >>> for j in range(J):
        >>>     output = model(data_batch_i)
        >>>     loss = ...
        >>>     loss.backward()
        >>> opt.step()  # Neighbor allreducing happens here

        Scenario 2) Local updating the model. (Used in case that decreasing the communication).

        >>> opt = bf.DistributedHierarchicalNeighborAllreduceOptimizer(
                        optimizer, model, num_steps_per_communication=J)
        >>> for j in range(J):
        >>>     output = model(data_batch_i)
        >>>     loss = ...
        >>>     opt.zero_grad()
        >>>     loss.backward()
        >>>     opt.step()  # Neighbor allreducing happens at the last iteration
    """
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedReduceOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model, "hierarchical.neighbor.allreduce",
               num_steps_per_communication)


def DistributedGradientAllreduceOptimizer(optimizer, model,
                                          backward_passes_per_step=1):
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
    return cls(optimizer.param_groups, model, backward_passes_per_step)
