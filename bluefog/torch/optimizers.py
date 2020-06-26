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
    def __init__(self, params, model):
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


class _DistributedNeighborAllreduceOptimizer(torch.optim.Optimizer):
    """ A distributed optimizer wrapper over torch optimizer.

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

    def __init__(self, params, model):
        super(self.__class__, self).__init__(params)

        named_parameters, models = _check_named_parameters(self, model)
        self.neighbor_weights = None
        self.self_weight = None
        self._models = models
        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self._handles = {}
        self._requires_update = set()
        self._synchronized = False
        self._should_synchronize = True
        self._timeline_hook_handles = []
        self._use_timeline = False
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
                    handle = self._neighbor_allreduce_data_async(p)
                    self._handles[p] = handle
        return hook

    def _neighbor_allreduce_data_async(self, p):
        name = self._parameter_names.get(p)
        handle = bf.neighbor_allreduce_nonblocking(p.data, name=name, self_weight=self.self_weight,
                                                   neighbor_weights=self.neighbor_weights)
        return handle

    def turn_on_timeline(self):
        handles = _register_timeline(
            self, self._models, self._parameter_names, 'neighbor.allreduce')
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
            handle = self._neighbor_allreduce_data_async(p)
            self._handles[p] = handle

        for p, handle in self._handles.items():
            if handle is None:
                handle = self._neighbor_allreduce_data_async(p)
                self._handles[p] = handle

        with torch.no_grad():
            for p, handle in self._handles.items():
                output = bf.synchronize(handle)
                p.set_(output)
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


class _DistributedBluefogOptimizer(torch.optim.Optimizer):

    def __init__(self, params, model, pull_style):
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

    def __init__(self, params, model):
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


def DistributedPushSumOptimizer(optimizer, model):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer through
    win_accumulate ops to implement the gradient push algorithm.

    Returned optimizer has two extra parameters `self_weight` and `neighbor_weights`.
    Set self_weight as some scalar and dst_weights dictionary as {rank: scaling} differently
    per iteration to achieve win_put over dynamic graph behavior.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.
    """
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedPushSumOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model)

def DistributedPullGetOptimizer(optimizer, model):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer with
    pull model average through bf.win_get ops.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.

    Returned optimizer has two extra parameters `src_weights` and `force_barrier`.
    Set src_weights dictionary as {rank: scaling} differently per iteration to achieve
    win_get over dynamic graph behavior. If force_barrier is True, a barrier function
    will put at `step()` to synchronous processes.
    """
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedBluefogOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model, pull_style=True)


def DistributedBluefogOptimizer(optimizer, model):
    """An distributed optimizer that wraps another torch.optim.Optimizer with
    pull model average through bf.win_put ops.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.

    Returned optimizer has two extra parameters `dst_weights` and `force_barrier`.
    Set dst_weights dictionary as {rank: scaling} differently per iteration to achieve
    win_put over dynamic graph behavior. If force_barrier is True, a barrier function
    will put at `step()` to synchronous processes.

    Example:
        >>> import bluefog.torch as bf
        >>> ...
        >>> bf.init()
        >>> optimizer = optim.SGD(model.parameters(), lr=lr * bf.size())
        >>> optimizer = bf.DistributedBluefogOptimizer(optimizer, model)
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method.
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedBluefogOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model, pull_style=False)


def DistributedNeighborAllreduceOptimizer(optimizer, model):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer through
    neighbor_allreduce ops.

    Returned optimizer has two extra parameters `self_weight` and `neighbor_weights`.
    Set self_weight as some scalar and dst_weights dictionary as {rank: scaling} differently
    per iteration to achieve win_put over dynamic graph behavior.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with neighbor_allreduce implementation.
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedNeighborAllreduceOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model)


def DistributedAllreduceOptimizer(optimizer, model):
    """
    An distributed optimizer that wraps another torch.optim.Optimizer through allreduce ops.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        model: The model or a list of models you want to train with.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model)
