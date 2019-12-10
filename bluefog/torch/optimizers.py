from contextlib import contextmanager
import warnings

import torch
import bluefog.torch as bf

#pylint: disable=unused-argument
class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters):
        super(self.__class__, self).__init__(params)

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = [
                ("allreduce.noname.%s" % i, v)
                for param_group in self.param_groups
                for i, v in enumerate(param_group["params"])
            ]

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError(
                "named_parameters should be a sequence of "
                "tuples (name, parameter), usually produced by "
                "model.named_parameters()."
            )

        dups = _DistributedOptimizer.find_duplicates(
            [k for k, _ in named_parameters])
        if dups:
            raise ValueError(
                "Parameter names in named_parameters must be unique. "
                "Found duplicates: %s" % ", ".join(dups)
            )

        all_param_ids = {
            id(v) for param_group in self.param_groups for v in param_group["params"]
        }
        named_param_ids = {id(v) for k, v in named_parameters}
        unnamed_param_ids = all_param_ids - named_param_ids
        if unnamed_param_ids:
            raise ValueError(
                "named_parameters was specified, but one or more model "
                "parameters were not named. Python object ids: "
                "%s" % ", ".join(str(id) for id in unnamed_param_ids)
            )

        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._synchronized = False
        self._should_synchronize = True
        if bf.size() > 1:
            self._register_hooks()

    @staticmethod
    def find_duplicates(lst):
        seen = set()
        dups = set()
        for el in lst:
            if el in seen:
                dups.add(el)
            seen.add(el)
        return dups

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

        handle = bf.allreduce_async(
            p.grad, average=True, name=name
        )
        return handle

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


class _DistributedConsensusOptimizer(torch.optim.Optimizer):
    """ A distributed optimizer wrapper over torch optimizer.

    Note: Unlike the _DistributedOptimizer class that registers hook for each named parameters,
    triggers the allreduce_async after gradient computation is finished, and updates the
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

    def __init__(self, params, named_parameters):
        super(self.__class__, self).__init__(params)

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = [
                ("allreduce.noname.%s" % i, v)
                for param_group in self.param_groups
                for i, v in enumerate(param_group["params"])
            ]

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError(
                "named_parameters should be a sequence of "
                "tuples (name, parameter), usually produced by "
                "model.named_parameters()."
            )

        dups = _DistributedOptimizer.find_duplicates(
            [k for k, _ in named_parameters])
        if dups > 0:
            raise ValueError(
                "Parameter names in named_parameters must be unique. "
                "Found duplicates: %s" % ", ".join(dups)
            )

        all_param_ids = {
            id(v) for param_group in self.param_groups for v in param_group["params"]
        }
        named_param_ids = {id(v) for k, v in named_parameters}
        unnamed_param_ids = all_param_ids - named_param_ids
        if unnamed_param_ids:
            raise ValueError(
                "named_parameters was specified, but one or more model "
                "parameters were not named. Python object ids: "
                "%s" % ", ".join(str(id) for id in unnamed_param_ids)
            )

        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._synchronized = False
        self._should_synchronize = True
        if bf.size() > 1:
            self._register_hooks()

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group["params"]:
                if p.requires_grad:
                    p.register_hook(self._make_hook(p))

    def _make_hook(self, p):
        def hook(*ignore):
            assert not p.grad.requires_grad
            handle = self._neighbor_allreduce_data_async(p)
            self._handles[p] = handle

        return hook

    def _neighbor_allreduce_data_async(self, p):
        name = self._parameter_names.get(p)
        handle = bf.neighbor_allreduce_async(p.data, average=True, name=name)
        return handle

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

    def __init__(self, params, named_parameters):
        super(self.__class__, self).__init__(params)

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = [
                ("win.put.noname.%s" % i, v)
                for param_group in self.param_groups
                for i, v in enumerate(param_group["params"])
            ]

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError(
                "named_parameters should be a sequence of "
                "tuples (name, parameter), usually produced by "
                "model.named_parameters()."
            )

        dups = _DistributedOptimizer.find_duplicates(
            [k for k, _ in named_parameters])
        if dups:
            raise ValueError(
                "Parameter names in named_parameters must be unique. "
                "Found duplicates: %s" % ", ".join(dups)
            )

        all_param_ids = {
            id(v) for param_group in self.param_groups for v in param_group["params"]
        }
        named_param_ids = {id(v) for k, v in named_parameters}
        unnamed_param_ids = all_param_ids - named_param_ids
        if unnamed_param_ids:
            raise ValueError(
                "named_parameters was specified, but one or more model "
                "parameters were not named. Python object ids: "
                "%s" % ", ".join(str(id) for id in unnamed_param_ids)
            )

        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._synchronized = False
        self._should_synchronize = True
        if bf.size() > 1:
            self._register_window()
            self._register_hooks()

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group["params"]:  # Is hook function blocking or not?
                if p.requires_grad:
                    p.register_hook(self._make_hook(p))

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

    def _make_hook(self, p):
        def hook(*ignore):
            assert not p.grad.requires_grad
            name = self._parameter_names.get(p)
            handle = bf.win_put(tensor=p.data, name=name)
            self._handles[p] = handle
        return hook

    def _win_put_async(self, p):
        name = self._parameter_names.get(p)
        handle = bf.win_put(tensor=p.data, name=name)
        return handle

    def synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle = self._win_put_async(p)
            self._handles[p] = handle

        for p, handle in self._handles.items():
            if handle is None:
                handle = self._win_put_async(p)
                self._handles[p] = handle

        # Here synchronize just to make sure win_put ops is finished
        # in one iteration.
        with torch.no_grad():
            for p, handle in self._handles.items():
                _ = bf.win_wait(handle)
                name = self._parameter_names.get(p)
                # Update p to the average of neighbors.
                p.set_(bf.win_sync(name=name))

        self._handles.clear()
        self._synchronized = True

    def step(self, closure=None):
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


def DistributedBluefogOptimizer(optimizer, named_parameters=None):
    """
    An consensus optimizer that wraps another torch.optim.Optimizer.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          window operations. Typically just ``model.named_parameters()``

    Example:
        >>> import bluefog.torch as bf
        >>> ...
        >>> bf.init()
        >>> optimizer = optim.SGD(model.parameters(), lr=lr * bf.size())
        >>> optimizer = bf.DistributedBluefogOptimizer(
        ...    optimizer, named_parameters=model.named_parameters()
        ... )
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedBluefogOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, named_parameters)


def DistributedConsensusOptimizer(optimizer, named_parameters=None):
    """
    An consensus optimizer that wraps another torch.optim.Optimizer.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just ``model.named_parameters()``
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedConsensusOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, named_parameters)


def DistributedOptimizer(optimizer, named_parameters=None):
    """
    An optimizer that wraps another torch.optim.Optimizer.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just ``model.named_parameters()``
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, named_parameters)
