.. _FAQ:

FAQ
===

.. contents:: Question Lists
  :local:

Why does the warning related to ``num_communication_per_step`` or ``backward_passes_per_step`` pop up in Bluefog optimizer?
--------------------------------------------------------------------------------------------

During your usage of Bluefog optimizer, you may encounter the following the two types of warnings.


.. code-block:: python

   Warning (unexpected behavior):
     After num_step_per_communication times of forward computation `y=model(x)` are called,
     an optimizer step() function must be called.
     It does not matter how many step() functions are called in between.
     Please adjust num_step_per_communication to update model parameters locally.
     More information can be found in the FAQ page.
.. code-block:: python
   
   Warning (unexpected behavior):
     After backward_passes_per_step times of backward computation `loss.backward()` are called,
     an optimizer step() function must be called.
     It does not matter how many step() functions are called in between.
     Please adjust backward_passes_per_step to accumulate gradients locally.
     More information can be found in the FAQ page.

.. note::
   The second warning is only encountered when using ``DistributedGradientAllreduceOptimizer`` or
   ``DistributedAdaptThenCombineOptimizer``. The difference with other Bluefog optimizer is when
   counting is triggered. All other Bluefog optimizers triggers this counting
   behavior during forward computation, while ``DistributedGradientAllreduceOptimizer`` and 
   ``DistributedAdaptThenCombineOptimizer`` triggers during backward computation.
   This is also reflected by the optimizer argument naming.
   All other optimizers uses ``num_step_per_communication``, while these two optimizers uses
   ``backward_passes_per_step``.
   The following discussion only focuses on forward computation case, but it works for backward
   scenario as well.

Consider the following admissible code snippet.

.. code-block:: python
   :emphasize-lines: 3,10,13
   
   opt = bf.DistributedAdaptWithCombineOptimizer(
     optimizer, model=model, communication_type=bf.CommunicationType.neighbor_allreduce,
     num_steps_per_communication=J)
   for _ in range(num_epochs):
      for data, target in dataloader:
        opt.zero_grad()
        for i in range(J):
          data_mini_batch = data[...]
          target_mini_batch = target[...]
          y = model(data_mini_batch) # Forward Computation <- Counting occurs.
          loss = mseloss(y, target_mini_batch)
          loss.backward()
        opt.step() # Step Function <- Communication occurs and counting is reset.

We use **F** to denote forward computation ``y=model(data_mini_batch)``,
and use **S** to denote step function ``opt.step()`` with communication occurrence.
For example, let's say ``J`` is 3. 
For each batch, the calling sequence is **FFFS**, ignoring other operations.
We can see that step function happens right after ``J=3`` forward computations.
With that, we accumulate the update for the model parameters in all ``J=3`` mini batches locally,
and reduce the model parameters during the step function in a AWC style.

Let's check another admissible code snippet.

.. code-block:: python
   :emphasize-lines: 3,9,13
   
   opt = bf.DistributedAdaptWithCombineOptimizer(
     optimizer, model=model, communication_type=bf.CommunicationType.neighbor_allreduce,
     num_steps_per_communication=J)
   for _ in range(num_epochs):
      for data, target in dataloader:
        for i in range(J):
          data_mini_batch = data[...]
          target_mini_batch = target[...]
          y = model(data_mini_batch) # Forward Computation <- Counting occurs.
          loss = mseloss(y, target_mini_batch)
          opt.zero_grad()
          loss.backward()
          opt.step() # Step Function <- Communication occurs at Jth iteration and counting is reset.

Similar as before, the calling sequence is **FsFsFS**.
Here **s** stands for the step function without communication (reduce) occurrence.
After ``J=3`` forward computations happen, a step function called triggers communication.
However, all other step functions in between, denoted by **s**, doesn't trigger communication.
With that, we update the model locally after each mini batch; and at the last mini batch,
the model parameters are reduced with its neighbors.

These are two common usages for ``num_communication_per_step`` or ``backward_passes_per_step`` for
Bluefog optimizer. But other usage is also allowed, as long as after ``num_communication_per_step``
forward computation or ``backward_passes_per_step`` backward propogation, the step function is 
executed. With that in mind, some other admissible calling procedures are **FFsFS**, **FsFFS**, etc.
Some inadmissible calling procedures are **FFFFS**, **FFsFFS**.

.. note::
   The previous discussion only applies to computation in the PyTorch computation graph.
   The code inside ``with torch.no_grad()`` or PyTorch tensors with ``requires_grad=False``
   are not counted.