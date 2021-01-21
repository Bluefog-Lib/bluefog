.. _FAQ:

FAQ
===

.. contents:: Question Lists
  :local:

Why does the warning related to ``num_steps_per_communication`` pop up in Bluefog optimizer?
--------------------------------------------------------------------------------------------

During your usage of Bluefog distributed optimizer, you may encounter the following
two types of warnings:


.. code-block:: python

   Warning (unexpected behavior):
     After num_steps_per_communication times of forward computation `y=model(x)` are called,
     an optimizer step() function must be called.
     It does not matter how many step() functions are called in between.
     Please adjust num_step_per_communication to update model parameters locally.
     More information can be found in the FAQ page.

.. code-block:: python
   
   Warning (unexpected behavior):
     After num_steps_per_communication times of backward computation `loss.backward()` are called,
     an optimizer step() function must be called.
     It does not matter how many step() functions are called in between.
     Please adjust num_steps_per_communication to accumulate gradients locally.
     More information can be found in the FAQ page.

.. note::
   The second warning is only encountered when using ``DistributedGradientAllreduceOptimizer`` or
   ``DistributedAdaptThenCombineOptimizer``. The difference with other Bluefog optimizer is when
   counting is triggered. All other Bluefog optimizers triggers this counting
   behavior during forward computation, while ``DistributedGradientAllreduceOptimizer`` and 
   ``DistributedAdaptThenCombineOptimizer`` triggers during backward computation.
   This is also reflected by the optimizer argument naming.

To understand the meaning of the above two warnings,
consider the following admissible code snippet for local gradient aggregation case:

.. code-block:: python
   :emphasize-lines: 3,10,13
   
   opt = bf.DistributedAdaptWithCombineOptimizer(
     optimizer, model=model, communication_type=bf.CommunicationType.neighbor_allreduce,
     num_steps_per_communication=J)
   for _ in range(num_epochs):
      for data, target in dataloader:
        opt.zero_grad()
        for i in range(J):
          data_mini_batch = data[i::J] # Make sure batch_size = J * mini_batch_size
          target_mini_batch = target[i::J]
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

If ``num_steps_per_communication`` here doesn't match the number of mini batchs,
the warning will be triggered.
For example, there are 4 mini batches, with a ``num_steps_per_communication`` of 3,
then the calling sequence is **FFFFS**. The warning will be thrown at the fourth **F**,
because after 3 **F**, it should expect an **S** following it.

Let's check another admissible code snippet for multi-round computation per communication case.

.. code-block:: python
   :emphasize-lines: 3,9,13
   
   opt = bf.DistributedAdaptWithCombineOptimizer(
     optimizer, model=model, communication_type=bf.CommunicationType.neighbor_allreduce,
     num_steps_per_communication=J)
   for _ in range(num_epochs):
      for data, target in dataloader:
        for i in range(J):
          data_mini_batch = data[i::J] # Make sure batch_size = J * mini_batch_size
          target_mini_batch = target[i::J]
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

In this case, if ``num_steps_per_communication`` here doesn't match the number of mini batchs,
the situation may be more dangerous, as no warning will be thrown.
For example, there are 4 mini batches, with a ``num_steps_per_communication`` of 3,
then the calling sequence is **FsFsFSFs** for the first batch. The communication is completed at
the third mini batch, and there is one mini batch left in the first batch. For the second batch,
the calling sequence is **FsFSFsFs**. We can see that the communication is finished at the second
mini batch here, due to the left over mini batch in the first batch. No warning is thrown during
this process, since after every 3 **F**, an **S** is followed.
This kind of behavior may not be desired, and users should be careful with this situation.

These are two common usages for ``num_steps_per_communication``  for
Bluefog distributed optimizer. But other usage is also allowed, as long as after
``num_steps_per_communication`` forward computation or  backward propogation, the step function is executed.
With that in mind, some other admissible calling procedures are **FFsFS**, **FsFFS**, etc.
Some inadmissible calling procedures are **FFFFS**, **FFsFFS**.

.. note::
   The previous discussion only applies to computation in the PyTorch computation graph.
   The code inside ``with torch.no_grad()`` or PyTorch tensors with ``requires_grad=False``
   are not counted.