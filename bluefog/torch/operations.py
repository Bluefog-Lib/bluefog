import time

import numpy
import torch
from typing import List

import bluefog.torch as bf
from bluefog.common.common import TensorTableEntry, WindowTableEntry, DeviceType, Status
from bluefog.common.operations import (
    EnqueueTensorAllReduce,
    EnqueueTensorBroadcast,
    EnqueueTensorAllGather,
    EnqueueTensorNeighborAllGather,
    EnqueueTensorNeighborAllReduce,
    WindowCreate,
    WindowFree,
    WindowSync,
    EnqueueTensorWindowPut,
    EnqueueTensorWindowGet
)
from bluefog.torch.handle_manager import HandleManager

# TODO(ybc) Check when mpi4py would enable support the CUPY.
from bluefog.torch.torch_to_mpi_format import to_mpi4py_aware_format, SUPPORT_CUPY


def EnqueueTorchTensorAllGather(tensor: torch.Tensor, output: torch.Tensor, name: str) -> int:
    # Find the sum of first dimension first. Assuming the rest dimensions are the same.
    # TODO(ybc) find a way to check the rest dimensions.
    gathered_first_dimension = bf.allreduce(
        torch.Tensor([tensor.shape[0]]), average=False)
    output_size = [int(gathered_first_dimension.data[0])]
    output_size.extend(tensor.shape[1:])
    output.resize_(output_size)
    output.zero_()
    if tensor.is_cuda:
        if not SUPPORT_CUPY:
            raise EnvironmentError("cupy is not installed.")
        _tensor = tensor.cpu().numpy()
        _output = output.cpu().numpy()
    else:
        _tensor = tensor.numpy()
        _output = output.numpy()

    handle = HandleManager.AllocateHandle()

    def GenerateCallback(handle, _output, output):
        def callback():
            # If mpi4py supports CUDA-AWARE MPI, we will be able to avoid this.
            if output.is_cuda:
                with torch.no_grad():
                    # TODO(ybc): copy_ vs set_ vs add_.
                    output.zero_()
                    output.add_(torch.as_tensor(_output, device=output.device))
            HandleManager.MarkDone(handle)

        return callback

    tensorname = "{name}_allgather_{handle}".format(name=name, handle=handle)
    tensor_table_entry = TensorTableEntry(
        tensor_name=tensorname,
        tensor_type=_tensor.dtype,
        tensor=_tensor,
        output=_output,
        root_rank=0,
        device=DeviceType.GPU if tensor.is_cuda else DeviceType.CPU,
        callback=GenerateCallback(handle, _output, output),
    )

    EnqueueTensorAllGather(tensor_table_entry)
    return handle


def EnqueueTorchTensorBoardcast(tensor: torch.Tensor, output: torch.Tensor,
                                root_rank: int, name: str) -> int:
    if tensor.is_cuda:
        if not SUPPORT_CUPY:
            raise EnvironmentError("cupy is not installed.")
        _tensor = tensor.cpu().numpy()
        _output = output.cpu().numpy()
    else:
        _tensor = tensor.numpy()
        _output = output.numpy()
        if bf.size() == root_rank:
            if id(_output) != id(_tensor):
                numpy.copyto(dst=_output, src=_tensor)

    handle = HandleManager.AllocateHandle()

    def GenerateCallback(handle, _output, output):
        def callback():
            # If mpi4py supports CUDA-AWARE MPI, we will be able to avoid this.
            if output.is_cuda:
                with torch.no_grad():
                    output.zero_()
                    output.add_(torch.as_tensor(_output, device=output.device))
            HandleManager.MarkDone(handle)

        return callback

    tensorname = "{name}_boardcast_{handle}".format(name=name, handle=handle)
    tensor_table_entry = TensorTableEntry(
        tensor_name=tensorname,
        tensor_type=_tensor.dtype,
        tensor=_tensor,
        output=_output,
        root_rank=root_rank,
        device=DeviceType.GPU if tensor.is_cuda else DeviceType.CPU,
        callback=GenerateCallback(handle, _output, output),
    )

    EnqueueTensorBroadcast(tensor_table_entry)
    return handle


def EnqueueTorchTensorAllReduce(tensor: torch.Tensor, output: torch.Tensor,
                                average: bool, name: str) -> int:
    # Here we use the torch tensor share same memory with numpy.
    # When we modified the value of numpy, the tensor is updated as well.
    # TODO(ybc) find a better to interact between numpy and torch tensor?
    if tensor.is_cuda:
        if not SUPPORT_CUPY:
            raise EnvironmentError("cupy is not installed.")
        _tensor = tensor.cpu().numpy()
        _output = output.cpu().numpy()
    else:
        _tensor = tensor.numpy()
        _output = output.numpy()

    handle = HandleManager.AllocateHandle()

    def GenerateCallback(handle, average, _output, output):
        def callback():
            # If mpi4py supports CUDA-AWARE MPI, we will be able to avoid this.
            if output.is_cuda:
                with torch.no_grad():
                    output.zero_()
                    output.add_(torch.as_tensor(_output, device=output.device))
            if average:
                output.div_(float(bf.size()))
            HandleManager.MarkDone(handle)

        return callback

    tensorname = "{name}_allreduce_{handle}".format(name=name, handle=handle)
    tensor_table_entry = TensorTableEntry(
        tensor_name=tensorname,
        tensor_type=_tensor.dtype,
        tensor=_tensor,
        output=_output,
        root_rank=-1,
        device=DeviceType.GPU if tensor.is_cuda else DeviceType.CPU,
        callback=GenerateCallback(handle, average, _output, output),
    )

    EnqueueTensorAllReduce(tensor_table_entry)
    return handle


def EnqueueTorchTensorNeighborAllReduce(tensor: torch.Tensor, output: torch.Tensor,
                                        average: bool, name: str) -> int:
    if tensor.is_cuda:
        if not SUPPORT_CUPY:
            raise EnvironmentError("cupy is not installed.")
        _tensor = tensor.cpu().numpy()
        _output = numpy.zeros_like(_tensor)  # Will re-allocate memeory later.
    else:
        _tensor = tensor.numpy()
        _output = numpy.zeros_like(_tensor)

    handle = HandleManager.AllocateHandle()

    def GenerateCallback(handle, average, tensor, output):
        def callback(recv_output: numpy.ndarray, neighbor_count: int):
            """ Generate the neighbor_reduced vector into torch.
            Because MPI only provided the neighbor_allgather routine, we have to manually
            reduce the all gathered into one tensor. Suppose the input tensor has shape --
            [d1, d2, ..., dn], then recv_output should be [d1*n, d2, .., dn], where n is the
            number of in-degree, i.e. number of recv_from_neighbors.

            Args:
                recv_output (numpy.ndarray): received numpy array from mpi neighbor_allgather.
                neighbor_count (int): The number of neighbor.
            """
            # If mpi4py supports CUDA-AWARE MPI, we will be able to avoid this.
            with torch.no_grad():
                output.zero_()
                output.add_(tensor)
                for i in range(neighbor_count):
                    s, e = output.shape[0]*i, output.shape[0]*(i+1)
                    output.add_(torch.as_tensor(
                        recv_output[s:e], device=output.device))

            if average:
                output.div_(float(neighbor_count + 1))
            HandleManager.MarkDone(handle)

        return callback

    tensorname = "{name}_neighbor_allreduce_{handle}".format(
        name=name, handle=handle)
    tensor_table_entry = TensorTableEntry(
        tensor_name=tensorname,
        tensor_type=_tensor.dtype,
        tensor=_tensor,
        output=_output,
        root_rank=-1,
        device=DeviceType.GPU if tensor.is_cuda else DeviceType.CPU,
        callback=GenerateCallback(handle, average, tensor, output),
    )
    EnqueueTensorNeighborAllReduce(tensor_table_entry)
    return handle


def EnqueueTorchTensorNeighborAllGather(tensor: torch.Tensor, output: torch.Tensor,
                                        name: str) -> int:
    # Pitfall: neighbor_allreduce include itself, while mpi_neighbor_allgather do not!
    gathered_first_dimension = bf.neighbor_allreduce(
        torch.Tensor([tensor.shape[0]]), average=False, name="pre-nallgather-nallreduce")  
    output_size = [int(gathered_first_dimension.data[0] - tensor.shape[0])]
    output_size.extend(tensor.shape[1:])
    output.resize_(output_size)
    output.zero_()
    if tensor.is_cuda:
        if not SUPPORT_CUPY:
            raise EnvironmentError("cupy is not installed.")
        _tensor = tensor.cpu().numpy()
        _output = output.cpu().numpy()
    else:
        _tensor = tensor.numpy()
        _output = output.numpy()

    handle = HandleManager.AllocateHandle()

    def GenerateCallback(handle, _output, output):
        def callback():
            # If mpi4py supports CUDA-AWARE MPI, we will be able to avoid this.
            if output.is_cuda:
                with torch.no_grad():
                    output.zero_()
                    output.add_(torch.as_tensor(_output, device=output.device))
            HandleManager.MarkDone(handle)

        return callback

    tensorname = "{name}_neighbor_allgather_{handle}".format(name=name, handle=handle)
    tensor_table_entry = TensorTableEntry(
        tensor_name=tensorname,
        tensor_type=_tensor.dtype,
        tensor=_tensor,
        output=_output,
        root_rank=0,
        device=DeviceType.GPU if tensor.is_cuda else DeviceType.CPU,
        callback=GenerateCallback(handle, _output, output),
    )

    EnqueueTensorNeighborAllGather(tensor_table_entry)
    return handle


def torch_poll(handle: int) -> bool:
    return HandleManager.PollHandle(handle)


def torch_wait_and_clear(handle: int) -> bool:
    while not torch_poll(handle):
        time.sleep(0.001)
    status = HandleManager.ReleaseHandle(handle)
    if status != Status.OK:
        raise ValueError(
            "MPI ops is not finished correcttly after calling wait_and_clear."
        )


# TODO(ybc) Definitely need a better way to deal with it.
TORCH_TYPE_MAP = {
    torch.float32: (numpy.float32, 1),
    torch.int: (numpy.int, 1),
    torch.double: (numpy.double, 2)
}


def TorchWindowCreate(tensor: torch.Tensor, name: str) -> bool:
    memory_shape = list(tensor.shape)
    dtype, disp_unit = TORCH_TYPE_MAP[tensor.dtype]
    window_table_entry = WindowTableEntry(
        tensor_type=dtype,
        memory_shape=memory_shape,
        memory=tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy(),
        window_name=name,
        disp_unit=disp_unit,
        device=DeviceType.GPU if tensor.is_cuda else DeviceType.CPU,
        device_id=int(tensor.get_device()) if tensor.is_cuda else -1
    )
    return WindowCreate(window_table_entry)


def TorchWindowFree(name: str) -> bool:
    return WindowFree(name)


def TorchWindowSync(name: str) -> torch.Tensor:
    neighbor_window_table_entries = WindowSync(name)
    neighbor_tensor_sum = None
    for neighbor_window_table_entry in neighbor_window_table_entries:
        neighbor_tensor = torch.from_numpy(neighbor_window_table_entry.memory)
        if neighbor_window_table_entry.device == DeviceType.GPU:
            neighbor_tensor = neighbor_tensor.cuda(device=neighbor_window_table_entry.device_id)
        if neighbor_tensor_sum is None:
            neighbor_tensor_sum = neighbor_tensor
        else:
            neighbor_tensor_sum.add_(neighbor_tensor)
    return neighbor_tensor_sum


def TorchWindowPut(tensor: torch.Tensor, name: str, dst_ranks: List[int]) -> int:
    memory_shape = list(tensor.shape)
    dtype, disp_unit = TORCH_TYPE_MAP[tensor.dtype]

    handle = HandleManager.AllocateHandle()

    def GenerateCallback(handle):
        def callback():
            HandleManager.MarkDone(handle)
        return callback

    window_table_entry = WindowTableEntry(
        tensor_type=dtype,
        memory_shape=memory_shape,
        window_name=name,
        dst_rank=dst_ranks,
        disp_unit=disp_unit,
        device=DeviceType.GPU if tensor.is_cuda else DeviceType.CPU,
        device_id=int(tensor.get_device()) if tensor.is_cuda else -1,
        memory=tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy(),
        callback=GenerateCallback(handle)
    )

    EnqueueTensorWindowPut(window_table_entry)
    return handle


def TorchWindowGet(tensor: torch.Tensor, name: str, src_ranks: List[int], average: bool) -> int:
    memory_shape = list(tensor.shape)
    dtype, disp_unit = TORCH_TYPE_MAP[tensor.dtype]

    handle = HandleManager.AllocateHandle()

    def GenerateCallback(handle, average, tensor):
        def callback(src_tensor_list):
            with torch.no_grad():
                tensor.zero_()
            for src_tensor in src_tensor_list:
                if tensor.is_cuda:
                    tensor.add_(torch.as_tensor(src_tensor, device=tensor.device))
                else:
                    tensor.add_(torch.from_numpy(src_tensor))
            if average:
                tensor.div_(len(src_tensor_list))
            HandleManager.MarkDone(handle)
        return callback
    window_table_entry = WindowTableEntry(
        tensor_type=dtype,
        memory_shape=memory_shape,
        window_name=name,
        src_rank=src_ranks,
        disp_unit=disp_unit,
        device=DeviceType.GPU if tensor.is_cuda else DeviceType.CPU,
        device_id=int(tensor.get_device()) if tensor.is_cuda else -1,
        memory=tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy(),
        callback=GenerateCallback(handle, average, tensor)
    )

    EnqueueTensorWindowGet(window_table_entry)
    return handle
