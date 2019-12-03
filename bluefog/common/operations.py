import collections
import queue
import threading
import traceback

import networkx as nx
import numpy as np
from bluefog.common.common import BlueFogError, TensorTableEntry, MPIOpsType, WindowTableEntry
from bluefog.common.global_state import BlueFogGlobalState
from bluefog.common.mpi_context import MPIContext
from bluefog.common.mpi_controller import MPIController


def BackgroundThreadLoop(bluefog_global: BlueFogGlobalState):
    # bluefog_global.controller.start_receiver_for_neighbors()
    while not bluefog_global.shut_down:
        bluefog_global.controller.process_queue()


def BackgroundWindowThreadLoop(bluefog_global: BlueFogGlobalState):
    while not bluefog_global.shut_down:
        bluefog_global.controller.process_window_queue()


def InitializeBlueFogOnce(bluefog_global: BlueFogGlobalState, topology: nx.DiGraph):
    if (bluefog_global is None) or (not bluefog_global.initialized):
        # A global state and context
        bluefog_global = BlueFogGlobalState(
            initialized=True,
            shut_down=False,
            ops_cnt=collections.defaultdict(int),
            tensor_queue=queue.Queue(),
            window_queue=queue.Queue(),
            queue_cv=threading.Condition(),
            context=MPIContext.getInstance(topology),
        )

        bluefog_global.topology = bluefog_global.context.topology
        bluefog_global.controller = MPIController(
            bluefog_global.tensor_queue,
            bluefog_global.window_queue,
            bluefog_global.context,
            bluefog_global.ops_cnt
        )
        background_thread = threading.Thread(
            target=BackgroundThreadLoop, args=(bluefog_global,), daemon=True
        )
        bluefog_global.background_thread = background_thread
        bluefog_global.background_thread.start()

        background_window_thread = threading.Thread(
            target=BackgroundWindowThreadLoop, args=(bluefog_global,), daemon=True
        )
        bluefog_global.background_window_thread = background_window_thread
        bluefog_global.background_window_thread.start()

    return bluefog_global


def EnqueueTensorAllReduce(tensor_table_entry: TensorTableEntry):
    bluefog_global = BlueFogGlobalState()
    if not bluefog_global.initialized or bluefog_global.shut_down:
        raise BlueFogError(
            "MPI does not initialized correctly or ready shutdown.")
    bluefog_global.tensor_queue.put((tensor_table_entry, MPIOpsType.ALLREDUCE))


def EnqueueTensorAllGather(tensor_table_entry: TensorTableEntry):
    bluefog_global = BlueFogGlobalState()
    if not bluefog_global.initialized or bluefog_global.shut_down:
        raise BlueFogError(
            "MPI does not initialized correctly or ready shutdown.")
    bluefog_global.tensor_queue.put((tensor_table_entry, MPIOpsType.ALLGATHER))


def EnqueueTensorBroadcast(tensor_table_entry: TensorTableEntry):
    bluefog_global = BlueFogGlobalState()
    if not bluefog_global.initialized or bluefog_global.shut_down:
        raise BlueFogError(
            "MPI does not initialized correctly or ready shutdown.")
    bluefog_global.tensor_queue.put((tensor_table_entry, MPIOpsType.BROADCASE))


def EnqueueTensorNeighborAllGather(tensor_table_entry: TensorTableEntry):
    bluefog_global = BlueFogGlobalState()
    if not bluefog_global.initialized or bluefog_global.shut_down:
        raise BlueFogError(
            "MPI does not initialized correctly or ready shutdown.")
    bluefog_global.tensor_queue.put(
        (tensor_table_entry, MPIOpsType.NEIGHBOR_ALLGATHER))


def EnqueueTensorNeighborAllReduce(tensor_table_entry: TensorTableEntry):
    bluefog_global = BlueFogGlobalState()
    if not bluefog_global.initialized or bluefog_global.shut_down:
        raise BlueFogError(
            "MPI does not initialized correctly or ready shutdown.")
    bluefog_global.tensor_queue.put((tensor_table_entry, MPIOpsType.NEIGHBOR_ALLREDUCE))


def EnqueueTensorWindowPut(window_table_entry: WindowTableEntry):
    bluefog_global = BlueFogGlobalState()
    if not bluefog_global.initialized or bluefog_global.shut_down:
        raise BlueFogError(
            "MPI does not initialized correctly or ready shutdown.")
    bluefog_global.window_queue.put((window_table_entry, MPIOpsType.WIN_PUT))

def EnqueueTensorWindowGet(window_table_entry: WindowTableEntry):
    bluefog_global = BlueFogGlobalState()
    if not bluefog_global.initialized or bluefog_global.shut_down:
        raise BlueFogError(
            "MPI does not initialized correctly or ready shutdown.")
    bluefog_global.window_queue.put((window_table_entry, MPIOpsType.WIN_GET))

def WindowCreate(window_table_entry: WindowTableEntry) -> bool:
    bluefog_global = BlueFogGlobalState()
    if not bluefog_global.initialized or bluefog_global.shut_down:
        raise BlueFogError(
            "MPI does not initialized correctly or ready shutdown.")
    try:
        bluefog_global.context.WindowCreate(window_table_entry)
        return True
    except KeyError:
        traceback.print_exc()
        return False
    else:
        traceback.print_exc()
        return False


def WindowFree(name: str) -> bool:
    bluefog_global = BlueFogGlobalState()
    if not bluefog_global.initialized or bluefog_global.shut_down:
        raise BlueFogError(
            "MPI does not initialized correctly or ready shutdown.")
    # TODO(ybc) What if there still are some win ops waiting in the queue
    # related with this window?
    try:
        bluefog_global.context.WindowFree(name)
        return True
    except KeyError:
        traceback.print_exc()
        return False
    else:
        traceback.print_exc()
        return False


def WindowSync(name: str):
    bluefog_global = BlueFogGlobalState()
    if not bluefog_global.initialized or bluefog_global.shut_down:
        raise BlueFogError(
            "MPI does not initialized correctly or ready shutdown.")
    neighbor_window_table_entries = bluefog_global.context.WindowSync(name)
    return neighbor_window_table_entries
