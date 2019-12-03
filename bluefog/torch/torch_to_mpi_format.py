try:
    import cupy
    from torch.utils.dlpack import from_dlpack
    from torch.utils.dlpack import to_dlpack

    def to_mpi4py_aware_format(tensor):
        dx = to_dlpack(tensor)  # Convert it into a DLPack tensor.
        cx = cupy.fromDlpack(dx)  # Convert it into a CuPy array.
        return cx

    SUPPORT_CUPY = True


except ImportError:

    def to_mpi4py_aware_format(tensor):
        return tensor.cpu().numpy()

    SUPPORT_CUPY = False
