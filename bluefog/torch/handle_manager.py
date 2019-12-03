import  threading
from typing import Dict

from bluefog.common.common import Status


class HandleManager:
    _last_handle: int = -1
    _results: Dict[int, Status] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def AllocateHandle(cls) -> int:
        with cls._lock:
            cls._last_handle += 1
            cls._results[cls._last_handle] = Status.INPROGRESS
        return cls._last_handle

    @classmethod
    def MarkDone(cls, handle) -> None:
        with cls._lock:
            cls._results[handle] = Status.OK

    @classmethod
    def PollHandle(cls, handle) -> bool:
        with cls._lock:
            if handle not in cls._results:
                raise ValueError(f"Handle {handle} was not created or has been cleared.")
            res = cls._results[handle] == Status.OK
        return res

    @classmethod
    def ReleaseHandle(cls, handle) -> Status:
        with cls._lock:
            if handle not in cls._results:
                raise ValueError(f"Handle {handle} was not created or has been cleared.")
            status = cls._results.pop(handle)
        return status
