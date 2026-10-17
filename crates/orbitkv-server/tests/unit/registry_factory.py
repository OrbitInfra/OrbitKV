"""CPU contract fixture for the registry's actual Python call arguments."""

import types


class Storage:
    def nbytes(self):
        return 64

    def data_ptr(self):
        return 4096


class Tensor:
    dtype = "torch.float16"
    device = types.SimpleNamespace(index=0)

    def set_(self, storage, offset, shape, strides):
        if not isinstance(storage, Storage) or offset != 2:
            raise ValueError("wrong storage or view offset")
        if shape != [8] or strides != [1]:
            raise ValueError("wrong shape or strides")
        self.shape = shape
        self.strides = strides
        return self

    def data_ptr(self):
        return 4100

    def storage_offset(self):
        return 2

    def stride(self):
        return self.strides

    def element_size(self):
        return 2


def empty(shape, *, device, dtype):
    if shape != (0,) or device != "cuda:0" or dtype != "torch.float16":
        raise ValueError("wrong empty factory arguments")
    return Tensor()


def tensor(data, **kwargs):
    if isinstance(data, bytes):
        raise TypeError("new(): invalid data type 'bytes'")
    raise AssertionError("registration must create an empty view")


def shared(*args):
    if len(args) != 8 or args[0] != 0 or args[2:4] != (64, 0):
        raise ValueError("wrong IPC arguments")
    return Storage()


torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(set_device=lambda device: None)
torch.UntypedStorage = types.SimpleNamespace(_new_shared_cuda=shared)
torch.float16 = "torch.float16"
torch.empty = empty
torch.tensor = tensor
