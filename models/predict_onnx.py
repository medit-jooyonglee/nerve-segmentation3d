import os

import numpy as np
import onnxruntime

from commons.common_utils import get_runtime_logger, timefn2


class NerveOnnx(object):
    input_names = ['input']
    output_names = ['output']

    # @timefn2
    def __init__(self, onnx_filename, device='cuda'):
        assert os.path.exists(onnx_filename), f'emtpy onnx file:{onnx_filename}'

        logger = get_runtime_logger()
        sess_options = onnxruntime.SessionOptions()
        # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = onnxruntime.get_available_providers()
        name = device.upper()
        cuda_providers = [prov for prov in providers if prov.find(name) >= 0]
        if len(cuda_providers) == 0:
            logger.warning('cannot use cuda')
        else:
            providers = cuda_providers

        self.ort_session = onnxruntime.InferenceSession(onnx_filename, sess_options, providers=providers)

    # @timefn2
    def predict(self, input):
        input_dict = {self.input_names[0]: input}
        res = self.ort_session.run(self.output_names, input_dict)

        return res[0]

    # FIXME
    # 현재 prediction 과정마다 binding
    def predict_binding(self, x, y):
        x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
        y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type(y, np.float32, 'cuda', 0)

        io_binding = self.ort_session.io_binding()
        io_binding.bind_input(name='input',
                              device_type=x_ortvalue.device_name(),
                              device_id=0,
                              element_type=np.float32,
                              shape=x_ortvalue.shape(),
                              buffer_ptr=x_ortvalue.data_ptr())
        io_binding.bind_output(name='output',
                               device_type=y_ortvalue.device_name(),
                               device_id=0,
                               element_type=np.float32,
                               shape=y_ortvalue.shape(),
                               buffer_ptr=y_ortvalue.data_ptr())
        self.ort_session.run_with_iobinding(io_binding)
        ort_output = io_binding.copy_outputs_to_cpu()[0]
        # ort_output = io_binding.get_outputs()[0]

        return ort_output


def to_type(x, dtype):
    if issubclass(type(x), (tuple, list)):
        return [to_type(v, dtype) for v in x]
    elif type(x) == dict:
        return {k: to_type(v) for k, v in x.items()}
    elif type(x) == np.ndarray:
        # 정수형일고 일치할때만
        if np.issubdtype(x.dtype, np.integer) and np.issubdtype(dtype, np.integer):
            if x.dtype == dtype:
                return x
            else:
                return x.astype(dtype)
        # 실수형일때
        elif np.issubdtype(x.dtype, np.floating) and np.issubdtype(dtype, np.floating):
            if x.dtype == dtype:
                return x
            else:
                return x.astype(dtype)
        else:
            return x
    else:
        raise NotImplementedError('not implemented for conversion as tensor:{}'.format(type(x)))


def to_batch(x):
    if issubclass(type(x), (tuple, list)):
        return [to_batch(v) for v in x]
    elif type(x) == dict:
        return {k: to_batch(v) for k, v in x.items()}
    elif type(x) == np.ndarray:
        return x[np.newaxis]
    else:
        raise NotImplementedError('not implemented for conversion as tensor:{}'.format(type(x)))

def data_convert_onnx(datas, **kwargs):
    dtype = kwargs.get('dtype', 'float32')
    res = datas
    res = to_type(res, dtype)
    res = to_batch(res)

    return res
