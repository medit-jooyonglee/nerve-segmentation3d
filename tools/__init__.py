try:
    from pyinterpolate.InterpolateWrapper import InterpolateWrapper
except Exception as e:
    print(e.args)
try:
    from . import dicom_read_wrapper
except Exception as e:
    print(e.args)