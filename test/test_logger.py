import pytest
from commons import get_runtime_logger
from trainer import utils

def test_logger():

    temp_log = utils.get_logger('temp')
    bac_log = utils.get_logger('bac')
    temp_log.info('temp_log')
    bac_log.info('bac_log')

    assert True

def test_logger():

    temp_log = get_runtime_logger()
    # bac_log = utils.get_logger('bac
    temp_log.info('this is runtime logger')
    assert True

if __name__ == '__main__':
    pytest.main([
        '-s',
        '--color=yes',
        '-rGA',
        __file__
    ])