# https://stackoverflow.com/questions/1093322/how-do-i-check-what-version-of-python-is-running-my-script
import sys

print(sys.version_info)

assert sys.version_info.minor == 6
assert sys.version_info >= (4, 5), "not supported python version:{}".format(sys.version_info)
