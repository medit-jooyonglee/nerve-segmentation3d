import os
import subprocess
import re
# // http://ramu492.blogspot.com/2014/03/auto-increment-build-numbers-in-visual.html
# https://gist.github.com/JarekParal/24e15de33971674a8c0f9955f6ad15e0

def _maturity(path):
    """
    Return maturity based on svn root URL of path.
    """
    # URL string to return value map
    root = {
        'trunk':    '',     # normal
        'branches': 'a',    # alpha
        'tags':     'post'  # post
    }
    args = ("svn", "info", path)
    # out = subprocess.check_output(args).decode('utf-8')
    out = subprocess.check_output(args).decode('euc-kr')
    match = re.search(r'Relative URL: (.*)', out)
    if not match:
        return None
    match = match.group(1)
    return next(( v for k, v in root.items() if match.find(k) > 0 ))

def get_version(path, major=None, maturity=None):
    """
    Get the svn version and convert to PEP 440 compliant format.
    """
    version = []
    if major:
        version.append(str(major))

    def parse(s):
        """ parse a revision from ``svnversion`` """
        # replace modified(M), switched(S), and partial(P)
        rev, _, dev = re.sub('[MSP]', '.dev', s).partition('.')
        return (rev, dev) # rev is number, dev is '' or '.dev'

    args = ("svnversion", path)
    out = subprocess.check_output(args).decode('utf-8').strip()
    # x:y -- mixed revision working copy
    rev, _, wc = out.partition(':')
    print('-->', rev)
    rev, dev = parse(rev)
    version.append(rev + (maturity or ''))
    if wc:
        wc, dev = parse(wc)
        version.append(wc)
    if dev:
        version.append(dev)
    return '.'.join(version)

__version__ = get_version(".", 1, _maturity("."))
print(__version__)