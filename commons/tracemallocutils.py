import tracemalloc
import linecache


from commons import get_runtime_logger

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))

    logger = get_runtime_logger()

    den = 1024**2
    top_stats = snapshot.statistics(key_type)

    msg = '\n'
    msg += "Top %s lines" % limit
    msg += '\n'
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        msg +=("#%s: %s:%s: %.2f MB"
              % (index, frame.filename, frame.lineno, stat.size / den))
        msg += '\n'
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            msg += ('    %s' % line)
            msg += '\n'

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        msg += ("%s other: %.2f MB" % (len(other), size / den))
        msg += '\n'
    total = sum(stat.size for stat in top_stats)
    msg += ("Total allocated size: %.2f MB" % (total / den))
    msg += '\n'

    logger.warning(msg)