def process(values):
    try:
        total = 0
        for v in values:
            total += v
        return total
    except Exception:
        return None

def list_max(values):
    if not values:
        return 0
    m = values[0]
    for v in values[1:]:
        if v > m:
            m = v
    return m

def list_min(values):
    if not values:
        return 0
    m = values[0]
    for v in values[1:]:
        if v < m:
            m = v
    return m

def filter_even(values):
    return [v for v in values if v % 2 == 0]

def dict_merge(a, b):
    out = {}
    out.update(a or {})
    out.update(b or {})
    return out

def map_square(values):
    return [v * v for v in values]

def string_reverse(s):
    return s[::-1]

def set_union(a, b):
    res = set()
    res.update(a or set())
    res.update(b or set())
    return sorted(res)

def abs_value(x):
    return x if x >= 0 else -x

def list_average(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def easy_pass():
    return 1
