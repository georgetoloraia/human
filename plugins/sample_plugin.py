def process(values):
    # Intentional placeholder: should return sum of numbers.
    total = 0
    for v in values:
        total += v
    return total


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
