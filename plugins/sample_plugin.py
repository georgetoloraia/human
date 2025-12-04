def process(values):
    if values is None:
        return None
    total = 0
    for v in values:
        total += v
    return total
