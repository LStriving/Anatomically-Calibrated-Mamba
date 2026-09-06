from time import perf_counter_ns

def elapsed_ms(start_ns):
    return (perf_counter_ns() - start_ns) / 1_000_000
