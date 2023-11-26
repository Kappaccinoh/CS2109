def count_instances(num, seq):
    seq = list(seq)
    if len(seq) == 0:
        return 0
    else:
        count = 0
        if seq[0] == num:
            count = 1
        seq.pop(0)
        seq = tuple(i for i in seq)
        return count_instances(num, seq) + count

    