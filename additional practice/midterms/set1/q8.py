def count_instances(num, seq):
    seq = list(seq)
    count = 0
    for i in range(len(seq)):
        if seq[i] == num:
            count += 1
    return count
