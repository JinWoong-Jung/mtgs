import itertools


def pair_vector_to_matrix(pair_values, num_people, reverse=False):
    n = round(0.5 + 0.5 * (1 + 4 * pair_values.shape[-1]) ** 0.5)
    offset = num_people - n
    mat = pair_values.new_full((*pair_values.shape[:-1], num_people, num_people), -1)
    for idx, (i, j) in enumerate(itertools.permutations(range(offset, offset + n), 2)):
        if reverse:
            mat[..., j, i] = pair_values[..., idx]
        else:
            mat[..., i, j] = pair_values[..., idx]
    return mat
