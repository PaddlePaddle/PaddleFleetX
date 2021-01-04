
def fake_ctr_reader():
    def reader():
        for _ in range(1000):
            deep = np.random.random_integers(0, 1e10, size=16).tolist()
            wide = np.random.random_integers(0, 1e10, size=8).tolist()
            label = np.random.random_integers(0, 1, size=1).tolist()
            yield [deep, wide, label]

    return reader

