# -*- coding: utf-8 -*-
"""
tomita datasets

@author: Kaiwen Hou
kaiwen.hou@mila.quebec
"""


from toy_datasets import *

if __name__ == '__main__':
    max_len    = 15
    data_split = 0.99999999
    rng_key = jax.random.PRNGKey(0)

    for tomita_num in range(1,8):
        dataset = tomita_dataset(rng_key, data_split, max_len, 
                                 tomita_num=tomita_num)[0]
        # dataset = [f"^{s}$" for s in dataset]
        dset_len = len(dataset)
        assert score_tomita(dataset + ['01110001100'], tomita_num) < \
               score_tomita(dataset, tomita_num) == 1.
        assert tomita_size(data_split, 1, max_len, tomita_num) == dset_len
        print(f"Tomita {tomita_num} with max_len={max_len} has "
              f"{dset_len} strings")