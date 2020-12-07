from tqdm import tqdm, trange
from time import sleep
from random import random, randint

with trange(10, desc="Epoch {}: ".format(1)) as t:
    for i in t:
        # Description will be displayed on the left
        t.set_description('GEN %i' % i)
        # Postfix will be displayed on the right,
        # formatted automatically based on argument's datatype
        t.set_postfix(loss=random(), gen=randint(1, 999), str='h',
                      lst=[1, 2])
        sleep(0.1)
