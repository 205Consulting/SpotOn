import pandas as pd
import numpy as np
def shuffle(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
    	df.apply(np.random.shuffle, axis=axis)
    return df