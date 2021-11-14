# functions for generating point clouds to run MLP on
# MLP Project - CSE489
# Joseph Coston & Douglas Newquist
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class pointCloud2D():
     
    def generate(self, f, n) -> pd.DataFrame:
        ''' Generate a separable point cloud of n coordiante pairs. 
            n:      the number of points to generate
            f:      lambda function to be used to artificially separate point cloud
            ret:    2 tuples of 2 arrays'''
        # generate random point cloud
        x = np.random.uniform(size=n)
        y = np.random.uniform(size=n)
        df = pd.DataFrame(columns=['x', 'y', 'op'])

        # separate point cloud according to passed function
        for i in range(n):
            df = df.append({'x': x[i], 'y': y[i], 'op': f(x[i],y[i])}, ignore_index=True)
        return df

    def to_fig(self):
        fig = plt.figure()
        ax = fig.add_subplot()

        groups = self.data.groupby('op')
        for name, group in groups:
            plt.plot(group['x'], group['y'], marker='o', linestyle='', label=name)
        plt.legend()

        ax.set_aspect('equal', adjustable='box')

    def __init__(self, f, n=1000) -> None:
        self.data = self.generate(f,n)