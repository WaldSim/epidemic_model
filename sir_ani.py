import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # do this before importing pylab

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

###############

def iterate_grid(grid, beta=0.5, gamma=0.5, neighborhood='vonNeumann'):
    if neighborhood=='vonNeumann':
        N = 5
        X = np.zeros((N, grid.shape[0], grid.shape[1]))
        X[0] = grid
        S, I, R = np.zeros_like(grid), np.zeros_like(grid), np.zeros_like(grid)
        for i, shift, axis in zip(range(1, 5), [1,-1,1,-1], [0,0,1,1]):
            X[i] = np.roll(grid, shift, axis=axis)
            S += X[i]==0
            I += X[i]==1
            R += X[i]==2
            
        rand1 = np.random.random(grid.shape)
        cond1 = (rand1 <= beta*S*I/N).astype(int)
        
        grid[grid==0] += cond1[grid==0]
        grid[grid==1] += (np.random.random(grid.shape) <= gamma*I/N)[grid==1]
            
    elif neighborhood=='moore':
        raise NotImplementedError('moore neighborhood not yet supported')
    else:
        raise ValueError('Non-valid neighborhood. Please chose one of [`moore`,`vonNeumann`]')
    
    return grid

def calc_stats(grid):
    # S = 0, I = 1, R = 2
    num_s = len(grid[grid==0])/np.prod(grid.shape)
    num_i = len(grid[grid==1])/np.prod(grid.shape)
    num_r = len(grid[grid==2])/np.prod(grid.shape)
    return {'S':num_s, 'I':num_i, 'R':num_r}
    

##################

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--s0', type=float, default=0.99)
    parser.add_argument('--i0', type=float, default=0.01)
    parser.add_argument('--r0', type=float, default=0.0)
    parser.add_argument('--grid_size', type=int, default=256)
    parser.add_argument('--n_steps', type=int, default=100)
    
    return parser


def track_stats(stats, grid):
    stat = calc_stats(grid)
    for key in stats.keys():
        stats[key].append(stat[key])
    
    
def plot_stats(stats):
    for key in stats.keys():
        plt.plot(stats[key], label=key, color=colors[key])
    plt.legend()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Affected respective population')
    
    
if __name__=='__main__':
    
    parser = create_parser()
    opts = parser.parse_args()
    
    r0 = opts.r0/(opts.r0+opts.s0+opts.i0)
    s0 = opts.s0/(opts.r0+opts.s0+opts.i0)
    i0 = opts.i0/(opts.r0+opts.s0+opts.i0)
    
    grid = np.random.choice(np.array([0,1,2]), size=(opts.grid_size, opts.grid_size), p=np.array([s0,i0,r0]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 3]}, figsize=(13,7))

    stats = {'S':[], 'R':[], 'I':[]}
    colors = {'S':'darkblue', 'I':'darkred', 'R':'darkgreen'}
    
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap([c('blue'), c('blue'), 0.33, c('red'), c('red'), 0.66, c('green')])
    
    def animate():
        tstart = time.time()                   # for profiling
        data=grid
        im=ax1.imshow(data, vmin=0, vmax=2, cmap=rvb)
        
        track_stats(stats, grid)
        for key in stats.keys():
            ax2.plot(np.arange(len(stats[key])), stats[key], label=key, color=colors[key])
        ax2.legend()
        ax2.grid()
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Affected respective population')
        
        for i in np.arange(1,opts.n_steps):
            data=iterate_grid(grid, beta=opts.beta, gamma=opts.gamma)
            track_stats(stats, data)
            im.set_data(data)
            ax1.set_title('Iteration {}'.format(i))
            for key in stats.keys():
                ax2.plot(np.arange(len(stats[key])), stats[key], label=key, color=colors[key])            
            fig.canvas.draw()                         # redraw the canvas
        print('FPS:' , opts.n_steps/(time.time()-tstart))
        return

    win = fig.canvas.manager.window
    fig.canvas.manager.window.after(100, animate)
    plt.show()
    plt.close()

