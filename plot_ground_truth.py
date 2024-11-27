#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot(args):
    frame_count = 900
    load_data = np.loadtxt(f'data/data-{args.index}.log')
    load_data = load_data.reshape((frame_count, 3,4))
    fig = plt.figure()
    traj = fig.add_subplot(111, projection='3d')
    traj.plot(load_data[:,:,3][:,0], # Get x position from each transform
              load_data[:,:,3][:,1], # Get y position from each transform
              load_data[:,:,3][:,2]) # Get z position from each transform
    traj.set_xlabel('x')
    traj.set_ylabel('y')
    traj.set_zlabel('z')
    # Invert axis so y-axis points "right" when facing in direction of positive x, and above xy plane
    plt.gca().invert_xaxis()
    plt.title('Ground Truth Visualization')
    plt.show()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Plots numpy saved data-*.log files.')
    argparser.add_argument(
        '--index',
        default=0,
        type=int,
        help='Index number (default: 0)')
    args = argparser.parse_args()
    plot(args)