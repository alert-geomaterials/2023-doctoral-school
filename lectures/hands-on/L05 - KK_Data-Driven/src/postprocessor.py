# -*- coding: utf-8 -*-
"""
Postprocessing routines

@author: Konstantinos Karapiperis
"""
import numpy as np
import matplotlib as mpl
from pylab import Line2D
from matplotlib import cm
import matplotlib.pyplot as plt

class PostProcessor():

    def __init__(self, coord, conn, mag_factor):
        
        self.coord = coord
        self.conn = conn
        self.mag_factor = mag_factor
        
#==============================================================================

    def plot_deformed_field(self, disp, qoi, qoi_name):
        '''
        qoi: quantity of interest (e.g stress component) given in a np array
        '''        
        # Setup plot
        fig, ax = plt.subplots(figsize=(10,7.5))

        # Compute deformed coordinates
        disp = disp.reshape((-1,2))
        x = self.coord + self.mag_factor * disp

        # Normalize
        qoi_norm = (qoi - qoi.min()) / qoi.ptp()
                           
        for i in range(len(x)):
                            
            # Find neighbouring nodes 
            neighb = set()
                         
            # Scan connectivity matrix rows
            for j in range(len(self.conn)):

                # If we find a row with this node
                if i in self.conn[j]:
                     
                    # Neighbour in an element
                    neighb_el_ind = np.where(self.conn[j] == i)[0]
                     
                    if neighb_el_ind in [0,2]:
                        neighb.add(int(self.conn[j][-1]))
                        neighb.add(int(self.conn[j][1]))
                    elif neighb_el_ind in [1,3]:
                        neighb.add(int(self.conn[j][0]))
                        neighb.add(int(self.conn[j][2]))
            
            # Linear grid between each neighboring nodes
            n_grid = 5
            
            # For every neighbor
            for j in neighb:
                x_master = np.linspace(x[i,0], x[j,0] ,n_grid)
                y_master = np.linspace(x[i,1], x[j,1], n_grid)
                
                for k in range(n_grid-1):
                    ax.add_line(Line2D([x_master[k], x_master[k+1]], 
                                       [y_master[k], y_master[k+1]],
                                       color=cm.jet(qoi_norm[i]),
                                       alpha = 1-float(k)/(n_grid-1)))

        ax.scatter(x[:,0], x[:,1], c=cm.jet(qoi_norm), s=20)    
        ax.set_xlim([np.min(x[:,0])-1, np.max(x[:,0])+1])
        ax.set_ylim([np.min(x[:,1])-1, np.max(x[:,1])+1])
        ax.axis('off')
        
        # Create a custom colorbar for the QoI values
        cmap = mpl.cm.jet
        norm = mpl.colors.Normalize(vmin=np.min(qoi),vmax=np.max(qoi))
        cax = fig.add_axes([0.75, 0.35, 0.02, 0.3]) 
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, 
                                       orientation='vertical', format='%.2f')
        cb.set_label(qoi_name, fontsize=12)
        plt.show()
