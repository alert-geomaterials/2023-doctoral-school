from matplotlib import pyplot as plt, patches
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

# Get the directory where the current module is located
current_directory = os.path.dirname(__file__)

# Create a relative path to a file in the same directory
hole_path = os.path.join(current_directory, "img/hole.png")
gold_path = os.path.join(current_directory, "img/gold.png")

def plot_env(fig,ax,N,Lx=5, Ly=5):
    # circle1 = patches.Circle((0., 0.), radius=0.5, color='green')
    M=N+1
    cx=np.linspace(-Lx/2,Lx/2,M)
    cy=np.linspace(-Ly/2,Ly/2,M)
    for i in range(M):
        plt.plot([cx[i],cx[i]],[-Ly/2,Ly/2],"k")
        plt.plot([-Lx/2,Lx/2],[cy[i],cy[i]],"k")
    # ax.add_patch(circle1)
    ax.axis('equal')
#     print('hello','1/9')
    im_skull=plt.imread(hole_path)#('./img/hole.png')
    im_skull=OffsetImage(im_skull, zoom=.8*Lx/(M-1)/1.25)
    offy=.4*Lx/(M-1)/1.25
    ab = AnnotationBbox(im_skull, xy=(-Lx/2+Lx/(M-1)/2,-Ly/2+offy),frameon=False,pad=0)
    ax.add_artist(ab)
#     print('hello','1/9-1')
    im_gold=plt.imread(gold_path)#('./img/gold.png')
    im_gold=OffsetImage(im_gold, zoom=1.2*Ly/(M-1)/1.25)
    offy=.4*Lx/(M-1)/1.25
    ab = AnnotationBbox(im_gold, xy=(Lx/2-Lx/(M-1)/2,Ly/2-Ly/(M-1)+offy),frameon=False,pad=0)
    ax.add_artist(ab)
#     print('hello','1/9-2')
    for i in range(M-1):
        for j in range(M-1):
            ax.text(-Lx/2+i*Lx/(M-1)+.1,-Ly/2+j*Ly/(M-1)+Ly/(M-1)-.1-.2, "C"+str((j*(M-1) +i+1)), fontsize=12)
    ax.set_xlim(-Lx/2,Lx/2)
    ax.set_ylim(-Ly/2,Ly/2)
    fig.set_figwidth(Lx)
    fig.set_figheight(Ly)
    fig.patch.set_visible(False)
    ax.axis('off')
    # print('hello','1/9-3')
    return

def plot_arrows(fig,ax,policy,N, Lx=5, Ly=5, fc='k',ec='k'):
    M=N+1
    l=min(Lx,Ly)/(M-1)*.4
    R=np.array([[0,1],[-1,0]])
    for i in range(M-1):
        for j in range(M-1):
            if (i!=0 or j!=0) and ((i+1)*(j+1)!=(M-1)**2):
                idx=j*(M-1) +i+1
                # print(idx)
                # print(policy[idx-1])
                if policy[idx-1]!=None: 
                    Ra=np.linalg.matrix_power(R,(policy[idx-1]))
                    av=Ra@np.array([l,0])
                    pxA=(i*Lx/(M-1)+(i+1)*Lx/(M-1))/2-av[0]/2-Lx/2
                    pyA=(j*Ly/(M-1)+(j+1)*Ly/(M-1))/2-av[1]/2-Ly/2
                    ax.arrow(pxA, pyA, av[0], av[1], fc=fc, ec=ec, head_width=0.05, head_length=0.1)
                else:
                    continue
