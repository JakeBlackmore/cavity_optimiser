
from matplotlib import pyplot as plt
from matplotlib import ticker,gridspec
from matplotlib.colors import ListedColormap
import numpy as np
from jqc import Ox_plot,jqc_plot

'''
#############################################################################
This is a module to normalise some plotting aspects for our cavity optimising
paper.

draw_brace() draws the curly bracket to indicate shared axis labels.

there are five colour maps defined in here:

jqc_red_blue - newcastle blue to durham red

jqc_sand_red - jqc sand to durham red

Ox_blue - oxford blue to red

Ox_yellow_orange - oxford beige/yellow to orange

Ox_green_blue - green to oxford blue

############################################################################
Author: J Blackmore 2021
############################################################################


'''

def draw_brace(ax, xspan, yy, text,axis = 'x',Radius=300.):
    """Draws an annotated brace outside the axes.

    #########################################################################
    args:
    ax (plt.axis) : axis to draw the brace on. To go over multiple axes should
                    be the last axis drawn!
    xspan ((float(xmin),float(xmax))): minimum and maximum to draw brace on the
                                        "x" axis. In this case the x axis is the
                                        long axis of the brace.
    yy (float): offset in the y axis. This is the "short" of the brace
    text (str): text to label the brace with.
    axis (str): either "x" or "y". Figure axis to draw brace along.
    Radius (float): Radius of curvature for the curly brace. This will be
                    fine-tuned for different plots.
    #########################################################################

    code adapted from : https://stackoverflow.com/a/68180887

    modifications: allows for x or y axis to be annotated.



    """
    #set up limits for the plotting
    xmin, xmax = xspan
    xspan = xmax - xmin

    #calculate path for the brace to follow
    resolution = int(10000)*2+1 # guaranteed uneven
    beta = Radius/xspan

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = abs(y)
    y = y-np.amin(y)
    y = y/np.amax(y)
    #this shifts the y position of the brace about
    y = yy-0.1*y# adjust vertical position

    if axis =='y':
        #if user has selected y axis then swap x and y
        x,y = y,x
    ax.autoscale(False)
    #draw the brace using plt.plot
    ax.plot(x,y , color='black', lw=2, clip_on=False,zorder=25,
            transform=ax.transAxes)
    #if x axis draw the text along x, and have it horizontal
    if axis =='x':
        ax.text((xmax+xmin)/2., np.amin(y), text, ha='center',
                va='top',transform=ax.transAxes)
    #else if y draw along y and be vertical
    elif axis =='y':
        ax.text(np.amin(x)-0.025,(xmax+xmin)/2., text, ha='right',
        va='center',transform=ax.transAxes,rotation=90)

'''
##############################################################################
From here we have the definitions of the colour maps

all colours are defined in Ox_plot.colours or jqc_plot.colours which are
python dicts.
##############################################################################

'''
N = 1024 #number of points to use for linear gradient. More than 1000 is not needed

one = 'blue' #reference to colour one
two = 'red' #reference to colour two

vals = np.ones((N,4)) #Nx4 array for colour map. 4th index is alpha/transparency
for i in range(3):
    vals[:N,i]=np.linspace(jqc_plot.colours[one][i],jqc_plot.colours[two][i],N)

jqc_red_blue = ListedColormap(vals)
jqc_red_blue_r = ListedColormap(vals[::-1]) #register colour map

#repeat:

##############################################################################

one = 'ox blue'
two = 'red'

vals = np.ones((N,4))
for i in range(3):
    vals[:N,i]=np.linspace(Ox_plot.colours[one][i],Ox_plot.colours[two][i],N)

Ox_red_blue = ListedColormap(vals)
Ox_red_blue_r = ListedColormap(vals[::-1])


##############################################################################
one = 'palegreen'
two = 'ox blue'


for i in range(3):
    vals[:N,i]=np.linspace(Ox_plot.colours[one][i],Ox_plot.colours[two][i],N)

Ox_green_blue= ListedColormap(vals)
Ox_green_blue_r = ListedColormap(vals[::-1])
################################################################################
three = "yellow"
four = "orange"

for i in range(3):
    vals[:N,i]=np.linspace(Ox_plot.colours[one][i],Ox_plot.colours[two][i],N)

Ox_yellow_orange = ListedColormap(vals)
Ox_yellow_r = ListedColormap(vals[::-1])

##############################################################################
one = 'red'
two = 'sand'

vals = np.ones((N,4))
for i in range(3):
    vals[:N,i]=np.linspace(jqc_plot.colours[one][i],jqc_plot.colours[two][i],N)

jqc_sand_red = ListedColormap(vals)
jqc_sand_red_r = ListedColormap(vals[::-1])
