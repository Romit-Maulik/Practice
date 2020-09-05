from urlparse import urlparse
 
import pygtk
import gtk
import tkSimpleDialog
 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
 
import numpy as np
 
def tellme(s):
    print s
    plt.title(s,fontsize=16)
    plt.draw()
 
def pic2data(source='clipboard',straight=True):
    """ GUI to get data from a XY graph image. Either provide the graph
        as a path to an image in 'source' or copy it to the clipboard.
    """
     
    ##### GET THE IMAGE
     
    clipboard = gtk.clipboard_get()
     
    if source=='clipboard':
         
        # This chunk tries the text content of the clipboard
        # and empties it if it is not a file path
         
        print "Waiting for an image in the clipboard..."
        while not ( clipboard.wait_is_uris_available()
                    or clipboard.wait_is_image_available()):
            pass
             
        if clipboard.wait_is_uris_available(): # it's a path to a file !
             
             source = clipboard.wait_for_uris()[0]
             source = urlparse(source).path
             return pic2data(source)
         
        image = clipboard.wait_for_image().get_pixels_array()
        origin = 'upper'
     
    else: # source is a path to a file !
         
        image = mpimg.imread(source)
        origin = 'lower'
 
    ###### DISPLAY THE IMAGE
     
    plt.ion() # interactive mode !
    fig, ax = plt.subplots(1)
    imgplot = ax.imshow(image, origin=origin)
    fig.canvas.draw()
    plt.draw()
     
    ##### PROMPT THE AXES
     
    def promptPoint(text=None):
         
        if text is not None: tellme(text)
        return  np.array(plt.ginput(1,timeout=-1)[0])
     
    def askValue(text='',initialvalue=0.0):
        return tkSimpleDialog.askfloat(text, 'Value:',
                                         initialvalue=initialvalue)
     
    origin = promptPoint('Place the origin')
    origin_value = askValue('X origin',0),askValue('Y origin',0)
                                          
    Xref =  promptPoint('Place the X reference')
    Xref_value = askValue('X reference',1.0)
     
    Yref =  promptPoint('Place the Y reference')
    Yref_value = askValue('Y reference',1.0)
     
    if straight :
         
        Xref[1] = origin[1]
        Yref[0] = origin[0]
     
    ##### PROMPT THE POINTS
     
    selected_points = []
     
    tellme("Select your points !")
    print "Right-click or press 's' to select"
    print "Left-click or press 'del' to deselect"
    print "Middle-click or press 'Enter' to confirm"
    print "Note that the keyboard may not work."
     
    selected_points = plt.ginput(-1,timeout=-1)
     
    ##### RETURN THE POINTS COORDINATES
     
    #~ selected_points.sort() # sorts the points in increasing x order
     
    # compute the coordinates of the points in the user-defined system
     
    OXref = Xref - origin
    OYref = Yref - origin
    xScale =  (Xref_value - origin_value[0]) / np.linalg.norm(OXref)
    yScale =  (Yref_value - origin_value[1]) / np.linalg.norm(OYref)
     
    ux = OXref / np.linalg.norm(OXref)
    uy = OYref / np.linalg.norm(OYref)
     
    result = [(ux.dot(pt - origin) * xScale + origin_value[0],
               uy.dot(pt - origin) * yScale + origin_value[1])
               for pt in selected_points ]
     
    # copy the result to the clipboard
     
     
    clipboard.set_text('[' + '\n'.join([str(p) for p in result]) + ']')
     
    clipboard.store() # makes the data available to other applications
     
    plt.ioff()
     
    return result

if __name__ == '__main__':
    data = pic2data()
    print(data)