import tensorflow as tf
import tensorflow.keras.backend as K
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from modules.generator import DataGenerator
import os
from scipy import ndimage
from skimage.transform import rescale, resize
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import io

def figure(dsVO_src, dsVO,
           generator, # actual Tensorflow generator
           train_names, # names for printout
           this_valid_mask, # mask for patient extraction (first dimension of data arrays)
           epoch=0, # epoch num for printout
           n_patients=5, #number of patients, if 1 will show multiple slices
           save=False, 
           multiquality=False, #show 1 and 3 qualities
           show=False,
           shuffle=False,
           buffer=False,
           force_fs=False,
           output="figures", # output folder
           savestr="validation_at_epoch_%%%", # image output name, %%% will be replaced by epoch
           show_outline=False, # show stroke outline extracted from mask
           show_pred=True,
           predroi_overlay=False,
           raw=None,
           only=False,
           as_numpy_result=None,
           maxquality=True
          ):
    
    # update file name
    savestr = savestr.replace("%%%",str(epoch))
    batch_size = 1 
    if raw is not None:
        batch_size = 1
    elif n_patients == 1:
        batch_size = np.sum(this_valid_mask)
        # create generator
        dsVO_src.reset_indices(this_valid_mask)
    else:
        dsVO_src.reset_indices(this_valid_mask)
        dsVO_src.shuffle = shuffle
        dsVO_src.on_epoch_end()

    # number of columns computation (b0, b1000, T2eg, GenT2EG)        
    if only == "real" or only == "synth":
        n_cols = 3
        st2egpos = 2
    else:
        n_cols = 4
        st2egpos = 3
    if show_pred :
        n_cols += 2
    
    # deactivation of inline matplotlib if needed
    if not show:
        plt.ioff()
    # preloading patient data
    if raw is not None:
        patients = [raw]
    else:
        patients = []
        for i in dsVO.take(n_patients):
            patients.append(as_numpy_result(i)[0])
    
    # computation of slices number
    if n_patients > 1:
        n_slices = n_patients
    elif raw is not None:
        print_slices = list(np.argwhere(np.max(patients[0][0]["mask"][...,1], axis=(1,2)) == 1))
        n_slices = len(print_slices)
        if n_slices == 0:
            print_slices = [int(patients[0][0]["img"].shape[0]/2)]
            n_slices = 1
    else:
        print_slices = list(np.argwhere(np.max(patients[0][0]["mask"][...,0].numpy(), axis=(1,2)) == 2).flatten())
        n_slices = len(print_slices)
        if n_slices == 0:
            print_slices = [int(patients[0][0]["img"].shape[0]/2)]
            n_slices = 1
        print(print_slices)
        
    # outline definition
    color1 = colorConverter.to_rgba('red',alpha=0.0)
    color2 = colorConverter.to_rgba('red',alpha=0.8)
    cmap1 = LinearSegmentedColormap.from_list('my_cmap',[color1,color2],256)
    colorA = colorConverter.to_rgba('blue',alpha=0)
    colorB = colorConverter.to_rgba('green',alpha=1)
    colorC = colorConverter.to_rgba('red',alpha=1)
    cmap2 = LinearSegmentedColormap.from_list('my_cmap',[colorA,colorB,colorC],256)
    plt.rcParams['figure.figsize'] = [5*n_cols, 5*n_slices]
    
    
    n = 0
    fig, axes = plt.subplots(n_slices,n_cols)
    for j in range(len(patients)):
        i = patients[j]
        # definition of slices if multiple patients
        if n_patients > 1:
            print_slices = [0]

        for z in print_slices:
            # create figure
            maskcut = np.ones_like(np.flipud(i[0]["mask"][z,...,0].T)>=1)
            if only == False:
                axes[n,0].set_title('Diffusion imaging (b0)')
                axes[n,0].set_ylabel(train_names[int(i[0]["patindex"][0,0])]+"_q"+str(int(i[0]["meta"][0,2]))+"_z"+str(int(i[0]["meta"][0,1])))
            axes[n,0].imshow(maskcut*(1+np.flipud(i[0]["img"][z,...,0].T)),cmap='gray',vmin=0.2,vmax=1.4)

            if only == False:
                axes[n,1].set_title('Diffusion imaging (b1000)')
            axes[n,1].imshow(maskcut*(1+np.flipud(i[0]["img"][z,...,1].T)),cmap='gray',vmin=0.2,vmax=1.4)
            roi = ndimage.laplace(ndimage.binary_dilation(np.flipud(i[0]["mask"][z,...,0].T)>1.5, iterations=4))
            axes[n,1].imshow(roi, cmap=cmap1, alpha=0.5)
            
            if only == "real" or only == False:
                if only == False:
                    axes[n,2].set_title('T2eg imaging')        
                axes[n,2].imshow(maskcut*(1+np.flipud(i[1][z][...,0].T)),cmap='gray',vmin=0.2,vmax=1.4)

            # create synthetic T2eg
            if only == "synth" or only == False:
                if not multiquality:
                    if maxquality:
                        qualities = [3]
                    else:
                        qualities = [i[0]["meta"][...,2][0]]
                for q in range(len(qualities)):
                    qualarr = np.tile(qualities[q], (i[0]["img"].shape[0],1))
                    fsarr = np.tile(0, (i[0]["img"].shape[0],1))
                    synthT2EG, synthBlob, synthPred = generator.predict([i[0]["img"][:,:,:,np.newaxis,:], qualarr, fsarr])
                    synthT2EG = np.reshape(synthT2EG,(synthT2EG.shape[0],256,256))
                    if only == False:
                        syntext = 'Synthetic T2EG (model created)' 
                        if multiquality:
                            syntext = 'Synthetic T2EG (qual'+str(qualities[q])+')'
                        axes[n,st2egpos+q].set_title(syntext)
                    axes[n,st2egpos+q].imshow(maskcut*(1+np.flipud(synthT2EG[z].T)),cmap='gray',vmin=0.2,vmax=1.4)

                if show_pred :
                    axes[n,st2egpos+len(qualities)].set_title("Hematoma " + str(int(i[0]["meta"][z,...,3])))
                    realBlob = i[0]["preclean"][z,...,0]+i[0]["preclean"][z,...,1]+(i[0]["preclean"][z,...,2]>2)
                    axes[n,st2egpos+len(qualities)].imshow(np.flipud(realBlob.T),vmin=0,vmax=3)
                    axes[n,st2egpos+len(qualities)+1].set_title("Hematoma Pred")
                    synthBlob = i[0]["preclean"][z,...,0]+synthBlob[z,...,0]+synthPred[z,...,0]
                    axes[n,st2egpos+len(qualities)+1].imshow(np.flipud(synthBlob.T),vmin=0,vmax=3)
                    
            n += 1
    fig1 = plt.gcf()
    # Show and/or save
    if show:
        plt.show()
    if save:
        fig1.savefig(os.path.join(output,savestr+'.png'))
    if buffer:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig1)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        del buf, fig1
        return image
    if not show:
        plt.close()
    if save:
        return "Saved to " + os.path.join(output,savestr+'.png')
