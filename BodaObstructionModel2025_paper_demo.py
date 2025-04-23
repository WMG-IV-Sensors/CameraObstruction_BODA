# BODA Obstruction model
# 27/03/2025
import numpy as np
import cv2
from scipy import ndimage
import random
import glob
import os

random.seed(a=None)

global BIT_DEPTH
global IMG_SIZE_Y
global IMG_SIZE_X
global GRID_NUMBER_Y
global GRID_NUMBER_X
global W_ROW_IDX_1D
global W_COL_IDX_1D
global DROPS_TRANS0
global DROPS_TRANS1
global DROPS_TRANS2
global DROPS_SIZESPREAD_MEAN_NOE
global DROPS_AVG_SIZE_SCALER
global BLUR_LAYER_ON
global BLUR_KSCALE
global BLUR_KROTA
global OPAQUE_OBSTRUCTION
global OPACITY

BIT_DEPTH = 255
IMG_SIZE_Y = 960
IMG_SIZE_X = 1280
GRID_NUMBER_Y = 10
GRID_NUMBER_X = 10
W_ROW_IDX_1D = np.arange(0,GRID_NUMBER_Y,1,dtype=int)
W_COL_IDX_1D = np.arange(0,GRID_NUMBER_X,1,dtype=int)
DROPS_TRANS0 = 0.25 #0.75
DROPS_TRANS1 = 0.125 #0.125
DROPS_TRANS2 = 0.5 #0.75
DROPS_SIZESPREAD_MEAN_NOE = 10
DROPS_AVG_SIZE_SCALER = 1
BLUR_LAYER_ON = 1
BLUR_KSCALE = 0.01
BLUR_KROTA = 30
OPAQUE_OBSTRUCTION = 1
OPACITY = 0.6

w = np.int64(np.zeros([GRID_NUMBER_Y,GRID_NUMBER_X]))
w_row_idx_start = int(IMG_SIZE_Y/GRID_NUMBER_Y)*W_ROW_IDX_1D
w_row_idx_finish = int(IMG_SIZE_Y/GRID_NUMBER_Y)*(W_ROW_IDX_1D+1)
w_col_idx_start = int(IMG_SIZE_X/GRID_NUMBER_X)*W_COL_IDX_1D
w_col_idx_finish = int(IMG_SIZE_X/GRID_NUMBER_X)*(W_COL_IDX_1D+1)
w_row_2didx_start = np.transpose(np.tile(w_row_idx_start, (GRID_NUMBER_X,1)))
w_col_2didx_start = np.tile(w_col_idx_start, (GRID_NUMBER_Y,1))

w_0 = np.zeros([GRID_NUMBER_Y,GRID_NUMBER_X])
w_1 = np.zeros([GRID_NUMBER_Y,GRID_NUMBER_X])
w_2 = np.zeros([GRID_NUMBER_Y,GRID_NUMBER_X])

# prepare the canvas and load patches
obs_canvas = np.int32(np.zeros([IMG_SIZE_Y, IMG_SIZE_X]))

# TODO: Up to three channels of obstructions can be loaded as masks, masks can be in greyscale .png
msk_0 = cv2.imread('mskg.png')[:,:,0] # example
msk_1 = cv2.imread('mskg.png')[:,:,0] # example
msk_2 = cv2.imread('mskg.png')[:,:,0] # example
#

def zoomRot(img, a, b, omega):
    
    if len(img.shape)==2:
        rows,cols = img.shape
    else:
        rows,cols,_ = img.shape
    
    M = cv2.getRotationMatrix2D((cols/2, rows/2),omega,1)
    rot = cv2.warpAffine(img,M,(cols, rows))

    if a <= 0:
        a = 1
    
    if b <= 0:
        b = 1

    resRot = cv2.resize(rot, None, fx=a, fy=b, interpolation = cv2.INTER_CUBIC)
    
    return resRot

def waterblur(img_input,kernel_input,channel):
    img = np.int32(img_input)[:,:,channel]
    kernel = DROPS_SIZESPREAD_MEAN_NOE*kernel_input[:,:,0]/kernel_input.size
    dstImg = np.uint8(np.clip(ndimage.convolve(img,kernel,mode='constant',cval=0.0),0,BIT_DEPTH))
    return dstImg
 
def weightImginGrids(img,
                     w_row_idx_start, w_row_idx_finish, 
                     w_col_idx_start, w_col_idx_finish, 
                      w_0, w_1, w_2 ):
    # get image weights in samples
    for i in W_ROW_IDX_1D:
        for j in W_COL_IDX_1D:
            w[i,j] = np.sum(img[w_row_idx_start[i]:w_row_idx_finish[i],w_col_idx_start[j]:w_col_idx_finish[j]])
    
    th = int(IMG_SIZE_Y/GRID_NUMBER_Y*IMG_SIZE_X/GRID_NUMBER_X*BIT_DEPTH)
    th0 = int(th*3/4)
    th1 = int(th0/2)
    
    w_0[w>=th0] = w[w>=th0]
    w_0 = np.int32(w_0)
    w_1[w>=th1] = w[w>=th1]
    w_1[w>=th0] = 0
    w_1 = np.int32(w_1)
    w_2[w<th1] = w[w<th1]
    w_2 = np.int32(w_2)

    if OPAQUE_OBSTRUCTION == 1:
        w_0 = (w_0 - th/3)*OPACITY
        w_1 = (w_1 - th/3)*OPACITY
        w_2 = (w_2 - th/3)*OPACITY

    return w_0, w_1, w_2

def randomPatch(w_row_2didx_start,w_col_2didx_start):
    # get random patch position
    patch_center_row_relatidx = np.int32(np.random.rand(GRID_NUMBER_Y, GRID_NUMBER_X)*IMG_SIZE_Y/GRID_NUMBER_Y)
    patch_center_col_relatidx = np.int32(np.random.rand(GRID_NUMBER_Y, GRID_NUMBER_X)*IMG_SIZE_X/GRID_NUMBER_X)
    patch_center_row_idx = patch_center_row_relatidx + w_row_2didx_start
    patch_center_col_idx = patch_center_col_relatidx + w_col_2didx_start

    # get random patch size and shape
    patch_size_shape = np.random.poisson(lam = DROPS_SIZESPREAD_MEAN_NOE, size = (GRID_NUMBER_Y,GRID_NUMBER_X,2))/DROPS_SIZESPREAD_MEAN_NOE*DROPS_AVG_SIZE_SCALER
    patch_size_orient = np.int32(np.random.rand(GRID_NUMBER_Y, GRID_NUMBER_X)*360)
    patch_scale_x = np.float32(patch_size_shape[:,:,1])
    patch_scale_y = np.float32(patch_size_shape[:,:,0])

    return patch_center_row_idx, patch_center_col_idx, patch_scale_x, patch_scale_y, patch_size_orient

def clipPatchWithinImg(halfMaskWidth,halfMaskHeight,patchCenterinImgX,patchCenterinImgY):
    # Calculate the boundary of the noisy patch in clean images
    topleft_ROW = np.maximum(0, patchCenterinImgY - halfMaskHeight)
    topleft_COL = np.maximum(0, patchCenterinImgX - halfMaskWidth)
    bottomright_ROW = np.minimum(IMG_SIZE_Y, patchCenterinImgY + halfMaskHeight)
    bottomright_COL = np.minimum(IMG_SIZE_X, patchCenterinImgX + halfMaskWidth)
    
    # Calculate the boundary of the noisy patch in noisy patch
    topleft_row = np.maximum(0, halfMaskHeight - patchCenterinImgY )
    topleft_col = np.maximum(0, halfMaskWidth - patchCenterinImgX )
    bottomright_row = np.minimum(halfMaskHeight + (IMG_SIZE_Y-patchCenterinImgY), 2*halfMaskHeight)
    bottomright_col = np.minimum(halfMaskWidth + (IMG_SIZE_X-patchCenterinImgX), 2*halfMaskWidth)

    return topleft_ROW, topleft_COL, bottomright_ROW, bottomright_COL, topleft_row, topleft_col, bottomright_row, bottomright_col

# TODO: Change directory to grab input images and specify output images directory
path_inputImg = 'F:/NoiseModel_testing/BodaObstructionModel/input' # example
path_outputImg = 'F:/NoiseModel_testing/BodaObstructionModel/output' # example
path_images = '*.png' # example
# 

# TODO: Choose to load mask as the blur kernel (optional)
transMsk = BLUR_LAYER_ON*cv2.imread('mskg.png') # example
#

os.chdir(path_inputImg)
files_img = glob.glob(path_images)

for img in files_img:
    img_org = cv2.imread(img)
    print(img)
    dstImg = np.uint8(np.zeros([IMG_SIZE_Y,IMG_SIZE_X,3]))
    obs_canvas_output = np.int32(np.zeros([IMG_SIZE_Y,IMG_SIZE_X]))
    obs_canvas = np.int32(np.zeros([IMG_SIZE_Y,IMG_SIZE_X]))
    obs_dst_img = np.uint8(np.zeros([IMG_SIZE_Y,IMG_SIZE_X,3]))

    img_int64 = np.int64(img_org[:,:,0])
    kernel_input = 1/(np.max(transMsk)+1)*zoomRot(transMsk,BLUR_KSCALE,BLUR_KSCALE,BLUR_KROTA)
    dstImg[:,:,0] = waterblur(img_org,kernel_input,0)
    dstImg[:,:,1] = waterblur(img_org,kernel_input,1)
    dstImg[:,:,2] = waterblur(img_org,kernel_input,2)

    w_0, w_1, w_2 = weightImginGrids(img_int64,
                                    w_row_idx_start, w_row_idx_finish, 
                                    w_col_idx_start, w_col_idx_finish,
                                    w_0, w_1, w_2 )

    patch_center_row_idx, patch_center_col_idx, patch_scale_x, patch_scale_y, patch_size_orient = randomPatch(w_row_2didx_start, w_col_2didx_start)

    msk_w = np.int32(patch_scale_x*int(transMsk.shape[1]))
    msk_h = np.int32(patch_scale_y*int(transMsk.shape[0]))

    msk_w_half = np.floor(msk_w*0.5).astype(int)
    msk_h_half = np.floor(msk_h*0.5).astype(int)

    topleft_ROW, topleft_COL, bottomright_ROW, bottomright_COL, topleft_row, topleft_col, bottomright_row, bottomright_col = clipPatchWithinImg(msk_w_half, 
                                                                                                                                                msk_h_half, 
                                                                                                                                                patch_center_col_idx, 
                                                                                                                                                patch_center_row_idx)

    for i in W_ROW_IDX_1D:
        for j in W_COL_IDX_1D:
            msk_0_inst = np.float32(zoomRot(msk_0,patch_scale_x[i,j],patch_scale_y[i,j],patch_size_orient[i,j])*DROPS_TRANS0)
            msk_0_inst = np.int32(w_0[i,j]/(int(IMG_SIZE_Y/GRID_NUMBER_Y)*int(IMG_SIZE_X/GRID_NUMBER_X)*BIT_DEPTH)*msk_0_inst)
            
            # patch DROPS LAYER0 in canvas
            topleft_ROW_inst = topleft_ROW[i,j]
            bottomright_ROW_inst = bottomright_ROW[i,j]
            topleft_COL_inst = topleft_COL[i,j]
            bottomright_COL_inst = bottomright_COL[i,j]

            topleft_row_inst = topleft_row[i,j]
            bottomright_row_inst = bottomright_row[i,j]
            topleft_col_inst = topleft_col[i,j]
            bottomright_col_inst = bottomright_col[i,j]

            obs_canvas[topleft_ROW_inst:bottomright_ROW_inst, 
                            topleft_COL_inst:bottomright_COL_inst
                            ] = obs_canvas[
                                topleft_ROW_inst:bottomright_ROW_inst, 
                                topleft_COL_inst:bottomright_COL_inst
                                ] + msk_0_inst[
                                    topleft_row_inst:bottomright_row_inst,
                                    topleft_col_inst:bottomright_col_inst
                                    ]
            
            msk_1_inst = np.float32(zoomRot(msk_1,patch_scale_x[i,j],patch_scale_y[i,j],patch_size_orient[i,j])*DROPS_TRANS1)
            msk_1_inst = np.int32(w_1[i,j]/(int(IMG_SIZE_Y/GRID_NUMBER_Y)*int(IMG_SIZE_X/GRID_NUMBER_X)*BIT_DEPTH)*msk_1_inst)
           
            # patch DROPS LAYER1 in canvas
            obs_canvas[topleft_ROW_inst:bottomright_ROW_inst, 
                            topleft_COL_inst:bottomright_COL_inst
                            ] = obs_canvas[
                                topleft_ROW_inst:bottomright_ROW_inst, 
                                topleft_COL_inst:bottomright_COL_inst
                                ] + msk_1_inst[
                                    topleft_row_inst:bottomright_row_inst,
                                    topleft_col_inst:bottomright_col_inst
                                    ]
            msk_2_inst = np.float32(zoomRot(msk_2,patch_scale_x[i,j],patch_scale_y[i,j],patch_size_orient[i,j])*DROPS_TRANS2)
            msk_2_inst = np.int32(w_2[i,j]/(int(IMG_SIZE_Y/GRID_NUMBER_Y)*int(IMG_SIZE_X/GRID_NUMBER_X)*BIT_DEPTH)*msk_2_inst)
            
            # patch DROPS LAYER2 in canvas
            obs_canvas[topleft_ROW_inst:bottomright_ROW_inst, 
                            topleft_COL_inst:bottomright_COL_inst
                            ] = obs_canvas[
                                topleft_ROW_inst:bottomright_ROW_inst, 
                                topleft_COL_inst:bottomright_COL_inst
                                ] + msk_2_inst[
                                    topleft_row_inst:bottomright_row_inst,
                                    topleft_col_inst:bottomright_col_inst
                                    ]

            
            
    obs_canvas_output = np.int32(np.clip(obs_canvas, (-1)*BIT_DEPTH, BIT_DEPTH))

    # obs_dst_img[:,:,0] = np.uint8(np.clip(obs_canvas_output+np.int32(dstImg[:,:,0]),0,BIT_DEPTH))
    # obs_dst_img[:,:,1] = np.uint8(np.clip(obs_canvas_output+np.int32(dstImg[:,:,1]),0,BIT_DEPTH))
    # obs_dst_img[:,:,2] = np.uint8(np.clip(obs_canvas_output+np.int32(dstImg[:,:,2]),0,BIT_DEPTH))
     
    obs_dst_img[:,:,0] = np.uint8(np.clip(obs_canvas_output+np.int32(img_org[:,:,0]),0,BIT_DEPTH))
    obs_dst_img[:,:,1] = np.uint8(np.clip(obs_canvas_output+np.int32(img_org[:,:,1]),0,BIT_DEPTH))
    obs_dst_img[:,:,2] = np.uint8(np.clip(obs_canvas_output+np.int32(img_org[:,:,2]),0,BIT_DEPTH))

    cv2.imshow('obs_Img.png', obs_dst_img)

    os.chdir(path_outputImg) 
    # TODO: Save img by using cv2.imwrite here
    # cv2.imwrite(img+'_obs.png',obs_dst_img)
    os.chdir(path_inputImg) 
    

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
os.chdir(path_inputImg) 
