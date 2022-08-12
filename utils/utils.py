import numpy as np
import matplotlib.pyplot as plt
from data_prepare_clipped_large_fields import prepare_directional_filters_std
import eff_multi_task_unet

class Prediction:
    i = 1
    def __init__(self,**kwargs):
        #Prediction.i +=1
        self.i = Prediction.i
        self.data_file = kwargs.setdefault('data_file',None)
        self.model = kwargs.setdefault('model',None)
        self.checkpoint_path = kwargs.setdefault('checkpoint_path', "training_3/eff_multiunet_cp.ckpt")
        
        
    def left_right_std_filters(self,data_file):
        std_avg, left_conv_bands,right_conv_bands = prepare_directional_filters_std(data_file)
        std_im, leftim, rightim = np.array(std_avg), np.array(left_conv_bands), np.array(right_conv_bands)
        shape = leftim.shape
        print(shape)
        std_im = np.expand_dims(std_im,axis=0)
        inpdata = np.concatenate((leftim,rightim,std_im),axis=0)
        #np.save('dir_filters_std(large_input).npy',inpdata)
        test = inpdata
        test = np.swapaxes(test, 0, 2)
        test = np.swapaxes(test, 0, 1)
        test = np.expand_dims(test,axis=0)
        return test
    def load_model_weights(self,input_shape,checkpoint_path):
        
        self.model = eff_multi_task_unet.build_effienet_unet(input_shape,self.i)
        return self.model.load_weights(checkpoint_path)
        
    def pred_data(self):
        
        test = self.left_right_std_filters(self.data_file)
        #input_shape = (1536,2048,33)#for large input
        
        input_shape = test[0].shape
        if self.model==None:
            self.model = eff_multi_task_unet.build_effienet_unet(input_shape,self.i)
            self.model.load_weights(self.checkpoint_path)
        ##Test model in a bigger image
        y_pred=self.model.predict(test) #channel 0 is boundary, channel 1 is field_mask, channel 2 is extent_mask
        plt.subplot(1,2,1)
        plt.imshow(test[0,:,:, 32],cmap='gray')
        #plt.show()
        plt.subplot(1,2,2)
        plt.imshow(y_pred[0][0,:,:,0],cmap='gray')
        #plt.imshow(y_pred[0,:,:,0],cmap='gray')
        plt.show()
        
        #return test
    def mIoU(self,x_test,y_test):
        input_shape = x_test[0].shape
        if self.model==None:
            self.model = eff_multi_task_unet.build_effienet_unet(input_shape,self.i)
            self.model.load_weights(self.checkpoint_path)
        y_pred=self.model.predict(x_test)
        pred_mask_img_test = y_pred[0]
        y_pred = pred_mask_img_test

        y_pred_thresholded = y_pred > 0.5
        y_pred_thresholded = y_pred_thresholded[:,:,:,0]

        intersection = np.logical_and(y_test, y_pred_thresholded)
        union = np.logical_or(y_test, y_pred_thresholded)
        iou_score = np.sum(intersection) / np.sum(union)
        print("IoU socre is: ", iou_score)

    def loss_plots(self,model_history):

        plt.figure(figsize=(13, 6), dpi=14)
        plt.subplot(2,2,1)
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('mtl_loss')
        plt.ylabel('val_loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.subplot(2,2,2)
        plt.plot(model_history.history['segment_boundary_loss'])
        plt.plot(model_history.history['val_segment_boundary_loss'])
        plt.title('Boundary Loss')
        plt.ylabel('val_loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        
        plt.subplot(2,2,3)
        plt.plot(model_history.history['segment_extent_loss'])
        plt.plot(model_history.history['val_segment_extent_loss'])
        plt.title('Segment Extent Loss')
        plt.ylabel('val_loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        
        plt.subplot(2,2,4)
        plt.plot(model_history.history['segment_field_loss'])
        plt.plot(model_history.history['val_segment_field_loss'])
        plt.title('Segmentation Loss')
        plt.ylabel('val_loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

"""
data_file_large = 'Clipped_large_fields.tif'
data_file_ = ['clipped_small_parcels.tif', 'clipped_small_parcels_3.tif','clipped_small_parcels_4.tif']

for file in data_file_: 
    p = Prediction(data_file=file)
    p.pred_data()
Prediction.i+=3
"""
