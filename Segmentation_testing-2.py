from Preprocessing import *

parser = argparse.ArgumentParser()

parser.add_argument('--dicom_nifti', type=str,
                    help='A required array ')

parser.add_argument('--roi', type=str,
                    help='Required ROI Images array')

args = parser.parse_args()     

def Segmentation():
    
    out_nii_array,h=extract_data_(args.dicom_nifti)
    #out_nii_array,h=extract_data_(args.dicom_nifti,args.roi)

    g_=[]
    for i in range(out_nii_array.shape[-1]):
        g_.append(resize(out_nii_array[0:,0:,i],(256,256)))
    out_nii_array=np.stack(g_).transpose(1,2,0)

#     out_nii_array,out_nii_roi,h=extract_data(args.dicom_nifti,args.roi)

#     for i in range(out_nii_array.shape[-1]):
#         out_nii_array[0:,0:,i]=np.resize(out_nii_array[0:,0:,i],(256,256))
#         out_nii_roi[0:,0:,i]=np.resize(out_nii_roi[0:,0:,i],(256,256))

    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('./best_model_276E_96I.pth'):
        best_model=download()
    else:
        best_model = torch.load('./best_model_276E_96I.pth')
        print('Loaded UNet model from this run.')

    dc=[]
    io=[]
    a=[]
    array=[]
    test_dataset_1=CustomDataset(np.transpose(out_nii_array))
    #test_dataset_1=CustomDataset(np.transpose(out_nii_array),np.transpose(out_nii_roi))

    for n in range(len(test_dataset_1)):
        image_vis=np.transpose(out_nii_array)[n]
        #gt_mask1 = test_dataset_1[n][1].squeeze().cpu().numpy()
        pr_mask1=best_model.predict(test_dataset_1[n].to(DEVICE).unsqueeze(0)).squeeze().cpu().numpy()
        array.append(pr_mask1)
        try:
            contours, _ = cv2.findContours(pr_mask1.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            biggest_contour = max(contours, key = cv2.contourArea)
            area = cv2.contourArea(biggest_contour)
            a.append(area)
        except:
            a.append(int(0.0))
        #dc.append(DICE_COE(gt_mask1, pr_mask1))
        #io.append(IOU(gt_mask1, pr_mask1))
#     df=pd.DataFrame(columns=['Name', 'Dice_Coefficient','Iou_score','Area'])
#     df['Name'],df['Dice_Coefficient'],df['Iou_score'],df['Area']=h,dc,io,a
#     df.to_csv('./study_results.csv')
    if not os.path.exists(args.dicom_nifti.split('/')[-1].split('.')[0]):
        os.makedirs(args.dicom_nifti.split('/')[-1].split('.')[0])
    ni_img = nib.Nifti1Image(np.array(array).transpose(), affine=np.eye(4))
    nib.save(ni_img, './'+str(args.dicom_nifti.split('/')[-1].split('.')[0])+'/'+str(args.dicom_nifti.split('/')[-1].split('.')[0])+'.nii')
    df=pd.DataFrame(columns=['Name','Area'])
    df['Name'],df['Area']=h,a
    df.to_csv('./'+str(args.dicom_nifti.split('/')[-1].split('.')[0])+'/'+'study_results.csv')

if __name__=='__main__':
    Segmentation()