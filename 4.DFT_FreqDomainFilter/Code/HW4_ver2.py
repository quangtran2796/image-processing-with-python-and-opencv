import cv2
import numpy as np

Original_gray=cv2.imread('Original_2.png',0)

#Chuan bi ngo vao input cho ham dft, cu the la tao 1 mang 2 lop chua phan thuc va phan ao, "originalComplex"
Original_gray_1=np.float32(Original_gray)
Original_Float=Original_gray_1/255.0
originalComplex=np.zeros((Original_Float.shape[0],Original_Float.shape[1],2))
originalComplex[:,:,0]=Original_Float
#A=np.array([[1,2,3],[4,5,6]])
#B=np.array([[7,8,9],[10,11,12]])
#abc=cv2.merge((A,B))
#C=A*B


#Sau khi ngo vao input ready, thuc hien bien doi DFT, ngo ra output la "dftOriginal"
dftOriginal=np.zeros((Original_Float.shape[0],Original_Float.shape[1],2))
cv2.dft(originalComplex,dftOriginal,16)


#Sau khi co DFT dang complex la "dftOriginal", thuc hien lay bien do, dung ham cv2.magnitude, ngo ra la "dftMagnitude"
dftMagnitude=np.zeros((Original_Float.shape[0],Original_Float.shape[1]))
cv2.magnitude(dftOriginal[:,:,0],dftOriginal[:,:,1],dftMagnitude)
dftMagnitude=dftMagnitude+1
cv2.log(dftMagnitude,dftMagnitude)
cv2.normalize(dftMagnitude,dftMagnitude,0,1,32)

#Lay kich thuoc cua pho bien do
dftMagnitude_centerX = dftMagnitude.shape[1]/2
dftMagnitude_centerY = dftMagnitude.shape[0]/2
dftMagnitude_2=np.zeros((dftMagnitude.shape[0],dftMagnitude.shape[1]))

#Sau khi co DFT, thuc hien Recenter DFT cho dftMagnitude, ngo ra se la dftMagnitude_2
dftMagnitude_2[0:dftMagnitude_centerY,0:dftMagnitude_centerX]=dftMagnitude[dftMagnitude_centerY:dftMagnitude_centerY*2,dftMagnitude_centerX:dftMagnitude_centerX*2]
dftMagnitude_2[dftMagnitude_centerY:dftMagnitude_centerY*2,dftMagnitude_centerX:dftMagnitude_centerX*2]=dftMagnitude[0:dftMagnitude_centerY,0:dftMagnitude_centerX]
dftMagnitude_2[dftMagnitude_centerY:dftMagnitude_centerY*2,0:dftMagnitude_centerX]=dftMagnitude[0:dftMagnitude_centerY,dftMagnitude_centerX:dftMagnitude_centerX*2]
dftMagnitude_2[0:dftMagnitude_centerY,dftMagnitude_centerX:dftMagnitude_centerX*2]=dftMagnitude[dftMagnitude_centerY:dftMagnitude_centerY*2,0:dftMagnitude_centerX]


#Luu anh
DFT_Original=cv2.convertScaleAbs(dftMagnitude,0,255)
DFT_Recentered=cv2.convertScaleAbs(dftMagnitude_2,0,255)


cv2.imwrite('Original_Image_Gray.jpg',Original_gray)
cv2.imwrite('DFT_Original.jpg',DFT_Original)
cv2.imwrite('DFT_Recentered.jpg',DFT_Recentered)


#Tao window
cv2.namedWindow('Original_Image', cv2. WINDOW_NORMAL)
cv2.namedWindow('DFT_Original', cv2. WINDOW_NORMAL)
cv2.namedWindow('DFT_Recentered', cv2. WINDOW_NORMAL)
#Hien thi anh gray goc
cv2.imshow('Original_Image',Original_gray)
#Hien thi bien doi DFT dang bien do magnitude
cv2.imshow('DFT_Original',dftMagnitude)
#Hien thi bien doi DFT dang bien do magnitude sau khi recenter
cv2.imshow('DFT_Recentered',dftMagnitude_2)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
def Low_Pass_Filter():   
    #Tao mask loc thong thap, ngo ra la "mask_filter_complex" la man tran 2 lop, bao gom phan thuc va phan ao
    mask_filter=np.zeros((dftMagnitude.shape[0],dftMagnitude.shape[1]))
    #ban kinh bo loc la "D_0", tuc ban kinh bo loc = D_0 pixel tinh tu trung tam
    D_0=inner_radius_axis.get()
    if D_0 != 0:
        cv2.circle(mask_filter,(dftMagnitude_centerX,dftMagnitude_centerY),D_0,1, -1)
    mask_filter_complex=np.zeros((dftMagnitude.shape[0],dftMagnitude.shape[1],2))
    mask_filter_complex[:,:,0]=mask_filter
    mask_filter_complex[:,:,1]=mask_filter

    #De loc thong thap, ta can recenter DFT cua anh dang complex(khong phai recenter bien do dftMagnitude nhu o tren), do la bien dftOriginal, bien nay chua dang complex
    dftOriginal_2=np.zeros((dftOriginal.shape[0],dftOriginal.shape[1],2))
    dftOriginal_2[0:dftMagnitude_centerY,0:dftMagnitude_centerX,:]=dftOriginal[dftMagnitude_centerY:dftMagnitude_centerY*2,dftMagnitude_centerX:dftMagnitude_centerX*2,:]
    dftOriginal_2[dftMagnitude_centerY:dftMagnitude_centerY*2,dftMagnitude_centerX:dftMagnitude_centerX*2,:]=dftOriginal[0:dftMagnitude_centerY,0:dftMagnitude_centerX,:]
    dftOriginal_2[dftMagnitude_centerY:dftMagnitude_centerY*2,0:dftMagnitude_centerX,:]=dftOriginal[0:dftMagnitude_centerY,dftMagnitude_centerX:dftMagnitude_centerX*2,:]
    dftOriginal_2[0:dftMagnitude_centerY,dftMagnitude_centerX:dftMagnitude_centerX*2,:]=dftOriginal[dftMagnitude_centerY:dftMagnitude_centerY*2,0:dftMagnitude_centerX,:]

    #Loc thong thap, ngo ra la "Post_Filter"
    Post_Filter=dftOriginal_2*mask_filter_complex

    #Sau khi loc, tien hanh bien doi nguoc IDFT khoi phuc anh ban dau
    Inverse_DFT=np.zeros((Original_Float.shape[0],Original_Float.shape[1],2))
    Inverse_DFT=cv2.idft(Post_Filter)

    #De xem duoc anh, ta thuc hien lay bien do, dung ham "cv2.magnitude", input la Inverse_DFT, output la idftMagnitude, sau do scale ve 0 : 1
    idftMagnitude=np.zeros((Original_Float.shape[0],Original_Float.shape[1]))
    cv2.magnitude(Inverse_DFT[:,:,0],Inverse_DFT[:,:,1],idftMagnitude)
    cv2.normalize(idftMagnitude,idftMagnitude,0,1,32)

    #Luu anh
    Mask_Low_Pass_Filter=cv2.convertScaleAbs(mask_filter,0,255)
    Inverse_DFT_After_Low_Filterd=cv2.convertScaleAbs(idftMagnitude,0,255)
    cv2.imwrite('Mask_Low_Pass_Filter.jpg',Mask_Low_Pass_Filter)
    cv2.imwrite('Low_Pass_Filtered.jpg',Inverse_DFT_After_Low_Filterd)

    #Hien thi mask bo loc co ban kinh D_0
    cv2.namedWindow('Mask Low Pass Filter', cv2. WINDOW_NORMAL)
    cv2.imshow('Mask Low Pass Filter',mask_filter)
    #Hien thi anh khi (dung bien doi IDFT) sau khi anh da duoc loc thong thap
    cv2.namedWindow('Low Pass Filter', cv2. WINDOW_NORMAL)
    cv2.imshow('Low Pass Filter',idftMagnitude)

    print(outer_radius_axis.get())
def High_Pass_Filter():
    #Tao mask loc thong cao, ngo ra la "mask_filter_complex" la man tran 2 lop, bao gom phan thuc va phan ao
    mask_filter=np.ones((dftMagnitude.shape[0],dftMagnitude.shape[1]))
    #ban kinh bo loc la "D_0", tuc ban kinh bo loc = D_0 pixel tinh tu trung tam
    D_0=inner_radius_axis.get()
    if D_0 != 0:
        cv2.circle(mask_filter,(dftMagnitude_centerX,dftMagnitude_centerY),D_0,0, -1)
    mask_filter_complex=np.zeros((dftMagnitude.shape[0],dftMagnitude.shape[1],2))
    mask_filter_complex[:,:,0]=mask_filter
    mask_filter_complex[:,:,1]=mask_filter

    #De loc thong cao, ta can recenter DFT cua anh dang complex(khong phai recenter bien do dftMagnitude nhu o tren), do la bien dftOriginal, bien nay chua dang complex
    dftOriginal_2=np.zeros((dftOriginal.shape[0],dftOriginal.shape[1],2))
    dftOriginal_2[0:dftMagnitude_centerY,0:dftMagnitude_centerX,:]=dftOriginal[dftMagnitude_centerY:dftMagnitude_centerY*2,dftMagnitude_centerX:dftMagnitude_centerX*2,:]
    dftOriginal_2[dftMagnitude_centerY:dftMagnitude_centerY*2,dftMagnitude_centerX:dftMagnitude_centerX*2,:]=dftOriginal[0:dftMagnitude_centerY,0:dftMagnitude_centerX,:]
    dftOriginal_2[dftMagnitude_centerY:dftMagnitude_centerY*2,0:dftMagnitude_centerX,:]=dftOriginal[0:dftMagnitude_centerY,dftMagnitude_centerX:dftMagnitude_centerX*2,:]
    dftOriginal_2[0:dftMagnitude_centerY,dftMagnitude_centerX:dftMagnitude_centerX*2,:]=dftOriginal[dftMagnitude_centerY:dftMagnitude_centerY*2,0:dftMagnitude_centerX,:]

    #Loc thong cao, ngo ra la "Post_Filter"
    Post_Filter=dftOriginal_2*mask_filter_complex

    #Sau khi loc, tien hanh bien doi nguoc IDFT khoi phuc anh ban dau
    Inverse_DFT=np.zeros((Original_Float.shape[0],Original_Float.shape[1],2))
    Inverse_DFT=cv2.idft(Post_Filter)

    #De xem duoc anh, ta thuc hien lay bien do, dung ham "cv2.magnitude", input la Inverse_DFT, output la idftMagnitude, sau do scale ve 0 : 1
    idftMagnitude=np.zeros((Original_Float.shape[0],Original_Float.shape[1]))
    cv2.magnitude(Inverse_DFT[:,:,0],Inverse_DFT[:,:,1],idftMagnitude)
    cv2.normalize(idftMagnitude,idftMagnitude,0,1,32)

    #Luu anh
    Mask_High_Pass_Filter=cv2.convertScaleAbs(mask_filter,0,255)
    Inverse_DFT_After_High_Filterd=cv2.convertScaleAbs(idftMagnitude,0,255)
    cv2.imwrite('Mask_High_Pass_Filter.jpg',Mask_High_Pass_Filter)
    cv2.imwrite('High_Pass_Filtered.jpg',Inverse_DFT_After_High_Filterd)

    #Hien thi mask bo loc co ban kinh D_0
    cv2.namedWindow('Mask_High_Pass_Filter', cv2. WINDOW_NORMAL)
    cv2.imshow('Mask_High_Pass_Filter',mask_filter)
    #Hien thi anh khi (dung bien doi IDFT) sau khi anh da duoc loc thong thap
    cv2.namedWindow('High Pass Filter', cv2. WINDOW_NORMAL)
    cv2.imshow('High Pass Filter',idftMagnitude)
    
def Band_Pass_Filter():
    #Tao mask loc band pass, ngo ra la "mask_filter_complex" la man tran 2 lop, bao gom phan thuc va phan ao
    mask_filter=np.ones((dftMagnitude.shape[0],dftMagnitude.shape[1]))
    mask_filter_2=np.zeros((dftMagnitude.shape[0],dftMagnitude.shape[1]))
    mask_filter_3=np.zeros((dftMagnitude.shape[0],dftMagnitude.shape[1]))
    #inner_radius = D_0 , outer_radius = D_1
    D_0=inner_radius_axis.get()
    D_1=outer_radius_axis.get()
    if D_0 != 0:
        cv2.circle(mask_filter,(dftMagnitude_centerX,dftMagnitude_centerY),D_0,0, -1)
    if D_1 != 0:
        cv2.circle(mask_filter_2,(dftMagnitude_centerX,dftMagnitude_centerY),D_1,1, -1)
    mask_filter_3=mask_filter*mask_filter_2
    mask_filter_complex=np.zeros((dftMagnitude.shape[0],dftMagnitude.shape[1],2))
    mask_filter_complex[:,:,0]=mask_filter
    mask_filter_complex[:,:,1]=mask_filter
    mask_filter_complex_2=np.zeros((dftMagnitude.shape[0],dftMagnitude.shape[1],2))
    mask_filter_complex_2[:,:,0]=mask_filter_2
    mask_filter_complex_2[:,:,1]=mask_filter_2
    mask_filter_complex_3=np.zeros((dftMagnitude.shape[0],dftMagnitude.shape[1],2))
    mask_filter_complex_3=mask_filter_complex*mask_filter_complex_2

    #De loc band pass, ta can recenter DFT cua anh dang complex(khong phai recenter bien do dftMagnitude nhu o tren), do la bien dftOriginal, bien nay chua dang complex
    dftOriginal_2=np.zeros((dftOriginal.shape[0],dftOriginal.shape[1],2))
    dftOriginal_2[0:dftMagnitude_centerY,0:dftMagnitude_centerX,:]=dftOriginal[dftMagnitude_centerY:dftMagnitude_centerY*2,dftMagnitude_centerX:dftMagnitude_centerX*2,:]
    dftOriginal_2[dftMagnitude_centerY:dftMagnitude_centerY*2,dftMagnitude_centerX:dftMagnitude_centerX*2,:]=dftOriginal[0:dftMagnitude_centerY,0:dftMagnitude_centerX,:]
    dftOriginal_2[dftMagnitude_centerY:dftMagnitude_centerY*2,0:dftMagnitude_centerX,:]=dftOriginal[0:dftMagnitude_centerY,dftMagnitude_centerX:dftMagnitude_centerX*2,:]
    dftOriginal_2[0:dftMagnitude_centerY,dftMagnitude_centerX:dftMagnitude_centerX*2,:]=dftOriginal[dftMagnitude_centerY:dftMagnitude_centerY*2,0:dftMagnitude_centerX,:]

    #Loc band pass, ngo ra la "Post_Filter"
    Post_Filter=dftOriginal_2*mask_filter_complex_3

    #Sau khi loc, tien hanh bien doi nguoc IDFT khoi phuc anh ban dau
    Inverse_DFT=np.zeros((Original_Float.shape[0],Original_Float.shape[1],2))
    Inverse_DFT=cv2.idft(Post_Filter)

    #De xem duoc anh, ta thuc hien lay bien do, dung ham "cv2.magnitude", input la Inverse_DFT, output la idftMagnitude, sau do scale ve 0 : 1
    idftMagnitude=np.zeros((Original_Float.shape[0],Original_Float.shape[1]))
    cv2.magnitude(Inverse_DFT[:,:,0],Inverse_DFT[:,:,1],idftMagnitude)
    cv2.normalize(idftMagnitude,idftMagnitude,0,1,32)

    #Luu anh
    Mask_Band_Pass_Filter=cv2.convertScaleAbs(mask_filter_3,0,255)
    Inverse_DFT_After_Band_Pass_Filterd=cv2.convertScaleAbs(idftMagnitude,0,255)
    cv2.imwrite('Mask_Band_Pass_Filter.jpg',Mask_Band_Pass_Filter)
    cv2.imwrite('Band_Pass_Filtered.jpg',Inverse_DFT_After_Band_Pass_Filterd)

    #Hien thi mask bo loc co ban kinh D_0
    cv2.namedWindow('Mask Band Pass Filter', cv2. WINDOW_NORMAL)
    cv2.imshow('Mask Band Pass Filter',mask_filter_3)
    #Hien thi anh khi (dung bien doi IDFT) sau khi anh da duoc loc thong thap
    cv2.namedWindow('Band Pass Filter', cv2. WINDOW_NORMAL)
    cv2.imshow('Band Pass Filter',idftMagnitude)


from Tkinter import *

root=Tk()

thelabel = Label(root, text="Adjust Paramter and Select Your filter",bg="yellow", fg="blue")
thelabel.grid()

inner_radius_label=Label(root,text="Inner radius")
inner_radius_label.grid(row=1,column=0)
inner_radius_axis = Scale(root, from_=0, to=200, orient=HORIZONTAL)
inner_radius_axis.grid(row=1,column=1)

outer_radius_label=Label(root,text="Outer radius")
outer_radius_label.grid(row=2,column=0)
outer_radius_axis = Scale(root, from_=0, to=200, orient=HORIZONTAL)
outer_radius_axis.grid(row=2,column=1)

Low_Pass_Filter  = Button(root, text="LOW PASS FILTER",bg="white",fg="blue",command=Low_Pass_Filter)
High_Pass_Filter = Button(root, text="HIGH PASS FILTER",bg="white",fg="blue",command=High_Pass_Filter)
Band_Pass_Filter = Button(root, text="BAND PASS FILTER",bg="white",fg="blue",command=Band_Pass_Filter)

Low_Pass_Filter.grid()
High_Pass_Filter.grid()
Band_Pass_Filter.grid()

root.mainloop()

