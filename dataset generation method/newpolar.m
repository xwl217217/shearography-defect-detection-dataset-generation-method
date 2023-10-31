function P1=newpolar(i,mask)
name1=[num2str(i),'.bmp'];
I=double(imread(name1));
FI=fft2(I);
FI1=fftshift(FI);
FI2=FI1.*mask; 
P1=ifft2(ifftshift(FI2));


