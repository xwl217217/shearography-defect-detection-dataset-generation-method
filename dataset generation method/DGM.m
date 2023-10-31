% 空间载波法计算相位
clear
clc

I1=imread('11.bmp'); %设置掩膜
I=I1;
T=fftshift(fft2(I));
imshow((1+abs(T)),[0.1e4,2e4])
[x,y] = ginput(2); % x 列，y 行
x=round(x);y=round(y);
mask=zeros(size(I1));
mask(y(1):y(2),x(1):x(2))=1;
P1=newpolar(11,mask);
P=zeros(2048,2448,7);
for i=11:17
    i
    P2=newpolar(i,mask);
    P(:,:,i-10)=P2.*conj(P1);
    A=angle(P2.*conj(P1)); %保存相位图
    ir=im2uint8(mat2gray(A));
    filename=[num2str(i),'.bmp'];
    imwrite(ir,filename);
end


%% label标注
imageLabeler %保存得到Label_1.png
%% 进行随机剪切
clear
close all
clc
load FF.mat
labelimage=imread("Label_1.png");
[x,y,z]=size(FF);
num=1;
for i=1:1
    for Z=2:z
        num
        %         dx=20+randi(200); %y 方向剪切
        %         dy=20+randi(200); %x 方向剪切
        dx=200;
        dy=100;
        FF1=FF(:,:,Z);
        FF2=circshift(FF(:,:,Z),[0,dx]);
        labelimage_cir=circshift(labelimage,[dx,dy]);
        labelimage_sum=labelimage_cir+labelimage;
        labelimage_sum=(labelimage_sum~=0);
        DF=im2uint8(mat2gray(angle(FF2.*conj(FF1))));
        filename=['a\',num2str(num),'.bmp'];
        imwrite(DF,filename);
        filename=['b\',num2str(num),'.bmp'];
        imwrite(uint8(labelimage_sum),filename);
        num=num+1;
    end

end



%% 裁剪
num=0;
for i=1:3000
    filename1=['images','\',num2str(i),'.jpg'];
    I1=imread(filename1);
    filename2=['annotations','\',num2str(i),'.png'];
    I2=imread(filename2);
    [x,y]=size(I1);
    for n=1:5
        num=num+1;
        limx=600+randi(212);
        limy=800+randi(212);
        x1=randi(x-1*limx);
        y1=randi(y-1*limy);
        l1=I1(x1:x1+limx,y1:y1+limy);
        l2=I2(x1:x1+limx,y1:y1+limy);
        %         l2 = imnoise(l2,'speckle',0.25);
        filename3=['1\',num2str(num+14000),'.jpg'];
        filename4=['2\',num2str(num+14000),'.png'];
        imwrite(l1,filename3);
        imwrite(l2,filename4);
    end
end














