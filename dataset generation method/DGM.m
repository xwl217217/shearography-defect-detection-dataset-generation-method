% ��ȡ��������ʱ�̵�ͼƬ��������λ
% ʱ���˲���С���ε���
%% ��ͬʱ�̵���λ�ų���λ���飬����ά��ʱ����
% ͼƬ image0-image119,120��
clear
clc
tic
I1=imread('11.bmp');
I=I1;
T=fftshift(fft2(I));
imshow((1+abs(T)),[0.1e4,2e4])
[x,y] = ginput(2); % x �У�y ��
x=round(x);y=round(y);
mask=zeros(size(I1));
mask(y(1):y(2),x(1):x(2))=1;
P1=newpolar(11,mask);
P=zeros(2048,2448,7);
for i=11:17
    i
    P2=newpolar(i,mask);
    P(:,:,i-10)=P2.*conj(P1);
    %     A=angle(P2.*conj(P1));
    %     ir=im2uint8(mat2gray(A));
    %     filename=[num2str(i),'.bmp'];
    %     imwrite(ir,filename);
end


%% label��ע
imageLabeler
% imshow(logical(imread(string(gTruth.LabelData.a(1,1)))));
%% �����������
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
        %         dx=20+randi(200); %y fangxiang qianqie
        %         dy=20+randi(200); %x fangxiang jianqie
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



%% �ü�
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







%%
I1=imread("179.jpg");
I2=imread("fai12.bmp");
I3=imread("fai12.bmp");
i1=I1(809:1461,261:989);
i2=I2(809:1461,261:989);
i3=I3(41:701,1017:1685);
imwrite(i1,'1.jpg');
imwrite(i2,'4.jpg');
imwrite(i3,'5.jpg');








