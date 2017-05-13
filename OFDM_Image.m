%*****************************************************
%*****************************************************
% Simulation program to realize OFDM image transmission
%*****************************************************
%******************preparation Part*******************
clc
clear all
para=256; % number of parallel channel to transmit
fftlen=256; %FFT length
noc=256; %number of carrier
nd=6; %number of information OFDM symbol for one loop
m1=2; %Modulation level:QPSK
sr=250000; %symbol rate
br=sr.*m1; %Bit rat per carrier
gilen=32; %Length of guard interval (points)
ebno=6; %Eb/No
Ipoint = 8; %Number of over samples
ofdm_length = para*nd*m1; %Total no for one loop
modh=1/(2*pi);
A=1;
%************************Transmitter******************
%***********************Data Generation***************
% im=imread('lena.bmp');
%im=imread('C:\Users\E. S. Hassan\Desktop\ima.gif');
%im=imread('C:\Documents and Settings\emad\Desktop\29-10\programs\lena512.bmp');
im=imread('mri.gif');
f=double(im);
[M,N]=size(f);
g=im2col(f, [M,N], [M,N], 'distinct');
h=dec2bin(double(g));
[M1,N1]=size(h) ;
z=zeros (M1,N1) ;
for i=1:M1
    for j=1:N1
        z(i,j)= str2num(h(i,j));
    end;
end;
[M2,N2] = size(z) ;
zz = reshape(z,M2*N2, 1); %parallel data
% *********** Dividing the image into blocks**********
nloops = ceil((M2*N2)/ofdm_length );
new_data = nloops*ofdm_length ;
nzeros   = new_data  - (M2*N2);
input_data = [zz;zeros(nzeros,1)];
input_data2 = reshape(input_data ,ofdm_length ,nloops);
%************** transmission ON OFDM *****************
demodata1 = zeros(ofdm_length ,nloops);
for jj = 1: nloops % loop for columns
    serdata1 = input_data2(:,jj)';
    
%*****************  Received image  ***************
[M3,N3] = size(demodata1);
%  demodata2 = demodata1(:);
yy1 = reshape (demodata1,M3*N3,1); %
received_image = yy1(1:M2*N2);
%**************  Regeneration of image  ***************
yy = reshape(received_image  ,M2, N2);
zn=zeros (M1,N1) ;
for i=1:M1
    for j=1:N1
        zn(i,j)=num2str(yy(i,j));
    end;
end;
hn=bin2dec(zn);
gn=col2im(hn, [M,N], [M,N], 'distinct');
gn=gn/255;
% ***************** The output results****************
%imwrite(gn,'image.tif', 'tif');
figure (1)
imshow(im)
figure (2)
imshow(gn)
%************   The Error between Trans  *************
MSE1=sum(sum((double(f)/255-gn).^2))/prod(size(f));
PSNR=10*log(1/MSE1)/log(10);