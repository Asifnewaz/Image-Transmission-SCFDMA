function [PSNR_ofdma1 y1_ofdma1] = DFT_OFDMA_wo(SP)

for sss=1: length(SP.SNR)
%======= Choose channel type
%======= Uniform Channel
%SP.channel=(1/sqrt(10))*(randn(10,SP.numRun)+sqrt(-1)*randn(10,SP.numRun))/sqrt(2);
%======= SUI3 Channel
SP.paths= fadchan(SP);
%======= Vehicular A Channel
% x(1)=1;
% x(2)=10^(-1/10);
% x(3)=10^(-9/10);
% x(4)=10^(-10/10);
% x(5)=10^(-15/10);
% x(6)=10^(-20/10);
% tuamp=sqrt(x);
% tp=tuamp/norm(tuamp);
%
% ch1=jake(120,6);
%
%
% SP.paths(1,:)=tp(1)*ch1(1,:);
% SP.paths(2,:)=tp(2)*ch1(2,:);
% SP.paths(3,:)=tp(3)*ch1(3,:);
% SP.paths(4,:)=tp(4)*ch1(4,:);
% SP.paths(5,:)=tp(5)*ch1(5,:);
% SP.paths(6,:)=tp(6)*ch1(6,:);
%=====%======= Choose Equalization Type
% SP.equalizerType ='ZERO';
SP.equalizerType ='MMSE';
%%%%%%%==================================
numSymbols = SP.FFTsize;
Q = numSymbols/SP.inputBlockSize;
%%%%%%%%%========== Channel Generation============
%%%%%%%%%===========Uniform channel============
% SP.channel=SP.channel1(:,k).';
%%%%%%%%%======= Vehicular A Channel ============
% vechA_=[SP.paths(1,k) SP.paths(2,k) 0 SP.paths(3,k) SP.paths(4,k) 0 0 SP.paths(5,k) 0 0 SP.paths(6,k)];
% SP.channel = vechA_;
%
%%%%%%%%%============ SUI3 channel============
SUI3_1=[SP.paths(1,1) 0 0 SP.paths(2,1) 0 SP.paths(3,1)];
SP.channel = SUI3_1/norm(SUI3_1);
%%%%%%%%%================= AWGN channel============
% SP.channel=1;
%%%%%%%%%========================================
H_channel = fft(SP.channel,SP.FFTsize);

im1=imread('mri.gif');%image reading
im=imresize(im1,[256 256]);

%xx=randomization(im);%image randomization
%******************Data Generation********************
f = zeros(256,256);
f = double(im);
[M,N]=size(f);
g=im2col(f, [M,N],'distinct');%image to column converter
h=dec2bin(double(g));%pixel value to binary conversion...every value replaced by 8 bits string
[M1,N1] = size(h);
z=zeros (M1,N1);
clear i j
for i=1:M1
    for j=1:N1
        z(i,j)= str2num(h(i,j)); %string to number conversion
    end;
end;
[M2,N2] = size(z) ;
zz = reshape(z,M2*N2, 1);%parallel data reshaping date to vector
% ********** Dividing the image into blocks***********
nloops = ceil((M2*N2)/SP.inputBlockSize );%number of image %blocks by approximation
new_data = nloops*SP.inputBlockSize ;%new vector proportional to %block size
nzeros = new_data  - (M2*N2);%number of zeros to be added to %old data vector
input_data = [zz;zeros(nzeros,1)];%construction of new data %vector
input_data2 = reshape(input_data ,SP.inputBlockSize ,nloops); %reshape the new data to matrix of block size rows to number of %blocks columns
save input_data2
%************** transmission ON SC-FDMA ***************
demodata1_ofdma = zeros(SP.inputBlockSize ,nloops);% this matrix to %store received data vector
clear jj
for jj =1: nloops % loop for columns
    b1= input_data2(:,jj)';%every block size
    %%%%%%%%%%%%%%% QPSK Modulation %%%%%%%%%%%%%%%%
    tmp = b1;
    tmp = tmp*2 - 1;
    inputSymbols = (tmp(1,:) + i*tmp(1,:))/sqrt(2);
    %%%%%%%%%%%% SC-FDMA  Modulation %%%%%%%%%%%%%
    inputSamples_ofdma = zeros(1,numSymbols);
    %%%%%%%%%%%% Subcarriers Mapping %%%%%%%%%%%%%
    inputSamples_ofdma(1+SP.subband:Q:numSymbols) = inputSymbols;
    inputSamples_ofdma = ifft(inputSamples_ofdma);
    %%%%%%%%%%%%% Add Cyclic Prefix %%%%%%%%%%%%%
    TxSamples_ofdma = [inputSamples_ofdma(numSymbols-SP.CPsize+1:numSymbols) inputSamples_ofdma];
    %%%%%%%%%%%%% Wireless channel %%%%%%%%%%%%%%
    RxSamples_ofdma = filter(SP.channel, 1,TxSamples_ofdma); % Multipath Channel
    %%%%%%%%%%%%% Noise Generation %%%%%%%%%%%%%
    tmp = randn(2, numSymbols+SP.CPsize);
    complexNoise = (tmp(1,:) + i*tmp(2,:))/sqrt(2);
    noisePower = 10^(-SP.SNR(sss)/10);
    %%%%%%%%%%%%%%% Received signal%%%%%%%%%%%%%%
    RxSamples_ofdma = RxSamples_ofdma + sqrt(noisePower/Q)*complexNoise;
    %%%%%%%%%%%% Remove Cyclic Prefix%%%%%%%%%%%%
    RxSamples_ofdma = RxSamples_ofdma(SP.CPsize+1:numSymbols+SP.CPsize);
    %%%%%%%%%%%%% SC-FDMA demodulation%%%%%%%%%%%
    Y_ofdma = fft(RxSamples_ofdma, SP.FFTsize);
    %%%%%%%%%%%% subcarriers demapping%%%%%%%%%%%
    Y_ofdma = Y_ofdma(1+SP.subband:Q:numSymbols);
    %%%%%%%%%%%%%%%%% Equalization %%%%%%%%%%%%%%%%%
    H_eff = H_channel(1+SP.subband:Q:numSymbols);
    if SP.equalizerType == 'ZERO'
        Y_ofdma = Y_ofdma./H_eff;
    elseif SP.equalizerType == 'MMSE'
        C = conj(H_eff)./(conj(H_eff).*H_eff +10^(-SP.SNR(sss)/10));
        Y_ofdma = Y_ofdma.*C;
    end
    %%%%%%%%%%%%%%%% demodulation%%%%%%%%%%%%%%%%
    EstSymbols_ofdma = sign(real(Y_ofdma)) ;
    EstSymbols_ofdma =(EstSymbols_ofdma+1)/2;
    demodata1_ofdma(:,jj) =  EstSymbols_ofdma(:);   % the output of scfdma columns%storing of received image data
    % the output of scfdma columns%storing of received image data
end
%*****************  Received image  ***************
[M3,N3] = size(demodata1_ofdma);
%  demodata2 = demodata1(:);
yy1_ifdma = reshape (demodata1_ofdma,M3,N3);
%reshaping the matrix to vector
received_image_ofdma = yy1_ifdma(1:M2*N2);%taking the original data
%**************  Regeneration of image  **************
zz1_ofdma=reshape(received_image_ofdma,M2* N2,1);
%reshaping to M2*N2 vector
yy_ofdma = reshape(zz1_ofdma,M2, N2);
clear i j
for i=1:M1
    for j=1:N1
        zn_ofdma(i,j)=num2str(yy_ofdma(i,j));
    end;
end;
hn_ofdma=bin2dec(zn_ofdma);
gn_ofdma=col2im(hn_ofdma, [M,N], [M,N], 'distinct');
%y1_ofdma=derandomization(gn_ofdma);

y1_ofdma=gn_ofdma/255;
% ***************** The output results****************
figure (311)
imshow(im)
figure (312)
imshow(y1_ofdma)
MSE1_ofdma=sum(sum((double(im)/255-y1_ofdma).^2))/prod(size(im));
PSNR_ofdma1(sss)=10*log(1/MSE1_ofdma)/log(10);
y1_ofdma1(:,:,sss)=y1_ofdma;
toc
end