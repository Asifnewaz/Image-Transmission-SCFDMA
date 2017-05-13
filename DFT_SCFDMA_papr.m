function [PSNR_DFT_ifdma PSNR_DFT_lfdma y1_ifdma_DFT y1_lfdma_DFT] = DFT_SCFDMA(SP)
%======= Choose simulation Parameters
SP.FFTsize = 512;
SP.inputBlockSize = 128;
SP.CPsize = 20;
%SP.subband = 15;
SP.subband = 0;%SP.numRun = 200;
SP.SNR =20;
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
% im=imread('lena.bmp');
im1=imread('mri.gif');%image reading
im=imresize(im1,[256 256]);
%%im = imresize(im,[256 256]); % Resize image
%im=imread('D:\my work\Chaotic Mapping data\programs\lena512.bmp');%im=imread('mri.tif');
%xx=randomization(im);%image randomization
%******************Data Generation********************
%f = zeros(256,256);
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
demodata1 = zeros(SP.inputBlockSize ,nloops);% this matrix to %store received data vector
clear jj
for jj =1: nloops % loop for columns
    b1= input_data2(:,jj)';%every block size
    %%%%%%%%%%%%%%% QPSK Modulation %%%%%%%%%%%%%%%%
    tmp = b1;
    tmp = tmp*2 - 1;
    inputSymbols = (tmp(1,:) + i*tmp(1,:))/sqrt(2);
    %%%%%%%%%%%% SC-FDMA  Modulation %%%%%%%%%%%%%
    inputSymbols_freq = fft(inputSymbols);
    inputSamples_ifdma = zeros(1,numSymbols);
    inputSamples_lfdma = zeros(1,numSymbols);
    %%%%%%%%%%%% Subcarriers Mapping %%%%%%%%%%%%%
    inputSamples_ifdma(1+SP.subband:Q:numSymbols) = inputSymbols_freq;
    inputSamples_lfdma([1:SP.inputBlockSize]+SP.inputBlockSize*SP.subband) = inputSymbols_freq;
    inputSamples_ifdma = ifft(inputSamples_ifdma);
    inputSamples_lfdma = ifft(inputSamples_lfdma);
    %%%%%%%%%%%%% Add Cyclic Prefix %%%%%%%%%%%%%
    TxSamples_ifdma = [inputSamples_ifdma(numSymbols-SP.CPsize+1:numSymbols) inputSamples_ifdma];
    TxSamples_lfdma = [inputSamples_lfdma(numSymbols-SP.CPsize+1:numSymbols) inputSamples_lfdma];
    %%%%%%%%%%%%% Wireless channel %%%%%%%%%%%%%%
    RxSamples_ifdma = filter(SP.channel, 1,TxSamples_ifdma); % Multipath Channel
    RxSamples_lfdma = filter(SP.channel, 1,TxSamples_lfdma); % Multipath Channel
    %%%%%%%%%%%%% Noise Generation %%%%%%%%%%%%%
    tmp = randn(2, numSymbols+SP.CPsize);
    complexNoise = (tmp(1,:) + i*tmp(2,:))/sqrt(2);
    noisePower = 10^(-SP.SNR(sss)/10);
    %%%%%%%%%%%%%%% Received signal%%%%%%%%%%%%%%
    RxSamples_ifdma = RxSamples_ifdma + sqrt(noisePower/Q)*complexNoise;
    RxSamples_lfdma = RxSamples_lfdma + sqrt(noisePower/Q)*complexNoise;
    %%%%%%%%%%%% Remove Cyclic Prefix%%%%%%%%%%%%
    RxSamples_ifdma = RxSamples_ifdma(SP.CPsize+1:numSymbols+SP.CPsize);
    RxSamples_lfdma = RxSamples_lfdma(SP.CPsize+1:numSymbols+SP.CPsize);
    %%%%%%%%%%%%% SC-FDMA demodulation%%%%%%%%%%%
    Y_ifdma = fft(RxSamples_ifdma, SP.FFTsize);
    Y_lfdma = fft(RxSamples_lfdma, SP.FFTsize);
    %%%%%%%%%%%% subcarriers demapping%%%%%%%%%%%
    Y_ifdma = Y_ifdma(1+SP.subband:Q:numSymbols);
    Y_lfdma = Y_lfdma([1:SP.inputBlockSize]+SP.inputBlockSize*SP.subband);
    %%%%%%%%%%%%%%%%% Equalization %%%%%%%%%%%%%%%%%
    H_eff = H_channel(1+SP.subband:Q:numSymbols);
    if SP.equalizerType == 'ZERO'
        Y_ifdma = Y_ifdma./H_eff;
    elseif SP.equalizerType == 'MMSE'
        C = conj(H_eff)./(conj(H_eff).*H_eff +10^(-SP.SNR(sss)/10));
        Y_ifdma = Y_ifdma.*C;
    end
    H_eff = H_channel([1:SP.inputBlockSize]+SP.inputBlockSize*SP.subband);
    if SP.equalizerType == 'ZERO'
        Y_lfdma = Y_lfdma./H_eff;
    elseif SP.equalizerType == 'MMSE'
        C = conj(H_eff)./(conj(H_eff).*H_eff +10^(-SP.SNR(sss)/10));
        Y_lfdma = Y_lfdma.*C;
    end
    EstSymbols_ifdma = ifft(Y_ifdma);
    EstSymbols_lfdma = ifft(Y_lfdma);
    %%%%%%%%%%%%%%%% demodulation%%%%%%%%%%%%%%%%
    EstSymbols_ifdma = sign(real(EstSymbols_ifdma)) ;
    EstSymbols_ifdma =(EstSymbols_ifdma+1)/2;
    EstSymbols_lfdma = sign(real(EstSymbols_lfdma)) ;
    EstSymbols_lfdma = (EstSymbols_lfdma+1)/2;
    demodata1_ifdma(:,jj) =  EstSymbols_ifdma(:);   % the output of scfdma columns%storing of received image data
    demodata1_lfdma(:,jj)  = EstSymbols_lfdma(:);
    % the output of scfdma columns%storing of received image data
end
%*****************  Received image  ***************
[M3,N3] = size(demodata1_ifdma);
%  demodata2 = demodata1(:);
yy1_ifdma = reshape (demodata1_ifdma,M3,N3);
%reshaping the matrix to vector
yy1_lfdma = reshape (demodata1_lfdma,M3,N3);
%reshaping the matrix to vector
received_image_ifdma = yy1_ifdma(1:M2*N2);%taking the original data
received_image_lfdma = yy1_lfdma(1:M2*N2);%taking the original data
%**************  Regeneration of image  **************
zz1_ifdma=reshape(received_image_ifdma,M2* N2,1);
%reshaping to M2*N2 vector
zz1_lfdma=reshape(received_image_lfdma,M2* N2,1);
%reshaping to M2*N2 vector
yy_ifdma = reshape(zz1_ifdma,M2, N2);
yy_lfdma = reshape( zz1_lfdma,M2, N2);
clear i j
for i=1:M1
    for j=1:N1
        zn_ifdma(i,j)=num2str(yy_ifdma(i,j));
        zn_lfdma(i,j)=num2str(yy_lfdma(i,j));
    end;
end;
hn_ifdma=bin2dec(zn_ifdma);
hn_lfdma=bin2dec(zn_lfdma);
gn_ifdma=col2im(hn_ifdma, [M,N], [M,N], 'distinct');
gn_lfdma=col2im(hn_lfdma, [M,N], [M,N], 'distinct');
%y1_ifdma=derandomization(gn_ifdma);
%y1_lfdma=derandomization(gn_lfdma);
y1_ifdma=gn_ifdma/255;
y1_lfdma=gn_lfdma/255;
% ***************** The output results****************
papr_DFT_SC_iFDMA = 10*log10(max(abs(EstSymbols_ifdma).^2)/mean(abs(EstSymbols_ifdma).^2));
papr_DFT_SC_lFDMA = 10*log10(max(abs(EstSymbols_lfdma).^2)/mean(abs(EstSymbols_lfdma).^2));
end
[N,X] = hist(papr_DFT_SC_iFDMA,1000);
semilogy(X,1-cumsum(N)/max(cumsum(N)),'b-')
axis([0 20 1e-4 1])
xlabel('PAPR(dB)'); ylabel('CCDF');