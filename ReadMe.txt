%======= Choose simulation Parameters 
SP.inputBlockSize = 128; 
SP.FFTsize = 512;% Set the values of the FFT and IFFT 
SP.CPsize = 20;%Set the Cyclic prefix length 
SP.subband = 0; 
SP.SNR =0:5:20;%Choose the range of the SNR in dB 
%%%% Choose the coding rate and the modulation Type%%% 
SP.cod_rate = '1/2'; 
%SP.cod_rate = '1'; 
%SP.modtype = '16QAM'; 
SP.modtype = 'QPSK'; 
%%%%%%%%% Choose the Equalization Type%% % 
SP.equalizerType = 'ZERO'; 
SP.equalizerType = 'MMSE';
