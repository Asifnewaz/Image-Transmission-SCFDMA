function runSimS_DFT_and_DCT_SC_FDMA()
clear all
tic;
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
%%%%%%%%% Choose the Equalization Type%%
% SP.equalizerType = 'ZERO';
SP.equalizerType = 'MMSE';


%%%%%%%%% Run the simulation for With Randomization PSNR_DFT_SCFDMA %%
[PSNR_DFT_ifdma PSNR_DFT_lfdma y1_ifdma_DFT y1_lfdma_DFT] = DFT_SCFDMA(SP);
%%%%%%%%% Run the simulation for DCT_SCFDMA %%
[PSNR_DCT_ifdma PSNR_DCT_lfdma y1_ifdma_DCT y1_lfdma_DCT] = DCT_SCFDMA(SP);
%%%%%%%%% Run the simulation for PSNR_OFDMA %%
[PSNR_ofdma y1_ofdma] = DFT_OFDMA(SP);
save PSNR_DCT_ifdma; 
save PSNR_DCT_lfdma;
save PSNR_DFT_ifdma; 
save PSNR_DFT_lfdma;
save PSNR_ofdma;
save y1_ifdma_DCT; 
save y1_lfdma_DCT;
save y1_ifdma_DFT; 
save y1_lfdma_DFT;
save y1_ofdma;
%%%%%%%%% Run the simulation for Without Randomization PSNR_DFT_SCFDMA %%
[PSNR_DFT_ifdma1 PSNR_DFT_lfdma1 y1_ifdma_DFT1 y1_lfdma_DFT1] = DFT_SCFDMA_wo(SP);
%%%%%%%%% Run the simulation for DCT_SCFDMA %%
[PSNR_DCT_ifdma1 PSNR_DCT_lfdma1 y1_ifdma_DCT1 y1_lfdma_DCT1] = DCT_SCFDMA_wo(SP);
%%%%%%%%% Run the simulation for PSNR_OFDMA %%
[PSNR_ofdma1 y1_ofdma1] = DFT_OFDMA_wo(SP);
save PSNR_DCT_ifdma1; 
save PSNR_DCT_lfdma1;
save PSNR_DFT_ifdma1; 
save PSNR_DFT_lfdma1;
save PSNR_ofdma1;
save y1_ifdma_DCT1; 
save y1_lfdma_DCT1;
save y1_ifdma_DFT1; 
save y1_lfdma_DFT1;
save y1_ofdma1;

%%%%%%%%% Run the simulation for ZERO Equalizer PSNR_DFT_SCFDMA %%
[PSNR_DFT_ifdma_Z PSNR_DFT_lfdma_Z y1_ifdma_DFT_Z y1_lfdma_DFT_Z] = DFT_SCFDMA_ZERO(SP);
%%%%%%%%% Run the simulation for DCT_SCFDMA %%
[PSNR_DCT_ifdma_Z PSNR_DCT_lfdma_Z y1_ifdma_DCT_Z y1_lfdma_DCT_Z] = DCT_SCFDMA_MyZ(SP);
%%%%%%%%% Run the simulation for PSNR_OFDMA %%
[PSNR_ofdma_Z y1_ofdma_Z] = DFT_OFDMA_ZERO(SP);
save PSNR_DCT_ifdma_Z; 
save PSNR_DCT_lfdma_Z;
save PSNR_DFT_ifdma_Z; 
save PSNR_DFT_lfdma_Z;
save PSNR_ofdma_Z;
save y1_ifdma_DCT_Z; 
save y1_lfdma_DCT_Z;
save y1_ifdma_DFT_Z; 
save y1_lfdma_DFT_Z;
save y1_ofdma_Z;


%%%%%%%%% Plot the Results %%
figure(44)
plot(SP.SNR,PSNR_DFT_ifdma,'rx-',SP.SNR,PSNR_DFT_lfdma,'mx-');
hold on
plot(SP.SNR,PSNR_DCT_ifdma,'bx-',SP.SNR,PSNR_DCT_lfdma,'gx-');
hold on
plot(SP.SNR,PSNR_ofdma,'yx-');
legend('DFT-IFDMA','DFT-LFDMA','DCT-IFDMA','DCT-LFDMA','OFDMA')
xlabel('SNR (dB)'); ylabel('PSNR(dB)');
axis([0 60 0 90])
grid on

%without randomization
figure(45)
plot(SP.SNR,PSNR_DFT_ifdma1,'rx-',SP.SNR,PSNR_DFT_lfdma1,'mx-');
hold on
plot(SP.SNR,PSNR_DCT_ifdma1,'bx-',SP.SNR,PSNR_DCT_lfdma1,'gx-');
hold on
plot(SP.SNR,PSNR_ofdma1,'yx-');
legend('DFT-IFDMA','DFT-LFDMA','DCT-IFDMA','DCT-LFDMA','OFDMA')
xlabel('SNR (dB)'); ylabel('PSNR(dB)');
axis([0 60 0 90])
grid on

%ZERO
figure(46)
plot(SP.SNR,PSNR_DFT_ifdma_Z,'rx-',SP.SNR,PSNR_DFT_lfdma_Z,'mx-');
hold on
plot(SP.SNR,PSNR_DCT_ifdma_Z,'bx-',SP.SNR,PSNR_DCT_lfdma_Z,'gx-');
hold on
plot(SP.SNR,PSNR_ofdma_Z,'yx-');
legend('DFT-IFDMA','DFT-LFDMA','DCT-IFDMA','DCT-LFDMA','OFDMA')
xlabel('SNR (dB)'); ylabel('PSNR(dB)');
axis([0 60 0 90])
grid on
toc
