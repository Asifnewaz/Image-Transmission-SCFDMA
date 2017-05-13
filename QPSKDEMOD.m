function temp_est = QPSKDEMOD(modulated_data1_conv)
scalin_fact = sqrt(1/2);
demod_symbol1_conv = modulated_data1_conv/scalin_fact;
%4QAM demodulation
demodulated_symbol1_conv = qamdemod(demod_symbol1_conv,4);
symbol_size = 2;
for y_conv = 1:size(demodulated_symbol1_conv)
demodulated_bit_conv = de2bi(demodulated_symbol1_conv(y_conv,:),symbol_size,'left-msb')';
demodulated_bit1_conv(y_conv,:) = demodulated_bit_conv(:);
end
temp_est = demodulated_bit1_conv;
