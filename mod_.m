function inputSymbols = mod_(encod_data1_conv,ModulationType)
switch (ModulationType)
    case 'QPSK'
        symbol_size = 2;
        scaling_fact = sqrt(1/2);
        for e_conv = 1:symbol_size:size(encod_data1_conv,2)
            mod_inp_conv(:,floor(e_conv/symbol_size)+1) = bi2de(encod_data1_conv(:,e_conv:e_conv+symbol_size-1),'left-msb');
        end
        constell_qpsk_conv = qammod(mod_inp_conv,4);
        %QPSK is implemented as 4 QAM
        %scaled modulated data
        inputSymbols = scaling_fact*constell_qpsk_conv;
    case '16QAM'
        symbol_size = 4;
        scaling_fact = sqrt(1/10);
        %convert the symbol into [0 . . . M1]
        for p_conv = 1:symbol_size:size(encod_data1_conv,2)
            mod_inp_conv(:,floor(p_conv/symbol_size)+1) = bi2de(encod_data1_conv(:,p_conv:p_conv+symbol_size-1),'left-msb');
        end
        constell_16qpsk_conv = qammod(mod_inp_conv,16);
        inputSymbols = scaling_fact*constell_16qpsk_conv;
    case '64QAM'
        symbol_size1 = 6;
        scaling_fact = sqrt(1/42);
        for p_conv = 1:symbol_size1:size(encod_data1_conv,2)
            mod_inp_conv(:,floor(p_conv/symbol_size1)+1) = bi2de(encod_data1_conv(:,p_conv:p_conv+symbol_size1-1),'left-msb');
        end
        constell_64qam_conv = qammod(mod_inp_conv,64);
        inputSymbols = scaling_fact*constell_64qam_conv;
end
