clc
clear 
close all
%% Parameters
samplingFrequency = 16000;
bitResolution = 16;
recordTime = 5;
channel = 1;
samplingFactor = 4;
M=4;
Thorivos=4;
% % u-law parameters
 bits = 8;
 Mu = 2^bits-1;
% %%??????? files 
 beforeTransmissionFile = 'soundfile1_lab2.wav';
 receviedSoundFilename = 'C:\Users\ellin\Desktop\Telecommunications\receivedVoicetest14.wav';

[soundData,recordedSound] = audioread('soundfile1_lab2.wav');
%% ????? ?? ??????
sound(soundData,samplingFrequency)
pause(5)
%% Downsample
downSampledSignal = downsample(soundData,samplingFactor);
%% Compand and Quantization
maximumValueofSignal = max(downSampledSignal);
compandedSignal = compand(downSampledSignal,Mu,maximumValueofSignal,'mu/compressor');
maxofCompandedSignal = max(soundData);
minofCompandedSignal = min(soundData);
stepSize = (maxofCompandedSignal - minofCompandedSignal)/Mu;
partition =[minofCompandedSignal:stepSize:maxofCompandedSignal];
codeBook = [(minofCompandedSignal-stepSize):stepSize:maxofCompandedSignal];
% quantized signal  
[index,quants] = quantiz(compandedSignal,partition,codeBook); 
%% SCoding
sourceCode = [];
sourceCode = [sourceCode,dec2bin(index,8)];
souceCodeTranspose = sourceCode';
sourceCodeStream = reshape(souceCodeTranspose,1,[]);
%% ?????????? QPSK
% Convert into 2 bit column stream
symbolValue = [];
counter = 0;
for bit  = 1:length(sourceCodeStream)/2
    symbolValue = [symbolValue;sourceCodeStream(bit+counter:bit+counter+1)];
    counter = counter +1;
end
% ????????? ????? ??? ???? ???????
decimalSymbolValue = bin2dec(symbolValue);
% ???????????? ????
modulatedSignal  = pskmod(decimalSymbolValue,M);
%% ??????
%% ???????(??????) 
%SNR in db
EbNodBVals = Thorivos;
EbNo=10.^(EbNodBVals./10);
EsNo=EbNo.*1.*log2(M);
% ???????? ???????
Es = sum(abs(modulatedSignal).^2)/(length(modulatedSignal));
No=Es./EsNo;
sigma=sqrt(No./2);
% AWGN Noise 
noise=sigma.*(randn(size(modulatedSignal))+1j.*randn(size(modulatedSignal)));
signalWithNoise = modulatedSignal + noise;
outputSignal = rectpulse(signalWithNoise,8);
%% ??????
signalReceivedProcessed = intdump(outputSignal,8);
    %% ????????????? QPSK
demodulatedSignal = pskdemod(signalReceivedProcessed,M);
[errors,ratio11]=biterr(decimalSymbolValue,demodulatedSignal);
% convert the symbols into binary ans reshape to one bit stream
receivedSignalinBin = dec2bin(demodulatedSignal);
receivedSignalStream = reshape(receivedSignalinBin',1,[]);
% convert them to 8 bit each to create the original symbol size
symbol8Bits = [];
counter = 0;
for bit = 1:length(receivedSignalStream)/8
    symbol8Bits = [symbol8Bits;receivedSignalStream(bit+counter:bit+counter+7)];
    counter = counter + 7;
end
decimalValuesReceived = bin2dec(symbol8Bits);
size(decimalValuesReceived);
size(index);
quantizedError = decimalValuesReceived-index;
find(quantizedError ~= 0);
recevieSideQuants = codeBook(decimalValuesReceived+1);
%% ???????? ?? ????
maxofReceivedQuants = max(recevieSideQuants);
expandedSignal = compand(recevieSideQuants,Mu,maxofReceivedQuants,'mu/expander');
%% Resample 16 Khz 
receivedResampledSignal = upsample(expandedSignal,4);
%% ????? ?? the audio files
audiowrite('C:\Users\ellin\Desktop\Telecommunications\receivedVoicetest4.wav',expandedSignal,4000);
%audiowrite(recoveredSoundFilename,filteredSignal,samplingFrequency);
%% Play the signal
sound(expandedSignal,4000);
pause(5)
%% Plotting all the signals 
% plot all the signal in time and frequency domain
%% Original Voice Signal
%Plot Signal in Time-Domain 
figure();
plot(soundData, 'b');
title('Orginal Voice')
xlabel('time (seconds)')

%% Received Signal
figure();
plot(expandedSignal, 'b');
title('Received Signal')
xlabel('time (seconds)')
