function [ MFCCFeat ] = ExtractMFCC(audiopath)

%This function extracts MFCC features from audio file
%Fs is the sampling or frame rate 
%We use 16k
[audio,Fs] = audioread(audiopath);
MFCCFeat = melcepst(audio, FS);

end

