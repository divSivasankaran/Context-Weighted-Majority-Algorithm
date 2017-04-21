%Read Flac files
%Filter Noisy and clean
%store.wav files
clear
clc
close all
% % % % %EvaluationSet Paths
% % % % % datasetPath='D:\Collaborations\Biometrics fusion\My data\Voice Experiment\DataSet\sitw_database.v4.tar\sitw_database.v4\eval\';
% % % % % NewdataPath='D:\Collaborations\Biometrics fusion\My data\Voice Experiment\DataSet\sitw_database.v4.tar\OurVoiceDataSet\';
% % % % % DataSetMetaPath='D:\Collaborations\Biometrics fusion\My data\Voice Experiment\DataSet\sitw_database.v4.tar\sitw_database.v4\eval\keys\MetaData_Eval_All.xlsx';
% % % % 
% % % % %DevelopmentSet Paths
% % % % datasetPath='D:\Collaborations\Biometrics fusion\My data\Voice Experiment\DataSet\sitw_database.v4.tar\sitw_database.v4\dev\';
% % % % NewdataPath='D:\Collaborations\Biometrics fusion\My data\Voice Experiment\DataSet\sitw_database.v4.tar\OurVoiceDataSet\';
% % % % DataSetMetaPath='D:\Collaborations\Biometrics fusion\My data\Voice Experiment\DataSet\sitw_database.v4.tar\sitw_database.v4\dev\keys\MetaData_Dev_All.xlsx';
% % % % 
% % % % %Creating the dataset for clean and noise audio with only one speaker
% % % % [No,AudioName]=xlsread(DataSetMetaPath,'A:A');
% % % % AudioName=AudioName(2:end);
% % % % [No,PersonID]=xlsread(DataSetMetaPath,'B:B');
% % % % PersonID=PersonID(2:end);
% % % % Noise=xlsread(DataSetMetaPath,'J:J');
% % % % speakersno=xlsread(DataSetMetaPath,'H:H');
% % % % count=0;
% % % % for i=1:length(AudioName)
% % % %     if(speakersno(i)==1)
% % % %         %Read
% % % %         [audio,Fs] = audioread(strcat(datasetPath,AudioName{i}));
% % % %         videoname= strsplit(AudioName{i},'/');
% % % %         videoname= strsplit(videoname{2},'.');
% % % %         videoname= videoname{1};
% % % %         %write
% % % %         if(Noise(i)==0)
% % % %             % sound(data,Fs);
% % % %             %Create folder for identity
% % % %             if(~exist(strcat(NewdataPath,'Clean\',PersonID{i})))
% % % %                 mkdir(strcat(NewdataPath,'Clean\',PersonID{i}));
% % % %             end
% % % %             audiowrite(strcat(NewdataPath,'Clean\',PersonID{i},'\',videoname,'.wav'),audio,Fs);
% % % %         else
% % % %             if(~exist(strcat(NewdataPath,'Noisy\',PersonID{i})))
% % % %                 mkdir(strcat(NewdataPath,'Noisy\',PersonID{i}));
% % % %             end
% % % %             audiowrite(strcat(NewdataPath,'Noisy\',PersonID{i},'\',videoname,'.wav'),audio,Fs);
% % % %         end
% % % %         % features = melcepst(data, Fs);
% % % %         count=count+1;
% % % %     end
% % % %     
% % % % end

%%Filter identities that have noise and clean files
datasetPath='D:\Collaborations\Biometrics fusion\My data\Voice Experiment\DataSet\sitw_database.v4.tar\OurVoiceDataSet\';
FilteredDataSet='D:\Collaborations\Biometrics fusion\My data\Voice Experiment\DataSet\sitw_database.v4.tar\OurVoiceDataSet\ID_Clean_Noise\';
Cleanfolders=dir(strcat(datasetPath,'Clean'));
Noisyfolders=dir(strcat(datasetPath,'Noisy'));

for i=1:length(Cleanfolders)
   for j=1:length(Noisyfolders)
    if(strcmp(Cleanfolders(i).name,Noisyfolders(j).name) && Cleanfolders(i).name(1)~='.'&& Noisyfolders(j).name(1)~='.')
        %Create folder for identity
            if(~exist(strcat(FilteredDataSet,Noisyfolders(j).name)))
                mkdir(strcat(FilteredDataSet,Noisyfolders(j).name));
            end
            copyfile(strcat(datasetPath,'Noisy\',Noisyfolders(j).name),strcat(FilteredDataSet,Noisyfolders(j).name,'\Noisy'));
            copyfile(strcat(datasetPath,'Clean\',Noisyfolders(j).name),strcat(FilteredDataSet,Noisyfolders(j).name,'\Clean'));
            %
    end
end
end
