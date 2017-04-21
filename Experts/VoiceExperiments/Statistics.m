clear
clc
close all
%Count sample for each identity

%from folders

DatasetPath='D:\ID_Clean_Noise_Splitted\ID_Clean_Noise\';
DataType='Noisy';
%%
% % % % %Code Section1:
% % % % %Read identities folders and count how many files in
% % % % identities= dir(strcat(DatasetPath));
% % % % CountSamples={};
% % % % for i=1:length(identities)
% % % %     if(identities(i).name(1)~='.'&& ~ strcmp(identities(i).name(end-3:end),'.mat'))
% % % %     Samples=dir(strcat(DatasetPath,identities(i).name,'\',DataType,'\*.wav'));
% % % %     CountSamples{end+1,1}=identities(i).name;
% % % %     CountSamples{end,2}=length(Samples);
% % % %     end
% % % %     
% % % % end
% % % % save(strcat(DatasetPath,'Statistics',DataType),'CountSamples');


%%
%Code Section2:
%minimum and maximum file duration
Durations={};
identities= dir(strcat(DatasetPath));

for i=1:length(identities)
    if(identities(i).name(1)~='.'&& ~ strcmp(identities(i).name(end-3:end),'.mat'))
        Samples=dir(strcat(DatasetPath,identities(i).name,'\',DataType,'\*.wav'));
        for j=1:length(Samples)
            info=audioinfo(strcat(strcat(DatasetPath,identities(i).name,'\',DataType,'\',Samples(j).name)));
            Durations{end+1,1}=identities(i).name;
            Durations{end,2}=Samples(j).name;
            Durations{end,3}=info.Duration; %Durations in seconds
        end
    end
end
minDur=min(cell2mat(Durations(:,3)));
maxDur=max(cell2mat(Durations(:,3)));
% save(strcat(DatasetPath,DataType,'_Durations'),'Durations','minDur','maxDur');





%%
%%Don't use
% % % %Code Section2:
% % % clear
% % % clc
% % % close all
% % % %How many samples are repeated on 2 sets
% % % 
% % % %EvaluationSet Paths
% % % Set2='D:\Collaborations\Biometrics fusion\My data\Voice Experiment\DataSet\sitw_database.v4.tar\sitw_database.v4\eval\keys\MetaData_Eval_All.xlsx';
% % % 
% % % %DevelopmentSet Paths
% % % Set2='D:\Collaborations\Biometrics fusion\My data\Voice Experiment\DataSet\sitw_database.v4.tar\sitw_database.v4\dev\keys\MetaData_Dev_All.xlsx';
% % % 
% % % %Read Set1
% % % [No,PersonID_Set1]=xlsread(DataSetMetaPath,'B:B');
% % % PersonID_Set1=PersonID_Set1(2:end);
% % % speakersno_Set1=xlsread(DataSetMetaPath,'H:H');
% % % 
% % % %Read Set2
% % % 
% % % [No,PersonID_Set2]=xlsread(DataSetMetaPath,'B:B');
% % % PersonID_Set2=PersonID_Set2(2:end);
% % % speakersno_Set2=xlsread(DataSetMetaPath,'H:H');
% % % 
% % % 
% % % %Compare
% % % CountSamples={};
% % % for i=1:length(PersonID_Set1)
% % %     CountSamples{i,2}=0;
% % %     if(speakersno_Set1(i)==1)
% % %         for j=1:length(PersonID_Set2)
% % %             if(strcmp(PersonID_Set1{i},PersonID_Set2{j}) && speakersno_Set2(j)==1)
% % %                 CountSamples{i,1}=PersonID_Set1(i);
% % %     CountSamples{i,2}=CountSamples{i,2}+1;
% % %             end
% % %         end
% % %     end
% % % end