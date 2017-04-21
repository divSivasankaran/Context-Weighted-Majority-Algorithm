%Filter less than 3 second, choose first 64 that has more samples
% % % clear
% % % clc
% % % %Load data
% % % DatasetPath='D:\ID_Clean_Noise_Splitted\ID_Clean_Noise\';
% % % ClDur=load(strcat(DatasetPath,'Clean_Durations'));
% % % NoiDur=load(strcat(DatasetPath,'Noisy_Durations'));
% % % IdentitiesNoChosen=64;
% % % 
% % % %Choose samples with durations==3
% % % ind=1;
% % % for i=1:length(ClDur.Durations)
% % %     if(ClDur.Durations{i,3}>=3)
% % %         CleanDur(ind,:)=ClDur.Durations(i,:);
% % %         ind=ind+1;
% % %     end
% % % end
% % % ind=1;
% % % for i=1:length(NoiDur.Durations)
% % %     if(NoiDur.Durations{i,3}>=3)
% % %         NoisyDur(ind,:)=NoiDur.Durations(i,:);
% % %         ind=ind+1;
% % %     end
% % % end
% % % save(strcat(DatasetPath,'FilteredDurations.mat'),'CleanDur','NoisyDur');

%%
% % %Find 64 with longest durations
% % filteredIds={};
% % Id=dir(DatasetPath);
% % for i=1:length(Id)
% %     if(Id(i).name(1)~='.'&& ~ strcmp(Id(i).name(end-3:end),'.mat'))
% %         cleancount=0;
% %         Noisycount=0;
% %         for j=1:length(CleanDur)
% %             if(strcmp(Id(i).name,CleanDur{j,1}))
% %                 cleancount=cleancount+1;
% %             end
% %         end
% %         for j=1:length(NoisyDur)
% %             if(strcmp(Id(i).name,NoisyDur{j,1}))
% %                 Noisycount=Noisycount+1;
% %             end
% %         end
% %     
% %         filteredIds{end+1,1}=Id(i).name;
% %         filteredIds{end,2}=min(cleancount,Noisycount);
% %         
% %         
% %     end
% %     
% % end
% % 
% FIds=cell2mat(filteredIds(:,2));
% [Sorted,index]=sort(FIds,'descend');
% SortedIds=filteredIds(index,:);

%Build the dataset
for k=1:IdentitiesNoChosen
    IDname=SortedIds{k,1};
    Newdataset='D:\Collaborations\Biometrics fusion\My data\Voice Experiment\DataSet\sitw_database.v4.tar\OurVoiceDataSet\ID_Clean_Noise_Splitted_Filtered\';
    %Create identity folder
    %Create Clean and noisy folders
    if(~exist(strcat(Newdataset,IDname,'\Clean\')));
        mkdir(strcat(Newdataset,IDname,'\Clean\'));
        mkdir(strcat(Newdataset,IDname,'\Noisy\'));
    end
    %get audio files from clean
    for i=1:length(CleanDur)
        if(strcmp(IDname,CleanDur{i,1}))
            copyfile(strcat(DatasetPath,IDname,'\Clean\',CleanDur{i,2}),strcat(Newdataset,IDname,'\Clean\'));
        end
    end
        
        %get audio files from noisy
        for i=1:length(NoisyDur)
            if(strcmp(IDname,NoisyDur{i,1}))
                copyfile(strcat(DatasetPath,IDname,'\Noisy\',NoisyDur{i,2}),strcat(Newdataset,IDname,'\Noisy\'));
            end
        end
        
   
    end
