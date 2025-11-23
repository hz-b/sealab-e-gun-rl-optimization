% Created by the authors of:
% Status Report of the Berlin Energy Recovery Linac Project BERLinPro, M. Abo-Bakr, DOI: 10.18429/JACoW-IPAC2016-TUPOW034
% SRF Gun Development for Energy Recovery Linac Applications, T. Kamps, DOI: 10.1364/EUVXRAY.2016.ET1A.3
% Setup and Status of an SRF Photoinjector for Energy-Recovery Linac Applications, T. Kamps, DOI: 10.18429/JACoW-IPAC2017-MOPVA010
% Development of a compact test facility for SRF Photoelectron injectors, J. Völker DOI: 10.18452/19322

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                  Definition der Zielfunktion
%
%                 zentrales Element der optimierung; hier läuft Astra!!!
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [varargout] = TrackingWrite(varargin)                                   
% z ist Output, n ist Input Variable, Emit_BunchLength ist File Name der Funktion
try
n=varargin{1};
Population=varargin{2};

Inputoptions=varargin{3};  
RUN=Inputoptions.run;

system(sprintf('mkdir %s',Population.General.Folder));

X0=zeros(n,length(Population.VarParameter));
X=zeros(n,length(Population.FixParameter));
for h=1:n
        fprintf('[%d] Run Job %d\n',RUN,h(1)),
        [X0(h,:),X(h,:)]=WriteSimulation(h,Population);
end;

%Test1-Parameter for an index after tracking:
% 0 = tracking waits for running or is running
% 1 = tracking is done -> the only 
% 2 = tracking is quitting early
% 3 = tracking ends with intern warning
% 4 = inefficient cut of particle by the pinhole (see WriteData)
% 5 = index is dead / error file from a previous run

%-----------------------------------------------------------------
%transfer of all data to the outer world:
OUT=0;
% OUT.Xemit = DistStat.Dist_Xemit;
% OUT.Yemit = DistStat.Dist_Yemit;
% OUT.Zemit = DistStat.Dist_Zemit;
% %OptimizerList={'horzBeam','vertBeam','longBeam','horzEmit','vertEmit','longEmit','Mom','diffMom','reldiffMom','Energy','diffEnergy','reldiffEnergy','charge'};
% OUT.Objects=[Param.sig_x(:,end) ,Param.sig_y(:,end) ,Param.sig_z(:,end),...
%              Param.mean_x(:,end) ,Param.mean_y(:,end) ,Param.mean_z(:,end),...
%              Param.emit_x(:,end),Param.emit_y(:,end),Param.emit_z(:,end),...
%              Param.momentum_avr(:,end),Param.momentum_std(:,end),...
%              Param.momentum_std(:,end)./Param.momentum_avr(:,end),...
%              Param.energy_avr(:,end),Param.energy_std(:,end),...
%              Param.energy_std(:,end)./Param.energy_avr(:,end),...
%              Param.charge(:,end)];
  varargout{1}=OUT;


DataSet_X=X;DataSet_Y=0;
DataSet_X(:,13)=[];
hdf5write(sprintf('%s/Data_MLP_RUN.hdf5',Population.General.Folder),'/X',DataSet_X,'/Y',DataSet_Y);
system(sprintf('cp Astra %s/Astra',Population.General.Folder));
system(sprintf('cp generator %s/generator',Population.General.Folder));

varargout{2}=''; %errorfile
catch ME
    save 'Errorfile_TrackingFUN.mat';
    varargout{4}='Errorfile_TrackingFUN.mat';
end
end

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%SubFunctions:
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

function [RUNV,RUNF]=WriteSimulation(h,Population)

h=h(:);

RUNV=zeros(1,length(Population.VarParameter));
RUNF=zeros(1,length(Population.FixParameter));%size(RUNF)


for j=1:length(Population.VarParameter)
    RUNV(1,j)=Population.VarParameter(j).ScanValues(h);
end;


for j=1:length(Population.FixParameter), %size(Population.FixParameter(j).RunValue),
    RUNF(1,j)=Population.FixParameter(j).RunValue(h);
    if Population.FixParameter(j).Flag==3
         RUNF(1,j)=Population.FixParameter(j).Value(RUNF(1,j));
    end;
end;

%size(RUN)
for i=h' 
    stest=fopen(Population.General.GeneratorFile,'r'); L=fscanf(stest,'%c'); fclose(stest); 
    stest=fopen(Population.General.ASTRAFile,'r');     M=fscanf(stest,'%c'); fclose(stest); 
   
    %implement Parameter either in GENERATOR or ASTRA file
    for j=1:length(Population.VarParameter)
        t=0;
        if ~isempty(strfind(L,Population.VarParameter(j).ASTRA))
%             if strcmp(Population.VarParameter(j).Format,'F'),
                L=strrep(L,Population.VarParameter(j).ASTRA,sprintf('%06.4e',RUNV(1,j)));
%             elseif strcmp(Population.VarParameter(j).Format,'I'),
%                 L=strrep(L,Population.VarParameter(j).ASTRA,sprintf('%05d',RUNV(1,j)));
%             end;
                
            t=t+1;
        end;
        if ~isempty(strfind(M,Population.VarParameter(j).ASTRA))
            %if strcmp(Population.VarParameter(j).Format,'F'),
                M=strrep(M,Population.VarParameter(j).ASTRA,sprintf('%06.4e',RUNV(1,j)));
           % elseif strcmp(Population.VarParameter(j).Format,'I'),
            %    M=strrep(M,Population.VarParameter(j).ASTRA,sprintf('%05d',RUN(i,j)));
           % end;
            t=t+1;
        end;
        if t==0, error('TrackingFUN:Name %s not found in generator and Astra file!',Population.VarParameter(j).ASTRA),end;
    end;
    for j=1:length(Population.FixParameter)
       t=0;
       if ~isempty(strfind(L,Population.FixParameter(j).ASTRA))
           if strcmp(Population.FixParameter(j).Format,'F'),
              L=strrep(L,Population.FixParameter(j).ASTRA,sprintf('%06.4e',RUNF(1,j)));
           elseif strcmp(Population.FixParameter(j).Format,'I'),
              L=strrep(L,Population.FixParameter(j).ASTRA,sprintf('%d',RUNF(1,j)));
           end;
            t=t+1;
       end;
       if ~isempty(strfind(M,Population.FixParameter(j).ASTRA))
           if strcmp(Population.FixParameter(j).Format,'F'),
              M=strrep(M,Population.FixParameter(j).ASTRA,sprintf('%06.4e',RUNF(1,j)));
           elseif strcmp(Population.FixParameter(j).Format,'I'),
              M=strrep(M,Population.FixParameter(j).ASTRA,sprintf('%d',RUNF(1,j)));
           end;
            t=t+1;
       end;
       if t==0, error('TrackingFUN:Name %s not found in generator and Astra file!',Population.FixParameter(j).ASTRA),end;
    end;
    
    
    [~,fnameG]=regexp(L,'FNAME ?= ?''');
    fnameG=fnameG(1);
    a=regexpi(L(fnameG(1)+1:end),'''');
    fnameG(2)=fnameG(1)+a(1);
    [~,fnameA]=regexp(M,'Distribution ?= ?''');
    fnameA=fnameA(1);
    a=regexpi(M(fnameA(1)+1:end),'''');
    fnameA(2)=fnameA(1)+a(1);
    
    if ~isempty(fnameG)
        B=sprintf('StepDistRun_%04d_N0.ini',i);A1=L(1:fnameG(1));A2=L(fnameG(2):end);L=[A1,B,A2];
    else
        error('MATSPEA2ASTRA:TrackingFUN: GENERATOR file did not have an output file name (-> FNAME)!\n')
    end;
    
    O=fopen(sprintf('%s/GEN_RUN_%04d.in',Population.General.Folder,i),'w'); fprintf(O,'%s',L);fclose(O); %-> write Generatorfile
    
    if ~isempty(fnameA)
        M1=M(1:fnameA(1));
        M2=M(fnameA(2):end);
    else
        error('MATSPEA2ASTRA:TrackingFUN: ASTRA file did not have an input file name (-> Distribution)!\n')
    end;
    
        MB=sprintf('StepDistRun_%04d_N%d.ini',i,0);Mt=[M1,MB,M2];
        Mt=strrep(Mt,Population.General.ASTRAStart.Name,sprintf('%06.4e',Population.General.ASTRAStart.Value(1)));
        Mt=strrep(Mt,Population.General.ASTRAStop.Name, sprintf('%06.4e',Population.General.ASTRAStop.Value(1)));
        O=fopen(sprintf('%s/ASTRA_RUN_%04d_N%d.in',Population.General.Folder,i,0),'w'); fprintf(O,'%s',Mt);fclose(O);

end;   

end



function RunGenerator(h)

h=h(:);n=length(h);
    for j=h'
       system(sprintf('./generator GEN_RUN_%04d.in > GEN_Out_%d.out &',j,j));
    end     
pause(1)

Test=zeros(n,1);loop=0;
while sum(Test)~=n,loop=loop+1;    
   for i=1:n,j=h(i); 
      if Test(i)==0 
        try 
            importdata(sprintf('StepDistRun_%04d_N0.ini',j)); 
            Test(i)=1;           
        catch             
            pause(1)
        end;
      end;
   end;
   if loop==5, fprintf('RunGenerator#%d; No Output found!',j);break;end;
end;

end



function [Param,I_OK,I_BAD]=StatData(h,ind,Dist,Param)

h=h(:);I_OK=[];I_BAD=[];
for i=h'
    try
    Dist0=Dist{i,ind(i),1};    
    %Convert Data distribution into statistical values:     
    Param.emit_x(i,1)=sqrt(det(cov(Dist0(:,1:3:4))))/.511; %hor. norm. emittances of the last distribution
    Param.emit_y(i,1)=sqrt(det(cov(Dist0(:,2:3:5))))/.511; %hor. norm. emittances of the last distribution
    Param.emit_z(i,1)=sqrt(det(cov(Dist0(2:end,3:3:6))))/.511;
    Param.mean_x(i,1)=mean(Dist0(:,1))*1e3;
    Param.mean_y(i,1)=mean(Dist0(:,2))*1e3;
    Param.mean_z(i,1)=(mean(Dist0(2:end,3))+Dist0(1,3))*1e3;
    Param.sig_x(i,1)=std(Dist0(:,1))*1e3;
    Param.sig_y(i,1)=std(Dist0(:,2))*1e3;
    Param.sig_z(i,1)=std(Dist0(2:end,3))*1e3;
    Param.momentum_avr(i,1)=(mean(Dist0(2:end,6))+Dist0(1,6))/1e6;
    Param.momentum_std(i,1)=std(Dist0(2:end,6))/1e6;
    Param.energy_avr(i,1)=mean(sqrt((Dist0(2:end,6)+Dist0(1,6)).^2/1e12+.511^2)-.511);
    Param.energy_std(i,1)=std(sqrt((Dist0(2:end,6)+Dist0(1,6)).^2/1e12+.511^2)-.511);    
    Param.charge(i,1)=sum(abs(Dist0(:,8)));
    I_OK=cat(1,I_OK,i);
    catch ME
        I_BAD=cat(1,I_BAD,i);
    end
end;
end

