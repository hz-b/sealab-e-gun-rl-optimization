% Created by the authors of:
% Status Report of the Berlin Energy Recovery Linac Project BERLinPro, M. Abo-Bakr, DOI: 10.18429/JACoW-IPAC2016-TUPOW034
% SRF Gun Development for Energy Recovery Linac Applications, T. Kamps, DOI: 10.1364/EUVXRAY.2016.ET1A.3
% Setup and Status of an SRF Photoinjector for Energy-Recovery Linac Applications, T. Kamps, DOI: 10.18429/JACoW-IPAC2017-MOPVA010
% Development of a compact test facility for SRF Photoelectron injectors, J. Völker DOI: 10.18452/19322

function OUT=ASTRA_Sampler(varargin)

rng('shuffle')                        % Für random Funktion neues seed
tic  
path=varargin{1};

reload=0;LoadFile='';StartIndex=1;
if length(varargin)>1
    for i=2:length(varargin)
        if ischar(varargin{i})
            a=strfind(varargin{i},'LoadFile');
            if ~isempty(a),
                LoadFile=varargin{i+1};reload=1;
                fprintf('Reload - Option\n'),
            end;
        end;  
    end;
end;


Time0=datestr(datetime('now'),'yyyy_mm_dd_HH-MM');
ReadMeFile=['Sampler_ReadMe_',Time0,'.dat'];
FinalFile=['Sampler_',Time0,'.mat'];
File=fopen(ReadMeFile,'a');% Start Stoppuhr

OUT.Error=[];
OUT.File=FinalFile;


try
%% Problem Definition

if reload==0
SAMPLER=ReadIntroFile(path);


%% SPEA2 Settings
%%------------------------------------------------------------------
MaxIt = SAMPLER.General.MaxIterations;          % Maximum Number of Iterations = T

%nPop = SAMPLER.General.Population;            % Population Size = N
nVarPar = length(SAMPLER.VarParameter);
nFixPar = length(SAMPLER.FixParameter);

options=[];
if isfield(SAMPLER.General,'ProgramFile')
    ProgRoutine=str2func(SAMPLER.General.ProgramFile.file); % Programmangabe für das Tracking und Datenerzeugung
    if isfield(SAMPLER.General.ProgramFile,'options')
        options=SAMPLER.General.ProgramFile.options;
    end;
else
    ProgRoutine=str2func('Emit_BunchLength');
end;

%%------------------------------------------------------------------
% ASTRAFile=Optimization.General.ASTRAFile;
% GeneratorFile=Optimization.General.GeneratorFile;
pop.VarParameter=SAMPLER.VarParameter;
pop.FixParameter=SAMPLER.FixParameter;
pop.General=SAMPLER.General;

pop.CorParameter=SAMPLER.CorParameter;

pop.Index = [];
for i=1:nVarPar
    pop.VarParameter(i).RunValues=[];
end;

%pop = repmat(pop,MaxIt+1,1);      % parent population definieren  

pop(1).Index = 1:MaxIt;
SCANdim=zeros(1,nVarPar);
SCANvec=cell(1,nVarPar);
SCANALL=0;SCANWIDTH=1;
for i=1:nVarPar
    SCANvec{i}=pop(1).VarParameter(i).Value(1):pop(1).VarParameter(i).StepSize:pop(1).VarParameter(i).Value(2);
    SCANdim(i)=length(SCANvec{i});
end;
 SCANMesh=cell(1,nVarPar);
if nVarPar==1,
    SCANMesh{1}=SCANvec{1};
    if strcmp(pop(1).VarParameter(1).Name,'ALL'), 
        SCANALL=1; 
        SCANWIDTH=length(SCANvec{1});
        pop(1).VarParameter=[];nVarPar=0;
    end; %all fix parameter will be randomized scanned with SCANWIDTH as th nuzmber of variations per iteration.
elseif nVarPar==2,
    SCANMesh{1}=repmat(SCANvec{1}(:),1,SCANdim(2));
    SCANMesh{2}=repmat(SCANvec{2}(:)',SCANdim(1),1);    
    
elseif nVarPar==3,
    SCANMesh{1}=repmat(SCANvec{1}(:),[1,SCANdim(2),SCANdim(3)]);
    SCANMesh{2}=repmat(SCANvec{2}(:)',[SCANdim(1),1,SCANdim(3)]);
    SCANMesh{3}=repmat(permute(SCANvec{3}(:),[3,2,1]),[SCANdim(1),SCANdim(2),1]);

elseif nVarPar==4,
    SCANMesh{1}=repmat(SCANvec{1}(:),[1,SCANdim(2),SCANdim(3),SCANdim(4)]);
    SCANMesh{2}=repmat(SCANvec{2}(:)',[SCANdim(1),1,SCANdim(3),SCANdim(4)]);
    SCANMesh{3}=repmat(permute(SCANvec{3}(:),[3,2,1]),[SCANdim(1),SCANdim(2),1,SCANdim(4)]);
    SCANMesh{4}=repmat(permute(SCANvec{3}(:),[4,3,2,1]),[SCANdim(1),SCANdim(2),SCANdim(3),1]);

else
    error('ASTRA_Sample: can only handle Parameter scans up to 4 dimensions.');
end;
nSCAN=prod(SCANdim);

pop = repmat(pop,MaxIt+1,1);      % parent population definieren  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf(File,'variable Input-Parameter: \n');
for i=1:nVarPar,
    fprintf(File,'%s (%s) :  [%s] (%s)\n ',pop(1).VarParameter(i).Name,pop(1).VarParameter(i).ASTRA,sprintf('%f ',pop(1).VarParameter(i).Value));
end;
fprintf(File,'fix Input-Parameter: \n');
for i=1:nFixPar,
    fprintf(File,'%s (%s) :  [%s]\n ',pop(1).FixParameter(i).Name,pop(1).FixParameter(i).ASTRA,sprintf('%f ',pop(1).FixParameter(i).Value));
end;
save([ReadMeFile '_Initial.mat']);


 toc                                                  % Erste Ausgabe Stoppuhr
else
    
    StartIndexN=1;LoadFileN=LoadFile;vararginN=varargin;
    for i=2:length(varargin)
        if ischar(varargin{i})
            a=strfind(varargin{i},'StartIndex');
            if ~isempty(a),
                StartIndexN=varargin{i+1};
                fprintf('StartIndex - Option\n')
            end;
        end; 
    end;
    fprintf('reload file: %s, starting at Index: %d.\n',LoadFileN,StartIndexN),
    load(LoadFileN);
    LoadFile=LoadFileN;
    StartIndex=StartIndexN;
    varargin=vararginN;
end;
for it=StartIndex:MaxIt                                         % Durchlauf Iterationen
    
    fprintf('Iteration: %d\n',it)
for i=1:nVarPar
    
    pop(it).VarParameter(i).ScanValues=reshape(SCANMesh{i},nSCAN,1);
end;

    CorVarPar=[];
    for i=1:nFixPar
        if pop(it).FixParameter(i).Flag==0 %constant value
            pop(it).FixParameter(i).RunValue=pop(it).FixParameter(i).Value*ones(SCANWIDTH,1);
        elseif pop(it).FixParameter(i).Flag==1 %equal distribution
            pop(it).FixParameter(i).RunValue=rand(SCANWIDTH,1)*abs(diff(pop(it).FixParameter(i).Value))+min(pop(it).FixParameter(i).Value); %random numbers in the given interval of the parameter (Free variation)
        elseif pop(it).FixParameter(i).Flag==2 %normal distribution
            pop(it).FixParameter(i).RunValue=randn(SCANWIDTH,1)*abs(pop(it).FixParameter(i).Value(2))+pop(it).FixParameter(i).Value(1); %random rms numbers with the width from the second value and the offset as the frist value from the Input
        elseif pop(it).FixParameter(i).Flag==3 %discret values from a given vector
            if strcmp(pop(it).FixParameter(i).Variation,'C')
                j=pop(it).FixParameter(i).VariationCor;
                CorVarPar=cat(1,CorVarPar,[i,j]);
                pop(it).FixParameter(i).RunValue=pop(it).FixParameter(j).RunValue; %random choice from the given parameter values (Discret variation)
            else
                pop(it).FixParameter(i).RunValue=randi(length(pop(it).FixParameter(i).Value),SCANWIDTH,1); %random choice from the given parameter values (Discret variation)
%                 l=pop(it).FixParameter(i).Value;
%                 pop(it).FixParameter(i).RunValue=l(randi(length(l))); %random choice from the given parameter values (Discret variation)
            end;
        end;
    end;
    pop(it).General.Folder=sprintf('RUN_%d',it);
    options.run=it;
    %options.BreakFlag=BreakFlag;
    options.file=ReadMeFile;
    [pop(it).OutputData,errorfile] =  ProgRoutine(nSCAN, pop(it),options);      
     if ~isempty(errorfile), 
         error('MATSPEA2ASTRA: Sampler : ProgRoutine -> %s',errorfile), 
     end

    save('Running_BufferData.mat');

end

save(FinalFile);
 
 
toc                                            % 2. Ausgabe Stoppuhr

catch ME
    save([ReadMeFile '_Errorfile.mat']);
    OUT.Error=ME;
    fprintf(ME.message)
end;
end
