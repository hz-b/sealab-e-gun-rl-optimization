% Created by the authors of:
% Status Report of the Berlin Energy Recovery Linac Project BERLinPro, M. Abo-Bakr, DOI: 10.18429/JACoW-IPAC2016-TUPOW034
% SRF Gun Development for Energy Recovery Linac Applications, T. Kamps, DOI: 10.1364/EUVXRAY.2016.ET1A.3
% Setup and Status of an SRF Photoinjector for Energy-Recovery Linac Applications, T. Kamps, DOI: 10.18429/JACoW-IPAC2017-MOPVA010
% Development of a compact test facility for SRF Photoelectron injectors, J. Völker DOI: 10.18452/19322

function Optimization=ReadIntroFile(path)

%default:
Output.Emit='ON';
Output.PhaseSpace='ON';
Output.Error='ON';
Output.Plots='ON';

General.MaxIterations=1;
General.Archive=1;
General.Cernel=25; %default: run the calculation on 25 cernels parallel
General.ASTRAFile=[];
General.GeneratorFile=[];

General.ASTRAStop.Name ='__Start__';
General.ASTRAStart.Value=0 ;
General.ASTRAStop.Name ='__Stop__';
General.ASTRAStop.Value=2.0;
%VarParameter=[];FixParameter=[];


%setting: run external Optimisation input
run(path);

if ~exist('VarParameter','var'), 
    fprintf('MATSPEA2ASTRA:ReadIntroFile: No variable Parameter was found! \n     -> Run only with given fix parameter setting!\n'),
    VarParameter=[];
else
    for i=1:length(VarParameter)
%         if isfield(VarParameter(i),'ASTRA'),
%             if isempty(VarParameter(i).ASTRA),error('MATSPEA2ASTRA:ReadIntroFile: variable parameter %d has no ASTRA name!\n',i);end;
%         else error('MATSPEA2ASTRA:ReadIntroFile: variable parameter %d has no ASTRA name!\n',i);
%         end;
        if isfield(VarParameter(i),'Name'),
            if isempty(VarParameter(i).Name), VarParameter(i).Name=VarParameter(i).ASTRA; end; %if no MATLAB name exists, use ASTRA name as MATLAB name!
        else VarParameter(i).Name=VarParameter(i).ASTRA;
        end;
    
        if isfield(VarParameter(i),'Value'),
            if isempty(VarParameter(i).Value),error('MATSPEA2ASTRA:ReadIntroFile: variable parameter %d has no or a wrong Value!\n',i);
            %elseif (strcmpi(VarParameter(i).Variation,'F') && length(VarParameter(i).Value)~=2),error('MATSPEA2ASTRA:ReadIntroFile: variable parameter %d has no or a wrong Value!\n',i);
            end;
        else error('MATSPEA2ASTRA:ReadIntroFile: variable parameter %d has no or a wrong Value!\n',i);
        end;
        
        if isfield(VarParameter(i),'StepSize'),
            if isempty(VarParameter(i).StepSize),error('MATSPEA2ASTRA:ReadIntroFile: variable parameter %d has no or a wrong StepSize!\n',i);
            elseif length(VarParameter(i).StepSize)~=1,error('MATSPEA2ASTRA:ReadIntroFile: variable parameter %d has no or a wrong StepSize!\n',i);
            end;
        else error('MATSPEA2ASTRA:ReadIntroFile: variable parameter %d has no or a wrong StepSize!\n',i);
        end;
    end;
end;

if ~exist('FixParameter','var'), 
    FixParameter=[];
else
    for i=1:length(FixParameter)

        if isfield(FixParameter(i),'Name'),
            if isempty(FixParameter(i).Name), FixParameter(i).Name=FixParameter(i).ASTRA; end; %if no MATLAB name exists, use ASTRA name as MATLAB name!
        else FixParameter(i).Name=FixParameter(i).ASTRA;
        end;
        
        if isfield(FixParameter(i),'Rand'),
            if isempty(FixParameter(i).Rand) || strcmpi(FixParameter(i).Rand,'exact'), FixParameter(i).Rand='exact';FixParameter(i).Flag=0;
            elseif strcmpi(FixParameter(i).Rand,'equal') || strncmpi(FixParameter(i).Rand,'e',1), FixParameter(i).Flag=1;
            elseif strcmpi(FixParameter(i).Rand,'normal') || strncmpi(FixParameter(i).Rand,'n',1), FixParameter(i).Flag=2;
            elseif strcmpi(FixParameter(i).Rand,'discret') || strncmpi(FixParameter(i).Rand,'d',1), FixParameter(i).Flag=3;
            elseif strncmpi(FixParameter(i).Rand,'C',1),      
                
                 %if the variable parameter should be directly correlated
                 %with another parameter j. both parameter have to be
                 %discret distributed and must have the same length.
                 A=textscan(FixParameter(i).Rand,'%s%d');%double(A{2})
                 FixParameter(i).Flag=3;
                 FixParameter(i).Variation='C';                 
                 FixParameter(i).VariationCor=double(A{2});
                 if A{2}>=i, 
                     error('MATSPEA2ASTRA:ReadIntroFile: fix parameter %d has no or a wrong correlation partner (the partner needs to have a smaller Index!!)!\n',i);
                 end;                
            else FixParameter(i).Rand='exact';FixParameter(i).Flag=0;
            end;
        end;
        
        if isfield(FixParameter(i),'Format'),
             if      strncmpi(FixParameter(i).Format,'f',1),%float
                 FixParameter(i).Format='F';
             elseif  strncmpi(FixParameter(i).Format,'i',1),%integer
                  FixParameter(i).Format='I';
             elseif  strncmpi(FixParameter(i).Format,'s',1),%string
                  FixParameter(i).Format='S';
             else FixParameter(i).Format='F'; %default
             end;
         else
             FixParameter(i).Format='F'; %default
         end;
         
        if isfield(FixParameter(i),'Value'),
            if isempty(FixParameter(i).Value) 
                error('MATSPEA2ASTRA:ReadIntroFile: fix parameter %d, no Value!\n',i);
            end;
            if ((length(FixParameter(i).Value)==1) && FixParameter(i).Flag~=0)
                error('MATSPEA2ASTRA:ReadIntroFile: fix parameter %d, length of Value >1, for non-exact value !\n',i);
            end
            if ((length(FixParameter(i).Value)>2) && FixParameter(i).Flag~=3)
                error('MATSPEA2ASTRA:ReadIntroFile: fix parameter %d, length of Value >2 for non-discret variation !\n',i);
            end;
            
        else error('MATSPEA2ASTRA:ReadIntroFile: fix parameter %d has no or a wrong Value!\n',i);
        end;
    end;    
    end    

    CorParameter=zeros(length(FixParameter),1);
    %CorrelationMatrixRun:
    cor=0;
    for i=1:length(FixParameter)
        if strcmp(FixParameter(i).Variation,'C'),            
            if CorParameter(FixParameter(i).VariationCor)<=0,
                cor=cor+1;
                CorParameter(i)=cor;
                CorParameter(FixParameter(i).VariationCor)=cor;
            else
                CorParameter(i)=CorParameter(FixParameter(i).VariationCor);
            end;
        elseif strcmp(FixParameter(i).Variation,'D') && (CorParameter(i)==0), 
            CorParameter(i)=-1;
        end;        
    end;



if isempty(General.ASTRAFile), error('MATSPEA2ASTRA:ReadIntroFile: No initial ASTRA file was found!\n');end;
if isempty(General.GeneratorFile), error('MATSPEA2ASTRA:ReadIntroFile: No initial GENERATOR file was found!\n');end;
%if General.Archive==1, General.Archive=General.Population;end %default: if Archive size is not given, it will be set to the population number
if isempty(General.ASTRAStart.Name) || isempty(General.ASTRAStart.Value) || isempty(General.ASTRAStop.Name) || isempty(General.ASTRAStop.Value),
    error('MATSPEA2ASTRA:ReadIntroFile: Missing Start/Stop parameter for ASTRA!\n');
end;
if length(General.ASTRAStart.Value)~=length(General.ASTRAStop.Value),
    error('MATSPEA2ASTRA:ReadIntroFile: Length of Start/Stop parameter vectors have to be the same!\n');
else
    if min(diff([General.ASTRAStart.Value;General.ASTRAStop.Value],1,1))<=0
        error('MATSPEA2ASTRA:ReadIntroFile: Stop values have to be larger than Start values!\n');
    end;
end;

Optimization.Output=Output;
Optimization.General=General;
Optimization.FixParameter=FixParameter(:)';
Optimization.VarParameter=VarParameter(:)';
Optimization.CorParameter=CorParameter(:)';
%Optimization.Object=Object;
%Optimization.Optimizer=Optimizer;
