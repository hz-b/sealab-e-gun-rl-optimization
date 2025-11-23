% Created by the authors of:
% Status Report of the Berlin Energy Recovery Linac Project BERLinPro, M. Abo-Bakr, DOI: 10.18429/JACoW-IPAC2016-TUPOW034
% SRF Gun Development for Energy Recovery Linac Applications, T. Kamps, DOI: 10.1364/EUVXRAY.2016.ET1A.3
% Setup and Status of an SRF Photoinjector for Energy-Recovery Linac Applications, T. Kamps, DOI: 10.18429/JACoW-IPAC2017-MOPVA010
% Development of a compact test facility for SRF Photoelectron injectors, J. Völker DOI: 10.18452/19322

General.MaxIterations=10;
General.Cernel=50000;
General.ASTRAFile='ASTRA_Setup.in';
General.ASTRAStart.Name ='__Start__';
General.ASTRAStart.Value=0;
General.ASTRAStop.Name ='__Stop__';
General.ASTRAStop.Value=1.737;
General.GeneratorFile='Generator_Setup.in';
General.ProgramFile.file='TrackingWrite';

% General.ASTRAChangeGunPivotPoint.Name='GunCavity';
% General.ASTRAChangeGunPivotPoint.XAngleName='__rotX__';
% General.ASTRAChangeGunPivotPoint.YAngleName='__rotY__';
% General.ASTRAChangeGunPivotPoint.XCathPosName='__SpotXPos__';
% General.ASTRAChangeGunPivotPoint.YCathPosName='__SpotYPos__';


General.AutoScanSolenoid={};

VarParameter(1).Name='ALL'; %all parameter will be "scanned" in their own variation
%VarParameter(1).ASTRA='__Gun_Phase__';
VarParameter(1).Value=[0,1]; %full range for the individual parameter ranges (2 -> two times the given parameter range)
VarParameter(1).StepSize=1/50000; %resulting in 500 variation per evolution



%-----------------------------------------
%abs. fix:

FixParameter(1).Name='Laser pulse lelength (ns)';
FixParameter(1).ASTRA='__pulselength__';
FixParameter(1).Rand='equal';
FixParameter(1).Value=[0.6,4]*1e-3;

FixParameter(2).Name='Laser spot size on cathode (mm)';
FixParameter(2).ASTRA='__spotsize__';
FixParameter(2).Rand='equal';
FixParameter(2).Value=[.2,.8];

FixParameter(3).Name='Bunch Charge [nC]';
FixParameter(3).ASTRA='__QBunch__';
FixParameter(3).Value=0.1e-3;

FixParameter(4).Name='Solenoid position (m)';
FixParameter(4).ASTRA='__zSol__';
FixParameter(4).Value=0.4625;

%-----------------------------------------
%variable fix (fix for a scan):

FixParameter(5).Name='Gun peak field [MV/m]';
FixParameter(5).ASTRA='__Gun_Epeak__';
FixParameter(5).Rand='equal';
FixParameter(5).Value=[12,18];

FixParameter(6).Name='Gun DC bias field [kV]';
FixParameter(6).ASTRA='__Gun_Bias__';
FixParameter(6).Rand='equal';
FixParameter(6).Value=[3,5];

FixParameter(7).Name='Cathode Position []';
FixParameter(7).ASTRA='__Gun_Field__';
FixParameter(7).Rand='discret';
FixParameter(7).Format='I';
FixParameter(7).Value=[-20:-10];

FixParameter(8).Name='Field Flattnes []';
FixParameter(8).ASTRA='__FF__';
FixParameter(8).Rand='equal';
FixParameter(8).Value=[-0.5,0.5];

FixParameter(9).Name='Laser hor. position [mm]';
FixParameter(9).ASTRA='__SpotXPos__';
FixParameter(9).Rand='equal';
FixParameter(9).Value=[-1.5,1.5];

FixParameter(10).Name='Laser ver. position [mm]';
FixParameter(10).ASTRA='__SpotYPos__';
FixParameter(10).Rand='equal';
FixParameter(10).Value=[-1.5,1.5];

FixParameter(11).Name='Solenoid hor. position [mm]';
FixParameter(11).ASTRA='__SolXPos__';
FixParameter(11).Rand='equal';
FixParameter(11).Value=[-2,2]*1e-3;

FixParameter(12).Name='Solenoid ver. position [mm]';
FixParameter(12).ASTRA='__SolYPos__';
FixParameter(12).Rand='equal';
FixParameter(12).Value=[-2,2]*1e-3;

FixParameter(13).Name='Solenoid Angle Y-axes [rad]';
FixParameter(13).ASTRA='__rotX__';
FixParameter(13).Rand='equal';
FixParameter(13).Value=[-10,10]*1e-3;

FixParameter(14).Name='Solenoid Angle X-axes [rad]';
FixParameter(14).ASTRA='__rotY__';
FixParameter(14).Rand='equal';
FixParameter(14).Value=[-10,10]*1e-3;

FixParameter(15).Name='Cathode Position (bunch) [mm]';
FixParameter(15).ASTRA='__SpotZPos__';
FixParameter(15).Rand='c 7';
FixParameter(15).Format='F';
FixParameter(15).Value=[-20:-10]*1e-4;

FixParameter(16).Name='Emission phase [deg]';
FixParameter(16).ASTRA='__Gun_Phase__';
FixParameter(16).Value=[-10,60];
FixParameter(16).Rand='equal';

FixParameter(17).Name='Solenoid strength [T]';
FixParameter(17).ASTRA='__BSol__';
FixParameter(17).Value=[-70,70]*1e-3;
FixParameter(17).Rand='equal';

% FixParameter(9).Name='Cavity Angle Y-axes [rad]';
% FixParameter(9).ASTRA='__rotX__';
% FixParameter(9).Rand='equal';
% FixParameter(9).Value=0;
% 
% FixParameter(10).Name='Cavity Angle X-axes [rad]';
% FixParameter(10).ASTRA='__rotY__';
% FixParameter(10).Rand='equal';
% FixParameter(10).Value=0;

Output.Emit='ON';
Output.PhaseSpace='ON';



