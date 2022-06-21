% The script spectra_raman_DiGenova-et-al-ChemGeo.m performs a full set of analyses on raman spectra 
%in the low-wavenumber and high-wavenumber regions. 
%
% You are free to use/alterate this script after quoting the original
% source: Di Genova et al Chem. Geol. Effect of iron and nanolites on Raman 
% spectra of volcanic glasses: reassessment of existing strategies to estimate the water content
%
% In details, correction for the frequency-dependence scattering intensity
% of the green laser and temperature (e.g. Long, 1977; Neuville and Mysen, 1996) is applied,
% then the script performs background subtraction and, finally, spectra are normalized for the silicate and water band areas 
% and the ratio between the two areas and the water concentration is computed.
%
% Before running spectra_raman_DiGenova-et-al-ChemGeo.m, for each of your sample, please copy 2 separate txt files in your folder as it follows:
% filenameLW.txt for the silicate region (from 50 to 500 cm-1) and
% filenameHW.txt for the water region (from 2700 to 4000 cm-1).
%
% Note: File names must contain identificative characters 'LW' and 'HW' for
% silicate and water regions, respectively.
% 
% The input file must contain only two columns: the first one is the wavenumber, whereas the second one represents the intensity. 
%
% INPUT arguments: laser wavelength (default value is given for the green laser)
%
% OUTPUT: The script generates 4 output sub-folders and 1 txt file.
%   Folders:
%   1- Long Corrected: corrected spectra for high and low wavenumber regions (*_cor.txt);
%   2- Subtracted: corrected spectra after background removal for high and low wavenumber regions (*_sub.txt);
%   3- Normalized to Silicate Area: normalized spectra to silicate area (*_norm2Si.txt);
%   4- Normalized to Silicate Area:normalized spectra to the total area (*_norm2tot.txt). 
%
%   Finally, absolute values of Silicate(LW) and Water(HW) band areas and their the ratio are stored in the 'Area_Data.txt' file.  
%   Columns are: 1: Spectrum label - 2: Silicate(LW) area - 3: Water(HW) area - 4: Silicate(LW)/Water(HW) area ratio
%


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  input   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc

% Define laser wavelength (nm)
laser=532;
%Define baseline interval in the low-wavenumber (LW) region (Silicate)
First_Si=[0 200];
Second_Si=[1240 1500];
%Define baseline interval in the high-wavenumber (HW) region (Water)
First_Wt=[2750 3100];
Second_Wt=[3750 3900];




%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  beginning of code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define Input folder
disp('Select Input Folder -- Hit Return')
[dir_input]             = uigetdir('*.*','Select Input Folder');
% Define Output folder
disp('Select Output Folder -- Hit Return')
[dir_output]             = uigetdir('*.*');

% Create Output subfolders
mkdir(dir_output,'1-Long Corrected')
dir_output_LC=fullfile(dir_output,'1-Long Corrected');
mkdir(dir_output,'2-Subtracted')
dir_output_Sub=fullfile(dir_output,'2-Subtracted');
mkdir(dir_output,'3-Normalized to Silicate Area')
dir_output_N2Si=fullfile(dir_output,'3-Normalized to Silicate Area');
mkdir(dir_output,'4-Normalized to total Area')
dir_output_N2tot=fullfile(dir_output,'4-Normalized to total Area');

% List low-wavenumber (LW) and high-wavenumber (HW) region files
list_all_LW=dir(fullfile(dir_input,'*LW*'));
list_all_HW=dir(fullfile(dir_input,'*HW*'));

% Create empty variables for following steps
area_Si_file=[];
area_Wt_file=[];
Ratio_file=zeros(1, 5);
samplelabel=[];
all_spectra_Si=[];
all_spectra_Wt=[];
all_x_Si=[];
all_x_Wt=[];

id='MATLAB:polyfit:RepeatedPointsOrRescale';
warning('off',id)
%% analyse group spectra LW

for i=1:length(list_all_LW);
    % open file txt
    filename2=list_all_LW(i).name;
    fid=fopen(fullfile(dir_input,filename2));
    file=textscan(fid, '%f %f');
    fclose(fid);
    % store  variables
    spectra2(:,1)=file{1,1};
    spectra2(:,2)=file{1,2};
    % apply correction for frequency-dependence scattering intensity and
    % temperature
    spectra2(:,2)=spectra2(:,2).*(10^7/laser)^3.*(1-exp(-6.62607*10.^(-34)*29979245800.*spectra2(:,1)/(1.3806488*10^(-23)*293.15))).*spectra2(:,1)./((10^7/532)-spectra2(:,1)).^4;
    % plot spectra figure
    figure
    plot(spectra2(:,1), spectra2(:,2));
    xlabel('Wavenumber(cm^-^1)')
    ylabel('Intensity (a.u.)')
    title(filename2,'Interpreter','none')
    hold on
    % Create interpolation
    ts=spectra2(1,1):1:spectra2(end,1);
    ts=ts';
    spectra_y=interp1q(spectra2(:,1),spectra2(:,2), ts);
    spectra_x=ts;
    % Write output
    dlmwrite(fullfile(dir_output_LC,strcat(filename2(1:end-4), '_cor.txt')),spectra2,'delimiter','\t');
    
    % Selecting nodes
        k=find(spectra_x<=First_Si(1,2));
        int1=spectra_x(k,1);
        int1=int1';
        l=find(spectra_x>=Second_Si(1,1) & spectra_x<=Second_Si(1,2));
        int2=spectra_x(l,1);
        int2=int2';
        Nodes_2=[int1 int2];
        Nodes_2=Nodes_2';
    % Find Intensity value for Nodes
    CInt_interp = interp1(spectra_x,spectra_y, Nodes_2,'spline') ;
    % Plot Nodes
    plot(Nodes_2,CInt_interp,'or')
    %Create baseline
    p = polyfit(Nodes_2,CInt_interp,3);
    end_v=[Nodes_2(1,1): Nodes_2(end,1)];
    y1=polyval(p,end_v);
    % Plot baseline
    hold on
    plot(end_v, y1)
    % Subtract baseline
    y1=y1';
    Sub_Int_Si=spectra_y-y1;
    % Plot Subtracted spectrum
    plot(spectra_x,Sub_Int_Si,'g');
    legend('Raw Spectrum (LW)','Nodes','Baseline','Subtracted Spectrum','Location','Best')
    % save Subtracted spectrum
    out_Spectrum=[spectra_x Sub_Int_Si];
    dlmwrite(fullfile(dir_output_Sub,strcat(filename2(1:end-4),'_sub.txt')),out_Spectrum,'delimiter','\t');
    % Calculate Silicate Area
    area_Si=trapz(Sub_Int_Si);
    area_Si_file=[area_Si_file; area_Si];
    all_spectra_Si=[all_spectra_Si Sub_Int_Si];
    all_x_Si=[all_x_Si spectra_x];
    clearvarlist= ['clearvarlist';setdiff(who,{'list_all_LW';'list_all_HW';'dir_input'; 'dir_output'; 'dir_output_LC'; 'dir_output_Sub'; 'dir_output_N2tot'; 'dir_output_N2Si'; 'laser'; 'all_x_Si'; 'all_spectra_Si'; 'all_x_Wt'; 'all_spectra_Wt'; 'area_Si_file';'area_Wt_file';'Ratio_file';'samplelabel';'First_Si'; 'Second_Si'; 'ThirdBIR_Si'; 'ForthBIR_Si';'First_Wt'; 'Second_Wt'})];
    clear(clearvarlist{:});
end
%% analyse group spectra HW

for i=1:length(list_all_HW);
    % open file
    filename1=list_all_HW(i).name;
    fid=fopen(fullfile(dir_input,filename1));
    file=textscan(fid, '%f %f');
    fclose(fid);
    % create spectra variables
    spectra1(:,1)=file{1,1};
    spectra1(:,2)=file{1,2};
    % apply correction for frequency-dependence scattering intensity and
    % temperature
    spectra1(:,2)=spectra1(:,2).*(10^7/laser)^3.*(1-exp(-6.62607*10.^(-34)*29979245800.*spectra1(:,1)/(1.3806488*10^(-23)*293.15))).*spectra1(:,1)./((10^7/532)-spectra1(:,1)).^4;
    % plot figure
    figure
    plot(spectra1(:,1), spectra1(:,2));
    xlabel('Wavenumber(cm^-^1)')
    ylabel('Intensity (a.u.)')
    title(filename1,'Interpreter','none')
    hold on
    % Write output
    dlmwrite(fullfile(dir_output_LC,strcat(filename1(1:end-4), '_cor.txt')),spectra1,'delimiter','\t');
    % Create interpolation
    ts=round(spectra1(1,1):1:spectra1(end,1));
    ts=ts';
    spectra_y=interp1q(spectra1(:,1),spectra1(:,2), ts);
    spectra_x=ts;
    % Selecting nodes
    k=find(spectra_x<=First_Wt(1,2) & spectra_x>=First_Wt(1,1));
    int1=spectra_x(k,1);
    int1=int1';
    l=find(spectra_x>=Second_Wt(1,1) & spectra_x<=Second_Wt(1,2)); %abbassata la soglia per orda 3940 a 3900
    int2=spectra_x(l,1);
    int2=int2';
    Nodes_2=[int1 int2];
    Nodes_2=Nodes_2';
    % Find Intensity value for Nodes
    CInt_interp = interp1(spectra_x,spectra_y, Nodes_2,'spline') ;
    % Plot Nodes
    plot(Nodes_2,CInt_interp,'or')
    %Create baseline
    p = polyfit(Nodes_2,CInt_interp,3);
    end_v=[Nodes_2(1,1): Nodes_2(end,1)];
    y1=polyval(p,end_v);
    % Plot baseline
    hold on
    plot(end_v, y1)
    % Subtract baseline
    cut=find(spectra_x==3900 );
    cut_pre=find(spectra_x==2750);
    spectra_y_cut=spectra_y(cut_pre:cut);
    spectra_x_cut=spectra_x(cut_pre:cut);
    y1=y1';
    Sub_Int_Wt=spectra_y_cut-y1;
    % Plot Subtracted spectrum
    plot(spectra_x_cut,Sub_Int_Wt,'g')
    legend('Raw Spectrum (HW)','Nodes','Baseline','Subtracted Spectrum','Location','NorthWest')
    % save Subtracted spectrum
    out_Spectrum=[spectra_x_cut Sub_Int_Wt];
    %Write output
    dlmwrite(fullfile(dir_output_Sub,strcat(filename1(1:end-4),'_sub.txt')),out_Spectrum,'delimiter','\t');
    % Calculate Water Area
    area_Wt=trapz(Sub_Int_Wt);
    area_Wt_file=[area_Wt_file; area_Wt];
    % Save and clean workspace
    all_spectra_Wt=[all_spectra_Wt Sub_Int_Wt];
    all_x_Wt=[all_x_Wt spectra_x_cut];
    clearvarlist= ['clearvarlist';setdiff(who,{'list_all_LW';'list_all_HW';'dir_input'; 'dir_output'; 'dir_output_LC'; 'dir_output_Sub'; 'dir_output_N2tot'; 'dir_output_N2Si'; 'laser'; 'all_x_Si'; 'all_spectra_Si'; 'all_x_Wt'; 'all_spectra_Wt'; 'area_Si_file';'area_Wt_file';'Ratio_file';'samplelabel';'First_Si'; 'Second_Si'; 'ThirdBIR_Si'; 'ForthBIR_Si';'First_Wt'; 'Second_Wt'})];
    clear(clearvarlist{:});
end
%%  Normalize spectra to Total Area and to Silicate Area only

Merged_SpectraY=[all_spectra_Si; all_spectra_Wt];
Merged_SpectraX=[all_x_Si; all_x_Wt];

for i=1:length(area_Wt_file)
    Total_Area(i,1)=area_Wt_file(i,1)+area_Si_file(i,1);
    %Normalize to total Area
    Norm_spect_tot(:,i)=Merged_SpectraY(:,i)./Total_Area(i,1);
    filename1=list_all_HW(i).name;
    fname_general=strrep(filename1,'_HW','');
    out_Norm_tot=[Merged_SpectraX(:,i),Norm_spect_tot(:,i)];
    %Write output
    dlmwrite(fullfile(dir_output_N2tot, strcat(fname_general(1:end-4),'_norm2tot.txt')),out_Norm_tot,'delimiter','\t')
    clear out_Norm_tot
    %Normalize to Silicate Area
    Norm_spect_Si(:,i)=Merged_SpectraY(:,i)./area_Si_file(i,1);
    out_Norm_Si=[Merged_SpectraX(:,i),Norm_spect_Si(:,i)];
    %Write output
    dlmwrite(fullfile(dir_output_N2Si, strcat(fname_general(1:end-4),'_norm2Si.txt')),out_Norm_Si,'delimiter','\t')
    clear out_Norm_Si
end
%%   Calculate Area Ratio and Water Concentration
fileID=fopen(fullfile(dir_output,'Area_Data.txt'),'w');
formatSpec = '%s %f %f %f\n';
for i=1:length(area_Wt_file)
    filename1=list_all_HW(i).name;
    fname_general={strrep(filename1,'_HW','')};
    Ratio=area_Wt_file(i,1)/area_Si_file(i,1);
    output={str2mat(fname_general), area_Si_file(i,1),area_Wt_file(i,1), Ratio};
    fprintf(fileID, formatSpec, output{1,:});
end
fclose(fileID);
