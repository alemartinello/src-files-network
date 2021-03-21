%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
%    BAYESIAN ESTIMATION, ANALYSIS AND REGRESSION (BEAR) TOOLBOX        %
%                                                                       %
%    This statistical package has been developed by the external        %
%    developments division of the European Central Bank.                %
%                                                                       %
%    Authors:                                                           %
%    Romain Legrand  (Romain.Legrand@ecb.europa.eu)                     %
%    Alistair Dieppe (Alistair.Dieppe@ecb.europa.eu)                    %
%    Björn van Roye  (Bjorn.van_Roye@ecb.europa.eu)                     %
%                                                                       %
%    Version 4.2                                                        %
%                                                                       %
%    The authors are grateful to the following people for valuable input%
%    and advice which contributed to improve the quality of the toolbox:%                                               %
%    Paolo Bonomolo, Marta Banbura, Fabio Canova, Matteo Ciccarelli,    %
%    Marek Jarocinski, Carlos Montes-Caldon, Michele Lenza, Chiara Osbat%
%    Gary Koop, Giorgio Primiceri, Michal Rubaszek and Barbara Rossi.   %
%    We would also partciularly like to thank Francesca Loria for       %
%    incorporating the forecast evaluation procedures.                  %
%                                                                       %
%    These programmes are the responsibilities of the authors and not of% 
%    the ECB and all errors and ommissions remain those of the authors. %
%                                                                       %
%    Using the BEAR toolbox implies acceptance of the End User Licence  %
%    Agreement and appropriate acknowledgement should be made.          %                                                        %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars -except respons stod spec IRFoutput kk varendo_base varendo_omx varendo_inter varendo_eplast varendo_immfirst varendo_vix lag_base lag_et lag_to varexo_base varexo_notrend print export savee varname
%clear all
close all
warning off;
clc

%-----------------|
% Interface phase |
%---------------- |

% first create initial elements to avoid later crash of the code
init;

% read the gui script: if the value is set to 1, the interfaces are used
% if the user switches the value to 0, the developer's version is used
gui;
%GUI=0;

% if the user wants to use the graphical interface
if GUI==1
    
   % initiate the interface phase

   % set the code to start with the first interface
   interface='interface1';
   % then run the interface phase: if it is not yet over...
   while strcmp(interface,'over')==0
   % ... then launch the relevant interface   
   
      % either interface 1 (initial interface)
      if strcmp(interface,'interface1')    
      [VARtype,frequency,startdate,enddate,endo,exo,lags,const,pref,interface,interinfo1]=interface1(interinfo1);
      % interface 2 (BVAR interface)
      elseif strcmp(interface,'interface2')   
      [prior,ar,lambda1,lambda2,lambda3,lambda4,lambda5,lambda6,lambda7,lambda8,It,Bu,hogs,bex,scoeff,iobs,lrp,interface,interinfo2]=interface2(pref,interinfo2);
      % interface 3 (mean-adjusted BVAR)
      elseif strcmp(interface,'interface3')   
      [ar,lambda1,lambda2,lambda3,lambda4,lambda5,It,Bu,bex,interface,interinfo3]=interface3(pref,interinfo3);
      % interface 4 (panel BVAR)
      elseif strcmp(interface,'interface4')   
      [panel,Units,It,Bu,pick,pickf,ar,lambda1,lambda2,lambda3,lambda4,s0,v0,alpha0,delta0,gama,rho,a0,b0,psi,interface,interinfo4]=interface4(pref,interinfo4);  
      % interface 5 (stochastic volatility BVAR)
      elseif strcmp(interface,'interface5')  
      [stvol,It,Bu,pick,pickf,bex,ar,lambda1,lambda2,lambda3,lambda4,lambda5,gamma,alpha0,delta0,gamma0,zeta0,interface,interinfo5]=interface5(pref,interinfo5);
      % interface 7 (Time-Varying BVAR)
      elseif strcmp(interface,'interface7')  
      [tvbvar,It,Bu,pick,pickf,alltirf,gamma,alpha0,delta0,interface,interinfo7]=interface7(pref,interinfo7);
      % or interface 6 (application interface, which is the final interface)
      elseif strcmp(interface,'interface6')   
      [IRF,F,FEVD,HD,CF,IRFt,Feval,CFt,IRFperiods,Fstartdate,Fenddate,Fendsmpl,cband,IRFband,Fband,FEVDband,HDband,window_size,evaluation_size,hstep,pref,interface,interinfo6]=interface6(pref,enddate,frequency,VARtype,panel,interinfo6);
      end
    
   % if the interface phase is over, end it 
   end

% short pause to allow interfaces to close before the code goes on running
pause(0.1);

% if the user prefers to use the developer's version, execute the settings file
% this file contains the same information as the interfaces, without the need to run them
elseif GUI==0
bear_settings;
% run a script to convert string into a list of endogenous, exogenous, and units (if applicable)
convertstrngs;

end

priorf=100;

%--------------------|
% Data loading phase |
%------------------- |

addpath(pref.datapath);
% initiation of Excel result file
initexcel;  

% generate the different sets of data
% if the model is the OLS VAR, the Bayesian VAR, the mean-adjusted BVAR, the stochastic volatility BVAR, ot the time-varying BVAR:
if VARtype==1 || VARtype==2 || VARtype==3 || VARtype==5 || VARtype==6
[names data data_endo data_endo_a data_endo_c data_endo_c_lags data_exo data_exo_a data_exo_p data_exo_c data_exo_c_lags Fperiods Fcomp Fcperiods Fcenddate]...
=gensample(startdate,enddate,VARtype,Fstartdate,Fenddate,Fendsmpl,endo,exo,frequency,lags,F,CF,pref);
% else, if the model is the panel BVAR
elseif VARtype==4
[names data data_endo data_endo_a data_endo_c data_endo_c_lags data_exo data_exo_a data_exo_p data_exo_c data_exo_c_lags Fperiods Fcomp Fcperiods Fcenddate]...
=gensamplepan(startdate,enddate,Units,panel,Fstartdate,Fenddate,Fendsmpl,endo,exo,frequency,lags,F,CF,pref);
end

%---------------------|
% Table loading phase |
%------------------- -|

% grid search table
if VARtype==2 && hogs==1
[grid]=loadhogs(scoeff,iobs,pref);
end
% block exogeneity table
if (VARtype==2 || VARtype==3 || VARtype==5) && bex==1
[blockexo]=loadbex(endo,pref);
end
% Long run prior table
H=[];
if (VARtype==2) && lrp==1
H=loadH;
end

% sign restrictions table
if (VARtype==2 || VARtype==3 || VARtype==4 || VARtype==5 || VARtype==6) % && IRF==1 && IRFt==4
    [signrestable signresperiods signreslabels]=loadsignres(endo,pref, IRFt);
end
% mean-adjusted prior table
if VARtype==3
[equilibrium,chvar,regimeperiods,Fpconfint,Fpconfint2,regime1,regime2,Dmatrix]=loadmaprior(endo,exo,startdate,pref,data);
end
%if VARtype==3
%[Fpconfint]=loadmaprior(endo,exo,pref);
%end
% conditional forecast tables (for BVAR, mean-adjusted BVAR, and stochastic volatility BVAR)
if (VARtype==2 || VARtype==3 || VARtype==5 || VARtype==6) && CF==1
[cfconds cfshocks cfblocks cfintervals]=loadcf(endo,CFt,Fstartdate,Fenddate,Fperiods,pref);
% conditional forecast tables (for panel BVAR model)
elseif VARtype==4 && CF==1
[cfconds cfshocks cfblocks]=loadcfpan(endo,Units,panel,CFt,Fstartdate,Fenddate,Fperiods,pref);
end

%--------------------|
% Excel record phase |
%--------------------|

% record the estimation information
excelrecord1;

%-----------------------|
% date generation phase |
%-----------------------|

% generate the strings and decimal vectors of dates
[decimaldates1,decimaldates2,stringdates1,stringdates2,stringdates3,Fstartlocation,Fendlocation]=gendates(names,lags,frequency,startdate,enddate,Fstartdate,Fenddate,Fcenddate,Fendsmpl,F,CF);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Beginning of rolling forecasting loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

stringdatesforecast=stringdates2;
startdateini=startdate;
data_endo_full=data_endo;
if VARtype==3
    data_endo_a_full=data_endo_a; 
    Dmatrix_full=Dmatrix; % Matrix for the regimes for the mean-adjusted BVARs
end
numt=1;% initialisation
Fstartdate_rolling={};%to keep track of iterations
if window_size>length(stringdates1)
   msgbox('Forecasting window size greater than sample size');
   error('Forecasting window size greater than sample size');
elseif window_size>0 
numt = length(stringdates1)-window_size+lags; % number of different dateroll dates
end   

for iteration=1:numt % beginning of forecasting loop

if window_size>0  
 data_endo = data_endo_full(iteration:window_size+iteration,:);
 if VARtype==3
    Dmatrix = Dmatrix_full(iteration:window_size+iteration,:);
    data_endo_a = data_endo_a_full(iteration:window_size+iteration,:);
 end
 %if size(data_exo)>0 %need to fix
 %data_exo = data_exo_full(iteration:window_size+iteration,:);
 %end
Fstartlocation1=find(strcmp(names(1:end,1),startdateini))+iteration-1;
startdate=char(names(Fstartlocation1,1));
Fendlocation=find(strcmp(names(1:end,1),startdateini))+window_size+iteration-1; 
enddate=char(names(Fendlocation,1));
if F>0;
Fstartdate=char(stringdatesforecast(find(strcmp(stringdatesforecast(1:end,1),enddate))+1,1));
Fenddate=char(stringdatesforecast(find(strcmp(stringdatesforecast(1:end,1),enddate))+hstep,1));
end

% if Fendlocation+hstep<=length(names)
%       Fcperiods=hstep;
%       % record the end date of the common periods
%       Fcenddate=char(stringdatesforecast(find(strcmp(stringdatesforecast(1:end,1),enddate))+hstep,1));
%       % if the forecast period ends later than the data set, the common periods end at the end of the data set
% else
%       Fcperiods=Fendlocation-Fstartlocation+1;
%       % record the end date of the common periods
%       Fcenddate=names{end,1};
% end

% generate the different sets of data
% if the model is the OLS VAR, the Bayesian VAR, the mean-adjusted BVAR, or the stochastic volatility BVAR:
if VARtype==1 || VARtype==2 || VARtype==3 || VARtype==5
[names data data_endo data_endo_a data_endo_c data_endo_c_lags data_exo data_exo_a data_exo_p data_exo_c data_exo_c_lags Fperiods Fcomp Fcperiods Fcenddate]...
=gensample(startdate,enddate,VARtype,Fstartdate,Fenddate,Fendsmpl,endo,exo,frequency,lags,F,CF,pref);
% else, if the model is the panel BVAR
elseif VARtype==4
[names data data_endo data_endo_a data_endo_c data_endo_c_lags data_exo data_exo_a data_exo_p data_exo_c data_exo_c_lags Fperiods Fcomp Fcperiods Fcenddate]...
=gensamplepan(startdate,enddate,Units,panel,Fstartdate,Fenddate,Fendsmpl,endo,exo,frequency,lags,F,CF,pref);
end


% generate the strings and decimal vectors of dates
[decimaldates1,decimaldates2,stringdates1,stringdates2,stringdates3,Fstartlocation,Fendlocation]=gendates(names,lags,frequency,startdate,enddate,Fstartdate,Fenddate,Fcenddate,Fendsmpl,F,CF);
  
end %window_size>0














%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% MAIN CODE (NOT TO BE CHANGED)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


















%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Grand loop 1: OLS VAR model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% if the selected model is an OLS/maximum likelihood  VAR, run this part
if VARtype==1

    
% model estimation
[Bhat betahat sigmahat X Xbar Y y EPS eps n m p T k q]=olsvar(data_endo,data_exo,const,lags);
% compute interval estimates 
[beta_median beta_std beta_lbound beta_ubound sigma_median]=olsestimates(betahat,sigmahat,X,k,q,cband);
% display the VAR results
olsvardisp(beta_median,beta_std,beta_lbound,beta_ubound,sigma_median,X,Y,n,m,p,k,q,T,IRFt,const,endo,exo,startdate,enddate,stringdates1,decimaldates1,pref);
% compute and display the steady state results
olsss(Y,X,n,m,p,Bhat,stringdates1,decimaldates1,endo,pref);


% Irf (if activated)
   if IRF==1
   [irf_estimates,D,gamma,D_estimates,gamma_estimates]=olsirf(betahat,sigmahat,IRFperiods,IRFt,Y,X,n,m,p,k,q,endo,decimaldates1,stringdates1,pref,IRFband);
   irfdisp(n,endo,IRFperiods,IRFt,irf_estimates,D_estimates,gamma_estimates,pref,[]);
   end
   
   
% forecasts (if activated)
   if F==1  
   [forecast_estimates]=olsforecast(data_endo_a,data_exo_p,Fperiods,betahat,Bhat,sigmahat,n,m,p,k,const,Fband);
   fdisp(Y,n,T,endo,stringdates2,decimaldates2,Fstartlocation,Fendlocation,forecast_estimates,pref);
   % forecast evaluation (if activated)
      if Feval==1 
      olsfeval(data_endo_c,stringdates3,Fstartdate,Fcenddate,Fcperiods,Fcomp,n,forecast_estimates,names,endo,pref);
      end
   end
   
   
% FEVD (if activated)
   if FEVD==1
   [fevd_estimates]=olsfevd(irf_estimates,IRFperiods,gamma,n,endo,pref);
   end
   
   
% historical decomposition (if activated)
   if HD==1
   [hd_estimates]=olshdecomp(betahat,Bhat,sigmahat,Y,X,n,m,p,T,k,IRFt,endo,stringdates1,decimaldates1,pref);
   end

   
% here finishes grand loop 1
% if the model selected is not an OLS VAR, this part will not be run
end


















%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Grand loop 2: BVAR model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% this is the part of the code that will be run if the selected VAR model is a BVAR
if VARtype==2

%% BLOCK 1: OLS ESTIMATES

% preliminary OLS VAR and univariate AR estimates
[Bhat betahat sigmahat, X Xbar Y y EPS eps n m p T k q]=olsvar(data_endo,data_exo,const,lags);
[arvar]=arloop(data_endo,const,p,n);

%% BLOCK 2: PRIOR EXTENSIONS

% if hyperparameter optimisation has been selected, run the grid search 
if hogs==1
    % grid for the Minnesota
    if prior==11||prior==12||prior==13
    [ar lambda1 lambda2 lambda3 lambda4 lambda6 lambda7]=mgridsearch(X,Y,y,n,m,p,k,q,T,lambda5,lambda6,lambda7,grid,arvar,sigmahat,data_endo,data_exo,prior,hogs,bex,blockexo,const,scoeff,iobs,pref,It,Bu);
    % grid for the normal- Wishart
    elseif prior==21||prior==22
    [ar lambda1 lambda3 lambda4 lambda6 lambda7]=nwgridsearch(X,Y,n,m,p,k,q,T,lambda2,lambda5,lambda6,lambda7,grid,arvar,data_endo,data_exo,prior,hogs,bex,const,scoeff,iobs,pref,It,Bu);
    end
    % update record of results on Excel
    excelrecord1
end

% implement any dummy observation extensions that may have been selected
[Ystar ystar Xstar Tstar Ydum ydum Xdum Tdum]=gendummy(data_endo,data_exo,Y,X,n,m,p,T,const,lambda6,lambda7,lambda8,scoeff,iobs,lrp,H);




%% BLOCK 3: POSTERIOR DERIVATION

% estimation of BVAR if a Minnesota prior has been chosen (i.e., prior has been set to 11,12 or 13)
if prior==11||prior==12||prior==13
   % set prior values
   [beta0 omega0 sigma]=mprior(ar,arvar,sigmahat,lambda1,lambda2,lambda3,lambda4,lambda5,n,m,p,k,q,prior,bex,blockexo);
   % obtain posterior distribution parameters
   [betabar omegabar]=mpost(beta0,omega0,sigma,Xstar,ystar,q,n);
   % run Gibbs sampling for the Minnesota prior
   [beta_gibbs sigma_gibbs]=mgibbs(It,Bu,betabar,omegabar,sigma,q);
   % compute posterior estimates 
   [beta_median beta_std beta_lbound beta_ubound sigma_median]=mestimates(betabar,omegabar,sigma,q,cband);
   

% estimation of BVAR if a normal-Wishart prior has been chosen (i.e., prior has been set to 21 or 22)
elseif prior==21||prior==22
   % set prior values
   [B0 beta0 phi0 S0 alpha0]=nwprior(ar,arvar,lambda1,lambda3,lambda4,n,m,p,k,q,prior);
   % obtain posterior distribution parameters
   [Bbar betabar phibar Sbar alphabar alphatilde]=nwpost(B0,phi0,S0,alpha0,Xstar,Ystar,n,Tstar,k);
   % run Gibbs sampling for the normal-Wishart prior
   [beta_gibbs sigma_gibbs]=nwgibbs(It,Bu,Bbar,phibar,Sbar,alphabar,alphatilde,n,k);
   % compute posterior estimates 
   [beta_median B_median beta_std beta_lbound beta_ubound sigma_median]=nwestimates(betabar,phibar,Sbar,alphabar,alphatilde,n,k,cband);
   
   
% estimation of BVAR if an independent normal-Wishart prior has been chosen (i.e., prior has been set to 31 or 32)
elseif prior==31||prior==32
   % set prior values
   [beta0 omega0 S0 alpha0]=inwprior(ar,arvar,lambda1,lambda2,lambda3,lambda4,lambda5,n,m,p,k,q,prior,bex,blockexo);
   % run Gibbs sampling for the mixed prior
   [beta_gibbs sigma_gibbs]=inwgibbs(It,Bu,beta0,omega0,S0,alpha0,Xstar,Ystar,ystar,Bhat,n,Tstar,q);
   % compute posterior estimates
   [beta_median beta_std beta_lbound beta_ubound sigma_median]=inwestimates(beta_gibbs,sigma_gibbs,cband,q,n,k);


% estimation of BVAR if a normal-diffuse prior has been chosen (i.e., prior has been set to 41 or 42)
elseif prior==41
   % set prior values
   [beta0 omega0]=ndprior(ar,arvar,lambda1,lambda2,lambda3,lambda4,lambda5,n,m,p,k,q,bex,blockexo);
   % run Gibbs sampling for the normal-diffuse prior
   [beta_gibbs sigma_gibbs]=ndgibbs(It,Bu,beta0,omega0,Xstar,Ystar,ystar,Bhat,n,Tstar,q);
   % compute posterior estimates
   [beta_median beta_std beta_lbound beta_ubound sigma_median]=ndestimates(beta_gibbs,sigma_gibbs,cband,q,n,k);
   
   
% estimation of BVAR if a dummy observation prior has been chosen (i.e., prior has been set to 51, 52 or 53)
elseif prior==51
   % set 'prior' values (here, the dummy observations)
   [Ystar Xstar Tstar]=doprior(Ystar,Xstar,n,m,p,Tstar,ar,arvar,lambda1,lambda3,lambda4);
   % obtain posterior distribution parameters
   [Bcap betacap Scap alphacap phicap alphatop]=dopost(Xstar,Ystar,Tstar,k,n);
   % run Gibbs sampling for the dummy observation prior
   [beta_gibbs sigma_gibbs]=dogibbs(It,Bu,Bcap,phicap,Scap,alphacap,alphatop,n,k);
   % compute posterior estimates 
   [beta_median B_median beta_std beta_lbound beta_ubound sigma_median]=doestimates(betacap,phicap,Scap,alphacap,alphatop,n,k,cband);
end




%% BLOCK 4: MODEL EVALUATION

% compute the marginal likelihood for the model
if prior==11||prior==12||prior==13
[logml log10ml ml]=mmlik(Xstar,Xdum,ystar,ydum,n,Tstar,Tdum,q,sigma,beta0,omega0,betabar,scoeff,iobs);
elseif prior==21||prior==22
[logml log10ml ml]=nwmlik(Xstar,Xdum,Ydum,n,Tstar,Tdum,k,B0,phi0,S0,alpha0,Sbar,alphabar,scoeff,iobs);
elseif prior==31||prior==32
[logml log10ml ml]=inwmlik(Y,X,n,k,q,T,beta0,omega0,S0,alpha0,beta_median,sigma_median,beta_gibbs,It,Bu,scoeff,iobs);
elseif prior==41||prior==51
log10ml=nan;
end

%compute the DIC test
if prior==11||prior==12||prior==13||prior==21||prior==22|| prior==31||prior==32||prior==41||prior==51
[dic]=dic_test(Y,X,n,T,lags,beta_gibbs,sigma_gibbs,It-Bu );
%[dic]=0;
end

% display the VAR results
bvardisp(beta_median,beta_std,beta_lbound,beta_ubound,sigma_median,log10ml,dic,X,Y,n,m,p,k,q,T,prior,bex,hogs,lrp,H,ar,lambda1,lambda2,lambda3,lambda4,lambda5,lambda6,lambda7,lambda8,IRFt,const,beta_gibbs,endo,exo,startdate,enddate,stringdates1,decimaldates1,pref,scoeff,iobs);

% compute and display the steady state results
[ss_record]=ssgibbs(n,m,p,k,X,beta_gibbs,It,Bu);
[ss_estimates]=ssestimates(ss_record,n,T,cband);
ssdisp(Y,n,endo,stringdates1,decimaldates1,ss_estimates,pref);





%% BLOCK 5: IRFs

% compute IRFs if the option has been retained

if IRF==1
% run the Gibbs sampler to obtain posterior draws
[irf_record]=irf(beta_gibbs,It,Bu,IRFperiods,n,m,p,k);

   % If IRFs have been set to an unrestricted VAR (IRFt=1):
   if IRFt==1
   % run a pseudo Gibbs sampler to obtain records for D and gamma (for the trivial SVAR)
   [D_record gamma_record]=irfunres(n,It,Bu,sigma_gibbs);
   % compute posterior estimates
   [irf_estimates,D_estimates,gamma_estimates]=irfestimates(irf_record,n,IRFperiods,IRFband,IRFt,[],[]);
   % display the results
   irfdisp(n,endo,IRFperiods,IRFt,irf_estimates,[],[],pref,[]);
   
   % If IRFs have been set to an SVAR with Choleski identification (IRFt=2):
   elseif IRFt==2
   % run the Gibbs sampler to transform unrestricted draws into orthogonalised draws
   [struct_irf_record D_record gamma_record]=irfchol(sigma_gibbs,irf_record,It,Bu,IRFperiods,n);
   % compute posterior estimates
   [irfchol_estimates,D_estimates,gamma_estimates]=irfestimates(struct_irf_record,n,IRFperiods,IRFband,IRFt,D_record,gamma_record);
   % display the results
   irfdisp(n,endo,IRFperiods,IRFt,irfchol_estimates,D_estimates,gamma_estimates,pref,[]);
   
   % If IRFs have been set to an SVAR with triangular factorisation (IRFt=3):
   elseif IRFt==3
   % run the Gibbs sampler to transform unrestricted draws into orthogonalised draws
   [struct_irf_record D_record gamma_record]=irftrig(sigma_gibbs,irf_record,It,Bu,IRFperiods,n);
   % compute posterior estimates
   [irftrig_estimates,D_estimates,gamma_estimates]=irfestimates(struct_irf_record,n,IRFperiods,IRFband,IRFt,D_record,gamma_record);
   % display the results
   irfdisp(n,endo,IRFperiods,IRFt,irftrig_estimates,D_estimates,gamma_estimates,pref,[]);
   
   % If IRFs have been set to an SVAR with sign restrictions (IRFt=4):
   elseif IRFt==4
   % run the Gibbs sampler to transform unrestricted draws into orthogonalised draws
   [struct_irf_record D_record gamma_record]=irfres(beta_gibbs,sigma_gibbs,It,Bu,IRFperiods,n,m,p,k,signrestable,signresperiods);
   % compute posterior estimates
   [irfres_estimates,D_estimates,gamma_estimates]=irfestimates(struct_irf_record,n,IRFperiods,IRFband,IRFt,D_record,gamma_record);
   % display the results
   irfdisp(n,endo,IRFperiods,IRFt,irfres_estimates,D_estimates,gamma_estimates,pref,signreslabels);
   end
   
   % If an SVAR was selected, also compute the structural shock series
   if IRFt==2||IRFt==3||IRFt==4
   % compute first the empirical posterior distribution of the structural shocks
   [strshocks_record]=strshocks(beta_gibbs,D_record,Y,X,n,k,It,Bu);
   % compute posterior estimates
   [strshocks_estimates]=strsestimates(strshocks_record,n,T,IRFband);
   % display the results
   strsdisp(n,decimaldates1,strshocks_estimates,endo,stringdates1,pref,IRFt,signreslabels);
   end

end


% estimate IRFs for exogenous variables
[exo_irf_record exo_irf_estimates]=irfexo(beta_gibbs,It,Bu,IRFperiods,IRFband,n,m,p,k);
% estimate IRFs for exogenous variables
irfexodisp(n,m,endo,exo,IRFperiods,exo_irf_estimates,pref);


%% BLOCK 6: FORECASTS

% compute forecasts if the option has been retained
if F==1
% run the Gibbs sampler to obtain draws form the posterior predictive distribution
[forecast_record]=forecast(data_endo_a,data_exo_p,It,Bu,beta_gibbs,sigma_gibbs,Fperiods,n,p,k,const);
% compute posterior estimates
[forecast_estimates]=festimates(forecast_record,n,Fperiods,Fband);
% display the results for the forecasts
fdisp(Y,n,T,endo,stringdates2,decimaldates2,Fstartlocation,Fendlocation,forecast_estimates,pref);
% finally, compute forecast evaluation if the option was selected
   if Feval==1 
       %OLS single variable with BIC lag selection VAR for Rossi test
      [OLS_Bhat, OLS_betahat, OLS_sigmahat, OLS_forecast_estimates, biclag]=arbicloop(data_endo,data_endo_a,const,p,n,m,Fperiods,Fband);
      [Forecasteval]=bvarfeval(data_endo_c,data_endo_c_lags,data_exo_c,stringdates3,Fstartdate,Fcenddate,Fcperiods,Fcomp,const,n,p,k,It,Bu,beta_gibbs,sigma_gibbs,forecast_record,forecast_estimates,names,endo,pref);
   end
end






%% BLOCK 7: FEVD

% compute FEVD if the option has been retained
if FEVD==1
% run the Gibbs sampler to compute posterior draws
[fevd_record]=fevd(struct_irf_record,gamma_record,It,Bu,IRFperiods,n);
% compute posterior estimates
[fevd_estimates]=fevdestimates(fevd_record,n,IRFperiods,FEVDband);
% display the results
fevddisp(n,endo,IRFperiods,fevd_estimates,pref,IRFt,signreslabels);
end




%% BLOCK 8: historical decomposition

% compute historical decomposition if the option has been retained
if HD==1
% run the Gibbs sampler to compute posterior draws 
[hd_record]=hdecomp(beta_gibbs,D_record,strshocks_record,It,Bu,Y,X,n,m,p,k,T);
% compute posterior estimates
[hd_estimates]=hdestimates(hd_record,n,T,HDband);
% display the results
hddisp(n,endo,Y,decimaldates1,hd_estimates,stringdates1,T,pref,IRFt,signreslabels);
end





%% BLOCK 9: conditional forecasts

% compute conditional forecasts if the option has been retained
if CF==1
   % if the type of conditional forecasts corresponds to the standard methodology
   if CFt==1||CFt==2
   % run the Gibbs sampler to obtain draws from the posterior predictive distribution of conditional forecasts
   [cforecast_record]=cforecast(data_endo_a,data_exo_a,data_exo_p,It,Bu,Fperiods,cfconds,cfshocks,cfblocks,CFt,const,beta_gibbs,D_record,gamma_record,n,m,p,k,q);
   % if the type of conditional forecasts corresponds to the tilting methodology
   elseif CFt==3||CFt==4
   [cforecast_record]=tcforecast(forecast_record,Fperiods,cfconds,cfintervals,CFt,n,Fband,It,Bu);
   end
   % compute posterior estimates
   [cforecast_estimates]=festimates(cforecast_record,n,Fperiods,Fband);
   % display the results for the forecasts
   cfdisp(Y,n,T,endo,stringdates2,decimaldates2,Fstartlocation,Fendlocation,cforecast_estimates,pref);
end

if numt>1
    save([pref.datapath '\results\' pref.results_sub Fstartdate '.mat']); % Save Workspace
end

Fstartdate_rolling=[Fstartdate_rolling; Fstartdate];

% here finishes grand loop 2
% if the model selected is not a BVAR, this part will not be run
end












%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Grand loop 3: mean-adjusted BVAR model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% if the selected model is a mean-adjusted VAR, run this part
if VARtype==3


% obtain the preliminary elements for the model
[Y X Z n m p T k1 k3 q1 q2 q3]=TVEmaprelim(data_endo,data_exo,const,lags,regimeperiods,names);
[arvar]=arloop(data_endo,const,p,n);
% set prior distribution parameters for the model
[beta0, omega0, psi0, lambda0, r]=maprior(ar,arvar,lambda1,lambda2,lambda3,lambda4,lambda5,n,m,p,k1,q1,q2,bex,blockexo,Fpconfint,Fpconfint2,chvar,regimeperiods,Dmatrix,equilibrium,data_endo,priorf);
% Create H matrix
[TVEH, TVEHfuture]=TVEcreateH(equilibrium,r,T,p,Fperiods);
% check the priors
checkpriors(psi0,lambda0,TVEH,decimaldates1,data_endo,Dmatrix);
q2=length(psi0);


% run Gibbs sampler for estimation
[beta_gibbs, sigma_gibbs, theta_gibbs, ss_record,indH]=TVEmagibbs(data_endo,It,Bu,beta0,omega0,psi0,lambda0,Y,X,n,T,k1,q1,p,regimeperiods,names,TVEH);
%[beta_gibbs psi_gibbs sigma_gibbs delta_gibbs ss_record]=magibbs(data_endo,data_exo,It,Bu,beta0,omega0,psi0,lambda0,Y,X,Z,n,m,T,k1,k3,q1,q2,q3,p);
% compute posterior estimates
[beta_median, beta_std, beta_lbound, beta_ubound, theta_median, theta_std, theta_lbound, theta_ubound, sigma_median]=TVEmaestimates(beta_gibbs,theta_gibbs,sigma_gibbs,cband,q1,q2,n);
%[beta_median beta_std beta_lbound beta_ubound psi_median psi_std psi_lbound psi_ubound sigma_median]=maestimates(beta_gibbs,psi_gibbs,sigma_gibbs,cband,q1,q2,n);


% display the VAR results and save them in txt file
%mavardisp(beta_median,beta_std,beta_lbound,beta_ubound,psi_median,psi_std,psi_lbound,psi_ubound,sigma_median,X,Z,Y,n,m,p,k1,k3,q1,q2,T,ar,lambda1,lambda2,lambda3,lambda4,lambda5,IRFt,beta_gibbs,endo,exo,psi_gibbs,startdate,enddate,stringdates1,decimaldates1,pref)
TVEmavardisp(beta_median,beta_std,beta_lbound,beta_ubound,theta_median,theta_std,theta_lbound,theta_ubound,sigma_median,X,Z,Y,n,m,p,k1,k3,q1,q2,T,ar,lambda1,lambda2,lambda3,lambda4,lambda5,IRFt,beta_gibbs,endo,exo,theta_gibbs,startdate,enddate,stringdates1,decimaldates1,pref.datapath,data_endo,TVEH,indH,pref)
% compute and display the steady state results
[ss_estimates]=ssestimates(ss_record,n,T,cband);
ssdisp(Y,n,endo,stringdates1,decimaldates1,ss_estimates,pref);


 % IRFs (if activated)
if IRF==1
   [irf_record]=mairf(beta_gibbs,It,Bu,IRFperiods,p,n,k1);
      % If IRFs have been set to an unrestricted VAR (IRFt=1):
      if IRFt==1
      % run a pseudo Gibbs sampler to obtain records for D and gamma (for the trivial SVAR)
      [D_record gamma_record]=irfunres(n,It,Bu,sigma_gibbs);
      % compute posterior estimates
      [irf_estimates,D_estimates,gamma_estimates]=irfestimates(irf_record,n,IRFperiods,IRFband,IRFt,[],[]);
      % display the results
      irfdisp(n,endo,IRFperiods,IRFt,irf_estimates,[],[],pref,[]);
      
      % If IRFs have been set to an SVAR with Choleski identification (IRFt=2):
      elseif IRFt==2
      % run the Gibbs sampler to transform unrestricted draws into orthogonalised draws
      [struct_irf_record D_record gamma_record]=irfchol(sigma_gibbs,irf_record,It,Bu,IRFperiods,n);
      % compute posterior estimates
      [irfchol_estimates,D_estimates,gamma_estimates]=irfestimates(struct_irf_record,n,IRFperiods,IRFband,IRFt,D_record,gamma_record);
      % display the results
      irfdisp(n,endo,IRFperiods,IRFt,irfchol_estimates,D_estimates,gamma_estimates,pref,[]);
      
      % If IRFs have been set to an SVAR with triangular factorisation (IRFt=3):
      elseif IRFt==3
      % run the Gibbs sampler to transform unrestricted draws into orthogonalised draws
      [struct_irf_record D_record gamma_record]=irftrig(sigma_gibbs,irf_record,It,Bu,IRFperiods,n);
      % compute posterior estimates
      [irftrig_estimates,D_estimates,gamma_estimates]=irfestimates(struct_irf_record,n,IRFperiods,IRFband,IRFt,D_record,gamma_record);
      % display the results
      irfdisp(n,endo,IRFperiods,IRFt,irftrig_estimates,D_estimates,gamma_estimates,pref,[]);
      
      % If IRFs have been set to an SVAR with sign restrictions (IRFt=4):
      elseif IRFt==4
      % run the Gibbs sampler to transform unrestricted draws into orthogonalised draws
      [struct_irf_record D_record gamma_record]=mairfres(beta_gibbs,sigma_gibbs,It,Bu,IRFperiods,n,p,k1,signrestable,signresperiods);
      % compute posterior estimates
      [irfres_estimates,D_estimates,gamma_estimates]=irfestimates(struct_irf_record,n,IRFperiods,IRFband,IRFt,D_record,gamma_record);
      % display the results
      irfdisp(n,endo,IRFperiods,IRFt,irfres_estimates,D_estimates,gamma_estimates,pref,signreslabels);
      end

      % If an SVAR was selected, also compute the structural shock series
      if IRFt==2||IRFt==3||IRFt==4
      % compute first the empirical posterior distribution of the structural shocks
      %[strshocks_record]=mastrshocks(beta_gibbs,delta_gibbs,D_record,Y,X,Z,n,k1,k3,It,Bu);
      [strshocks_record]=TVEmastrshocks(beta_gibbs,theta_gibbs,D_record,n,k1,It,Bu,TVEH,indH,data_endo,p);
      % compute posterior estimates
      [strshocks_estimates]=strsestimates(strshocks_record,n,T,IRFband);
      % display the results
      strsdisp(n,decimaldates1,strshocks_estimates,endo,stringdates1,pref,IRFt,signreslabels);
      end
end
   
   
% forecasts (if activated)
   if F==1
   [forecast_record]=TVEmaforecast(data_endo_a,data_exo_a,data_exo_p,It,Bu,beta_gibbs,sigma_gibbs,Fperiods,n,m,p,k1,k3,theta_gibbs,TVEHfuture,ss_record,indH);   %[forecast_record]=maforecast(data_endo_a,data_exo_a,data_exo_p,It,Bu,beta_gibbs,sigma_gibbs,delta_gibbs,Fperiods,n,m,p,k1,k3);
   [forecast_estimates]=festimates(forecast_record,n,Fperiods,Fband);
   fdisp(Y,n,T,endo,stringdates2,decimaldates2,Fstartlocation,Fendlocation,forecast_estimates,pref);
   % finally, compute forecast evaluation if the option was selected
      if Feval==1 
      % OLS single variable with BIC lag selection VAR for Rossi test
      [OLS_Bhat, OLS_betahat, OLS_sigmahat, OLS_forecast_estimates, biclag]=arbicloop(data_endo,data_endo_a,const,p,n,1,Fperiods,Fband); % m=1, otherwise the code crashes 
      [Forecasteval]=TVEmafeval(data_endo_a,data_endo_c,data_endo_c_lags,data_exo_c,data_exo_c_lags,stringdates3,Fstartdate,Fcenddate,Fcperiods,Fcomp,const,n,m,p,k1,k3,It,Bu,beta_gibbs,sigma_gibbs,forecast_record,forecast_estimates,names,endo,pref,theta_gibbs,TVEHfuture,ss_record,indH);
      end
   end
   

% FEVD (if activated)
   if FEVD==1
      % run the Gibbs sampler to compute posterior draws
      [fevd_record]=fevd(struct_irf_record,gamma_record,It,Bu,IRFperiods,n);
      % compute posterior estimates
      [fevd_estimates]=fevdestimates(fevd_record,n,IRFperiods,FEVDband);
      % display the results
      fevddisp(n,endo,IRFperiods,fevd_estimates,pref,IRFt,signreslabels);
   end
   
   
% historical decomposition (if activated)
   if HD==1
   % run the Gibbs sampler to compute posterior draws  
   [hd_record]=TVEmahdecomp(beta_gibbs,D_record,strshocks_record,It,Bu,Y,n,p,k1,T);
   % compute posterior estimates
   [hd_estimates]=hdestimates(hd_record,n,T,HDband);
   % display the results
   %hddisp(n,endo,decimaldates1,hd_estimates,stringdates1,T,pref,IRFt,signreslabels);
   hddisp(n,endo,Y,decimaldates1,hd_estimates,stringdates1,T,pref,IRFt,signreslabels)
   end
   
   
% Conditional forecasts (if activated)
if CF==1
   % if the type of conditional forecasts corresponds to the standard methodology
   if CFt==1||CFt==2
   % run the Gibbs sampler to obtain draws form the posterior predictive distribution     
   [cforecast_record]=TVEmacforecast(data_endo_a,data_exo_a,data_exo_p,It,Bu,Fperiods,cfconds,cfshocks,cfblocks,CFt,n,m,p,k1,k3,beta_gibbs,D_record,gamma_record,theta_gibbs,TVEHfuture,ss_record,indH);
   % if the type of conditional forecasts corresponds to the tilting methodology
   elseif CFt==3||CFt==4
   [cforecast_record]=tcforecast(forecast_record,Fperiods,cfconds,cfintervals,CFt,n,Fband,It,Bu);
   end   
   % compute posterior estimates
   [cforecast_estimates]=festimates(cforecast_record,n,Fperiods,Fband);
   % display the results for the forecasts
   cfdisp(Y,n,T,endo,stringdates2,decimaldates2,Fstartlocation,Fendlocation,cforecast_estimates,pref);
end
   
if numt>1
    save([pref.datapath '\results\' pref.results_sub Fstartdate '.mat']); % Save Workspace
end

Fstartdate_rolling=[Fstartdate_rolling; Fstartdate];   
% here finishes grand loop 3
% if the model selected is not a mean-adjusted VAR, this part will not be run
end












%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Grand loop 4: panel VAR model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if the selected model is a panel VAR, run this part
if VARtype==4

    
%% BLOCK 1: MODEL ESTIMATION  

   % if the model is the OLS mean group estimator
   if panel==1
   % compute preliminary elements
   [X Y N n m p T k q]=panel1prelim(data_endo,data_exo,const,lags);
   % obtain the estimates for the model
   [bhat sigmahatb sigmahat]=panel1estimates(X,Y,N,n,q,k,T);
   % plot a first set of results
   panel1plot(endo,Units,X,Y,N,n,m,p,k,T,bhat,decimaldates1,stringdates1,pref);
   
   % else, if the model is the Bayesian pooled estimator
   elseif panel==2
   % compute preliminary elements
   [X Xmat Y Ymat N n m p T k q]=panel2prelim(data_endo,data_exo,const,lags,Units);
   % obtain prior elements (from a standard normal-Wishart)
   [B0 beta0 phi0 S0 alpha0]=panel2prior(N,n,m,p,T,k,q,data_endo,ar,lambda1,lambda3,lambda4);
   % obtain posterior distribution parameters
   [Bbar betabar phibar Sbar alphabar alphatilde]=nwpost(B0,phi0,S0,alpha0,X,Y,n,N*T,k);
   % run the Gibbs sampler
   [beta_gibbs sigma_gibbs]=nwgibbs(It,Bu,Bbar,phibar,Sbar,alphabar,alphatilde,n,k);
   % compute posterior estimates 
   [beta_median B_median beta_std beta_lbound beta_ubound sigma_median]=nwestimates(betabar,phibar,Sbar,alphabar,alphatilde,n,k,cband);
   % plot a first set of results
   panel2plot(endo,Units,Xmat,Ymat,N,n,m,p,k,T,beta_median,beta_gibbs,It,Bu,decimaldates1,stringdates1,pref,cband);
   
   % else, if the model is the random effect model (Zellner and Hong)
   elseif panel==3
   % compute preliminary elements
   [Xi Xibar Xbar Yi yi y N n m p T k q h]=panel3prelim(data_endo,data_exo,const,lags);
   % obtain prior elements
   [b bbar sigeps]=panel3prior(Xibar,Xbar,yi,y,N,q);
   % compute posterior distribution parameters
   [omegabarb betabar]=panel3post(h,Xbar,y,lambda1,bbar,sigeps);
   % run the Gibbs sampler
   [beta_gibbs sigma_gibbs]=panel3gibbs(It,Bu,betabar,omegabarb,sigeps,h,N,n,q);
   % compute posterior estimates
   [beta_median beta_std beta_lbound beta_ubound sigma_median]=panel3estimates(N,n,q,betabar,omegabarb,sigeps,cband);
   % plot a first set of results
   panel3plot(endo,Units,Xi,Yi,N,n,m,p,k,T,beta_median,beta_gibbs,It,Bu,decimaldates1,stringdates1,pref,cband);
   
   % else, if the model is the random effect model (hierarchical)
   elseif panel==4
   % compute preliminary elements
   [Xi Xibar Xbar Yi yi y N n m p T k q h]=panel4prelim(data_endo,data_exo,const,lags);
   % obtain prior elements
   [omegab]=panel4prior(N,n,m,p,T,k,data_endo,q,lambda3,lambda2,lambda4);
   % run the Gibbs sampler
   [beta_gibbs sigma_gibbs]=panel4gibbs(N,n,h,T,k,q,Yi,Xi,s0,omegab,v0,It,Bu,pick,pickf);
   % compute posterior estimates
   [beta_median beta_std beta_lbound beta_ubound sigma_median]=panel4estimates(N,n,q,beta_gibbs,sigma_gibbs,cband);
   % plot a first set of results
   panel4plot(endo,Units,Xi,Yi,N,n,m,p,k,T,beta_median,beta_gibbs,It,Bu,decimaldates1,stringdates1,pref,cband);
   
   % else, if the model is the factor model (static)
   elseif panel==5  
   % compute preliminary elements
   [Ymat Xmat N n m p T k q h]=panel5prelim(data_endo,data_exo,const,lags);
   % obtain prior elements
   [d1 d2 d3 d4 d5 d Xi1 Xi2 Xi3 Xi4 Xi5 Xi Y y Xtilde Xdot theta0 Theta0]=panel5prior(N,n,p,m,k,q,h,T,Ymat,Xmat);
   % run the Gibbs sampler
   [theta_gibbs sigma_gibbs sigmatilde_gibbs sig_gibbs]=panel5gibbs(y,Y,Xtilde,Xdot,N,n,T,d,theta0,Theta0,alpha0,delta0,It,Bu,pick,pickf);
   % compute posterior estimates
   [theta_median theta_std theta_lbound theta_ubound sigma_median]=panel5estimates(d,N,n,theta_gibbs,sigma_gibbs,cband);
   % plot a first set of results 
   panel5plot(endo,Units,Xmat,Xdot,Ymat,N,n,m,p,k,T,theta_median,theta_gibbs,Xi,It,Bu,decimaldates1,stringdates1,pref,cband)

   
   % else, if the model is the factor model (dynamic)
   elseif panel==6   
   % compute preliminary elements
   [Ymat Xmat N n m p T k q h]=panel6prelim(data_endo,data_exo,const,lags);
   % obtain prior elements
   [d1 d2 d3 d4 d5 d Xi1 Xi2 Xi3 Xi4 Xi5 Xi y Xtilde thetabar theta0 H Thetatilde Theta0 G]=panel6prior(N,n,p,m,k,q,h,T,Ymat,Xmat,rho,gama);
   % run the Gibbs sampler
   [theta_gibbs sigmatilde_gibbs Zeta_gibbs sigma_gibbs phi_gibbs B_gibbs acceptrate]=panel6gibbs(y,Xtilde,N,n,T,theta0,Theta0,thetabar,alpha0,delta0,a0,b0,psi,d1,d2,d3,d4,d5,d,It,Bu,H,G,pick,pickf,gama);
   % compute posterior estimates
   [theta_median theta_std theta_lbound theta_ubound sigma_median]=panel6estimates(d,N,n,T,theta_gibbs,sigma_gibbs,cband);
   % plot a first set of results
   panel6plot(endo,Units,Xmat,Xtilde,Ymat,N,n,m,p,k,T,d,theta_median,theta_gibbs,Xi,Zeta_gibbs,It,Bu,decimaldates1,stringdates1,pref,cband,d1,d2,d3,d4,d5);
   end
   
   
%% BLOCK 2: IRFS

   % impulse response functions (if activated)
   if IRF==1
       
      % if the model is the OLS mean group estimator
      if panel==1
      % estimate the IRFs
      [irf_estimates,D,gamma,D_estimates,gamma_estimates,strshocks_estimates]=panel1irf(Y,X,N,n,m,p,k,q,IRFt,bhat,sigmahatb,sigmahat,IRFperiods,IRFband);
      % display the results
      panel1irfdisp(N,n,Units,endo,irf_estimates,strshocks_estimates,IRFperiods,IRFt,stringdates1,T,decimaldates1,pref);
      
      % else, if the model is the Bayesian pooled estimator
      elseif panel==2
      % estimate the IRFs
      [irf_record D_record gamma_record struct_irf_record irf_estimates D_estimates gamma_estimates strshocks_record strshocks_estimates]=panel2irf(Ymat,Xmat,beta_gibbs,sigma_gibbs,It,Bu,IRFperiods,IRFband,N,n,m,p,k,T,IRFt,signrestable,signresperiods);
      % display the results
      panel2irfdisp(N,n,Units,endo,irf_estimates,strshocks_estimates,IRFperiods,IRFt,stringdates1,T,decimaldates1,pref);
      
      % else, if the model is the random effect model (Zellner and Hong)
      elseif panel==3  
      % estimate the IRFs
      [irf_record D_record gamma_record struct_irf_record irf_estimates D_estimates gamma_estimates strshocks_record strshocks_estimates]=panel3irf(Yi,Xi,beta_gibbs,sigma_gibbs,It,Bu,IRFperiods,IRFband,N,n,m,p,k,T,IRFt,signrestable,signresperiods);
      % display the results
      panel3irfdisp(N,n,Units,endo,irf_estimates,strshocks_estimates,IRFperiods,IRFt,stringdates1,T,decimaldates1,pref);

      % else, if the model is the random effect model (hierarchical)
      elseif panel==4
      % estimate the IRFs
      [irf_record D_record gamma_record struct_irf_record irf_estimates D_estimates gamma_estimates strshocks_record strshocks_estimates]=panel4irf(Yi,Xi,beta_gibbs,sigma_gibbs,It,Bu,IRFperiods,IRFband,N,n,m,p,k,T,IRFt,signrestable,signresperiods);
      % display the results
      panel4irfdisp(N,n,Units,endo,irf_estimates,strshocks_estimates,IRFperiods,IRFt,stringdates1,T,decimaldates1,pref);

      % else, if the model is the factor model (static)
      elseif panel==5
      % estimate the IRFs      
      [irf_record D_record gamma_record struct_irf_record irf_estimates D_estimates gamma_estimates strshocks_record strshocks_estimates]=panel5irf(Y,Xdot,theta_gibbs,sigma_gibbs,Xi,It,Bu,IRFperiods,IRFband,N,n,m,p,k,T,IRFt);
      % display the results
      panel5irfdisp(N,n,Units,endo,irf_estimates,strshocks_estimates,IRFperiods,IRFt,stringdates1,T,decimaldates1,pref);

      % else, if the model is the factor model (dynamic)
      elseif panel==6   
      % estimate the IRFs      
      [irf_record D_record gamma_record struct_irf_record irf_estimates D_estimates gamma_estimates strshocks_record strshocks_estimates]=panel6irf(y,Xtilde,theta_gibbs,sigma_gibbs,B_gibbs,Xi,It,Bu,IRFperiods,IRFband,IRFt,rho,thetabar,N,n,m,p,T,d);
      % display the results
      panel6irfdisp(N,n,Units,endo,irf_estimates,strshocks_estimates,IRFperiods,IRFt,stringdates1,T,decimaldates1,pref);
      end      
      
   end

   
   
   
%% BLOCK 3: FORECASTS

   % forecasts (if activated)
   if F==1
       
      % if the model is the OLS mean group estimator
      if panel==1
      % estimate the forecasts
      [forecast_estimates]=panel1forecast(sigmahat,bhat,k,n,const,data_exo_p,Fperiods,N,data_endo_a,p,T,m,Fband);
      % display the results
      panel1fdisp(N,n,T,Units,endo,Y,stringdates2,decimaldates2,Fstartlocation,Fendlocation,forecast_estimates,pref);

      % else, if the model is the Bayesian pooled estimator
      elseif panel==2
      % estimate the forecasts
      [forecast_record forecast_estimates]=panel2forecast(N,n,p,k,data_endo_a,data_exo_p,It,Bu,beta_gibbs,sigma_gibbs,Fperiods,const,Fband);
      % display the results
      panel2fdisp(N,n,T,Units,endo,Ymat,stringdates2,decimaldates2,Fstartlocation,Fendlocation,forecast_estimates,pref);

      % else, if the model is the random effect model (Zellner and Hong)
      elseif panel==3 
      % estimate the forecasts
      [forecast_record forecast_estimates]=panel3forecast(N,n,p,k,data_endo_a,data_exo_p,It,Bu,beta_gibbs,sigma_gibbs,Fperiods,const,Fband);
      % display the results
      panel3fdisp(N,n,T,Units,endo,Yi,stringdates2,decimaldates2,Fstartlocation,Fendlocation,forecast_estimates,pref);

      % else, if the model is the random effect model (hierarchical)
      elseif panel==4
      % estimate the forecasts
      [forecast_record forecast_estimates]=panel4forecast(N,n,p,k,data_endo_a,data_exo_p,It,Bu,beta_gibbs,sigma_gibbs,Fperiods,const,Fband);
      % display the results
      panel4fdisp(N,n,T,Units,endo,Yi,stringdates2,decimaldates2,Fstartlocation,Fendlocation,forecast_estimates,pref);

      % else, if the model is the factor model (static)
      elseif panel==5    
      % estimate the forecasts     
      [forecast_record forecast_estimates]=panel5forecast(N,n,p,data_endo_a,data_exo_p,It,Bu,theta_gibbs,sigma_gibbs,Xi,Fperiods,const,Fband);
      % display the results
      panel5fdisp(N,n,T,Units,endo,Ymat,stringdates2,decimaldates2,Fstartlocation,Fendlocation,forecast_estimates,pref);

      % else, if the model is the factor model (dynamic)
      elseif panel==6  
      % estimate the forecasts
      [forecast_record forecast_estimates]=panel6forecast(const,data_exo_p,Fstartlocation,It,Bu,data_endo_a,p,B_gibbs,sigmatilde_gibbs,N,n,phi_gibbs,theta_gibbs,Zeta_gibbs,Fperiods,d,rho,thetabar,gama,Xi,Fband);
      % display the results
      panel6fdisp(N,n,T,Units,endo,Ymat,stringdates2,decimaldates2,Fstartlocation,Fendlocation,forecast_estimates,pref)
      end  
 
   end
   
   
   
%% BLOCK 4: FEVD
   
   % FEVD (if activated)
   if FEVD==1
       
      % if the model is the OLS mean group estimator
      if panel==1
      % estimate FEVD and display the results
      [fevd_estimates]=panel1fevd(N,n,irf_estimates,IRFperiods,gamma,Units,endo,pref);
      
      % else, if the model is the Bayesian pooled estimator
      elseif panel==2
      % estimate the FEVD
      [fevd_record fevd_estimates]=panel2fevd(struct_irf_record,gamma_record,It,Bu,IRFperiods,n,FEVDband);
      % display the results
      panel2fevddisp(n,endo,fevd_estimates,IRFperiods,pref);
      
      % else, if the model is the random effect model (Zellner and Hong)
      elseif panel==3 
      % estimate the FEVD  
      [fevd_record fevd_estimates]=panel3fevd(N,struct_irf_record,gamma_record,It,Bu,IRFperiods,n,FEVDband);
      % display the results
      panel3fevddisp(n,N,Units,endo,fevd_estimates,IRFperiods,pref);
      
      % else, if the model is the random effect model (hierarchical)
      elseif panel==4     
      % estimate the FEVD      
      [fevd_record fevd_estimates]=panel4fevd(N,struct_irf_record,gamma_record,It,Bu,IRFperiods,n,FEVDband);
      % display the results
      panel4fevddisp(n,N,Units,endo,fevd_estimates,IRFperiods,pref);
      
      % else, if the model is the factor model (static)
      elseif panel==5        
      % estimate the FEVD      
      [fevd_record fevd_estimates]=panel5fevd(N,n,struct_irf_record,gamma_record,It,Bu,IRFperiods,FEVDband);    
      % display the results   
      panel5fevddisp(n,N,Units,endo,fevd_estimates,IRFperiods,pref);

      % else, if the model is the factor model (dynamic)
      elseif panel==6  
      % estimate the FEVD
      [fevd_record fevd_estimates]=panel6fevd(N,n,T,struct_irf_record,gamma_record,It,Bu,IRFperiods,FEVDband);
      % display the results  
      panel6fevddisp(n,N,Units,endo,fevd_estimates,IRFperiods,pref);
      end
      
   end

   
   
   
%% BLOCK 5: HISTORICAL DECOMPOSITION
   
   % historical decomposition (if activated)
   if HD==1
       
      % if the model is the OLS mean group estimator
      if panel==1
      % estimate historical decomposition and display the results
      [hd_estimates]=panel1hd(Y,X,N,n,m,p,T,k,D,bhat,endo,Units,decimaldates1,stringdates1,pref);   
      
      % else, if the model is the Bayesian pooled estimator
      elseif panel==2
      % estimate historical decomposition
      [hd_record hd_estimates]=panel2hd(beta_gibbs,D_record,strshocks_record,It,Bu,Ymat,Xmat,N,n,m,p,k,T,HDband);
      % display the results
      panel2hddisp(N,n,T,Units,endo,hd_estimates,stringdates1,decimaldates1,pref);
      
      % else, if the model is the random effect model (Zellner and Hong)     
      elseif panel==3
      % estimate historical decomposition   
      [hd_record hd_estimates]=panel3hd(beta_gibbs,D_record,strshocks_record,It,Bu,Yi,Xi,N,n,m,p,k,T,HDband);
      % display the results
      panel3hddisp(N,n,T,Units,endo,hd_estimates,stringdates1,decimaldates1,pref);    
      
      % else, if the model is the random effect model (hierarchical)      
      elseif panel==4  
      % estimate historical decomposition   
      [hd_record hd_estimates]=panel4hd(beta_gibbs,D_record,strshocks_record,It,Bu,Yi,Xi,N,n,m,p,k,T,HDband);  
      % display the results
      panel4hddisp(N,n,T,Units,endo,hd_estimates,stringdates1,decimaldates1,pref);
      
      % else, if the model is the factor model (static)
      elseif panel==5        
      % estimate historical decomposition   
      [hd_record hd_estimates]=panel5hd(Xi,theta_gibbs,D_record,strshocks_record,It,Bu,Ymat,Xmat,N,n,m,p,k,T,HDband);
      % display the results
      panel5hddisp(N,n,T,Units,endo,hd_estimates,stringdates1,decimaldates1,pref);
      
      % else, if the model is the factor model (dynamic)
      elseif panel==6  
      % estimate historical decomposition      
      [hd_record hd_estimates]=panel6hd(Xi,theta_gibbs,D_record,strshocks_record,It,Bu,Ymat,N,n,m,p,k,T,d,HDband);
      % display the results
      panel6hddisp(N,n,T,Units,endo,hd_estimates,stringdates1,decimaldates1,pref);
      end  
           
   end
   
   
      
%% BLOCK 6: CONDITIONAL FORECASTS
   
   % conditional forecast (if activated)
   if CF==1
   
      % if the model is the Bayesian pooled estimator  
      if panel==2
      % estimate conditional forecasts
      [nconds cforecast_record cforecast_estimates]=panel2cf(N,n,m,p,k,q,cfconds,cfshocks,cfblocks,data_endo_a,data_exo_a,data_exo_p,It,Bu,Fperiods,const,beta_gibbs,D_record,gamma_record,CFt,Fband);
      % display the results
      panel2cfdisp(N,n,T,Units,endo,Ymat,stringdates2,decimaldates2,Fstartlocation,Fendlocation,cforecast_estimates,pref,nconds);
      
      % else, if the model is the random effect model (Zellner and Hong)    
      elseif panel==3
      % estimate conditional forecasts
      [nconds cforecast_record cforecast_estimates]=panel3cf(N,n,m,p,k,q,cfconds,cfshocks,cfblocks,data_endo_a,data_exo_a,data_exo_p,It,Bu,Fperiods,const,beta_gibbs,D_record,gamma_record,CFt,Fband);
      % display the results
      panel3cfdisp(N,n,T,Units,endo,Yi,stringdates2,decimaldates2,Fstartlocation,Fendlocation,cforecast_estimates,pref,nconds);
      
      % else, if the model is the random effect model (hierarchical)
      elseif panel==4      
      % estimate conditional forecasts     
      [nconds cforecast_record cforecast_estimates]=panel4cf(N,n,m,p,k,q,cfconds,cfshocks,cfblocks,data_endo_a,data_exo_a,data_exo_p,It,Bu,Fperiods,const,beta_gibbs,D_record,gamma_record,CFt,Fband);
      % display the results
      panel4cfdisp(N,n,T,Units,endo,Yi,stringdates2,decimaldates2,Fstartlocation,Fendlocation,cforecast_estimates,pref,nconds);
      
      % else, if the model is the factor model (static)
      elseif panel==5        
      % estimate conditional forecasts
      [cforecast_record cforecast_estimates]=panel5cf(N,n,m,p,k,q,cfconds,cfshocks,cfblocks,data_endo_a,data_exo_a,data_exo_p,It,Bu,Fperiods,const,Xi,theta_gibbs,D_record,gamma_record,CFt,Fband);
      % display the results
      panel5cfdisp(N,n,T,Units,endo,Ymat,stringdates2,decimaldates2,Fstartlocation,Fendlocation,cforecast_estimates,pref);
      
      % else, if the model is the factor model (dynamic)
      elseif panel==6  
      % estimate conditional forecasts
      [cforecast_record cforecast_estimates]=panel6cf(N,n,m,p,k,d,cfconds,cfshocks,cfblocks,It,Bu,Fperiods,const,Xi,data_exo_p,theta_gibbs,B_gibbs,phi_gibbs,Zeta_gibbs,sigmatilde_gibbs,Fstartlocation,Ymat,rho,thetabar,gama,CFt,Fband);
      % display the results
      panel6cfdisp(N,n,T,Units,endo,Ymat,stringdates2,decimaldates2,Fstartlocation,Fendlocation,cforecast_estimates,pref);
      end
      
   end
       
       
       
       
       
      
%% BLOCK 7: DISPLAY OF THE RESULTS


   % if the model is the OLS mean group estimator
   if panel==1
   panel1disp(X,Y,n,N,m,p,T,k,q,const,bhat,sigmahat,sigmahatb,Units,endo,exo,gamma_estimates,D_estimates,startdate,...
   enddate,Fstartdate,Fcenddate,Fcperiods,Feval,Fcomp,data_endo_c,forecast_estimates,stringdates3,cband,pref,IRF,IRFt,names);

   % else, if the model is the Bayesian pooled estimator
   elseif panel==2
   panel2disp(n,N,m,p,k,T,Ymat,Xmat,Units,endo,exo,const,beta_gibbs,B_median,beta_median,beta_std,beta_lbound,beta_ubound,sigma_gibbs,...
   sigma_median,D_estimates,gamma_estimates,ar,lambda1,lambda3,lambda4,startdate,enddate,forecast_record,forecast_estimates,Fcperiods,...
   stringdates3,Fstartdate,Fcenddate,Feval,Fcomp,data_endo_c,data_endo_c_lags,data_exo_c,It,Bu,IRF,IRFt,pref,names); 

   % else, if the model is the random effect model (Zellner and Hong)    
   elseif panel==3
   panel3disp(n,N,m,p,k,T,Yi,Xi,Units,endo,exo,const,beta_gibbs,beta_median,beta_std,beta_lbound,beta_ubound,sigma_gibbs,...
   sigma_median,D_estimates,gamma_estimates,lambda1,startdate,enddate,forecast_record,forecast_estimates,Fcperiods,stringdates3,...
   Fstartdate,Fcenddate,Feval,Fcomp,data_endo_c,data_endo_c_lags,data_exo_c,It,Bu,IRF,IRFt,pref,names);

   % else, if the model is the random effect model (hierarchical)
   elseif panel==4   
   panel4disp(n,N,m,p,k,T,Yi,Xi,Units,endo,exo,const,beta_gibbs,beta_median,beta_std,beta_lbound,beta_ubound,sigma_gibbs,...
   sigma_median,D_estimates,gamma_estimates,lambda2,lambda3,lambda4,s0,v0,startdate,enddate,forecast_record,forecast_estimates,...
   Fcperiods,stringdates3,Fstartdate,Fcenddate,Feval,Fcomp,data_endo_c,data_endo_c_lags,data_exo_c,It,Bu,IRF,IRFt,pref,names);

   % else, if the model is the factor model (static)
   elseif panel==5 
   panel5disp(n,N,m,p,k,T,d1,d2,d3,d4,d5,Ymat,Xdot,Units,endo,exo,const,Xi,theta_gibbs,theta_median,theta_std,theta_lbound,theta_ubound,sigma_gibbs,...
   sigma_median,D_estimates,gamma_estimates,alpha0,delta0,startdate,enddate,forecast_record,forecast_estimates,Fcperiods,...
   stringdates3,Fstartdate,Fcenddate,Feval,Fcomp,data_endo_c,data_endo_c_lags,data_exo_c,It,Bu,IRF,IRFt,pref,names);
      
   % else, if the model is the factor model (dynamic)
   elseif panel==6  
   panel6disp(n,N,m,p,k,T,d1,d2,d3,d4,d5,d,Ymat,Xtilde,Units,endo,exo,const,Xi,theta_median,theta_std,theta_lbound,theta_ubound,sigma_median,...
   D_estimates,gamma_estimates,alpha0,delta0,gama,a0,b0,rho,psi,acceptrate,startdate,enddate,forecast_record,forecast_estimates,Fcperiods,...
   stringdates3,Fstartdate,Fcenddate,Feval,Fcomp,data_endo_c,IRF,IRFt,pref,names);

   end
   
end
    







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Grand loop 5: Stochastic volatility BVAR model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if the selected model is the stochastic volatility BVAR, run this part
if VARtype==5



%% BLOCK 1: OLS ESTIMATES AND PRELIMINARY ELEMENTS

% preliminary OLS VAR and univariate AR estimates
[Bhat betahat sigmahat X Xbar Y y EPS eps n m p T k q]=olsvar(data_endo,data_exo,const,lags);
[arvar]=arloop(data_endo,const,p,n);
[yt Xt Xbart]=stvoltmat(Y,X,n,T);



%% BLOCK 2: POSTERIOR DERIVATION

% if the model is the standard model
if stvol==1
   % obtain prior elements
   [beta0 omega0 G I_o omega f0 upsilon0]=stvol1prior(ar,arvar,lambda1,lambda2,lambda3,lambda4,lambda5,n,m,p,T,k,q,bex,blockexo,gamma);
   % run the Gibbs sampling algorithm to recover the posterior distributions
   [beta_gibbs F_gibbs L_gibbs phi_gibbs sigma_gibbs lambda_t_gibbs sigma_t_gibbs sbar]=stvol1gibbs(Xbart,yt,beta0,omega0,alpha0,delta0,f0,upsilon0,betahat,sigmahat,gamma,G,I_o,omega,T,n,q,It,Bu,pick,pickf);
   % compute posterior estimates
   [beta_median beta_std beta_lbound beta_ubound sigma_median sigma_t_median sigma_t_lbound sigma_t_ubound]=stvol1estimates(beta_gibbs,sigma_gibbs,sigma_t_gibbs,n,T,cband);

   
% if the model is the random inertia model
elseif stvol==2
   % obtain prior elements
   [beta0 omega0 I_o omega f0 upsilon0]=stvol2prior(ar,arvar,lambda1,lambda2,lambda3,lambda4,lambda5,n,m,p,T,k,q,bex,blockexo);
   % run the Gibbs sampling algorithm to recover the posterior distributions
   [beta_gibbs F_gibbs gamma_gibbs L_gibbs phi_gibbs sigma_gibbs lambda_t_gibbs sigma_t_gibbs sbar]=stvol2gibbs(Xbart,yt,beta0,omega0,alpha0,delta0,gamma0,zeta0,f0,upsilon0,betahat,sigmahat,I_o,omega,T,n,q,It,Bu,pick,pickf);
   % compute posterior estimates
   [beta_median beta_std beta_lbound beta_ubound sigma_median sigma_t_median sigma_t_lbound sigma_t_ubound gamma_median]=stvol2estimates(beta_gibbs,sigma_gibbs,sigma_t_gibbs,gamma_gibbs,n,T,cband);

   
% if the model is the stochastic volatility model for large BVARs
elseif stvol==3
   % obtain prior elements   
   [B0 phi0 G I_o omega f0 upsilon0]=stvol3prior(ar,arvar,lambda1,lambda3,lambda4,n,m,p,T,k,q,gamma);
   % run the Gibbs sampling algorithm to recover the posterior distributions
   [beta_gibbs F_gibbs L_gibbs phi_gibbs sigma_gibbs lambda_t_gibbs sigma_t_gibbs sbar]=stvol3gibbs(Xbart,Xt,yt,B0,phi0,alpha0,delta0,f0,upsilon0,betahat,sigmahat,gamma,G,I_o,omega,T,n,k,It,Bu,pick,pickf);
   % compute posterior estimates
   [beta_median beta_std beta_lbound beta_ubound sigma_median sigma_t_median sigma_t_lbound sigma_t_ubound]=stvol3estimates(beta_gibbs,sigma_gibbs,sigma_t_gibbs,n,T,cband);

end   
   
   

%% BLOCK 3: MODEL EVALUATION
  
% display the VAR results
stvoldisp(beta_median,beta_std,beta_lbound,beta_ubound,sigma_median,sigma_t_median,sigma_t_lbound,sigma_t_ubound,gamma_median,X,Y,n,m,p,k,T,stvol,bex,ar,lambda1,lambda2,lambda3,lambda4,lambda5,gamma,alpha0,delta0,gamma0,zeta0,IRFt,const,endo,exo,startdate,enddate,stringdates1,decimaldates1,pref);

% compute and display the steady state results
[ss_record]=ssgibbs(n,m,p,k,X,beta_gibbs,It,Bu);
[ss_estimates]=ssestimates(ss_record,n,T,cband);
ssdisp(Y,n,endo,stringdates1,decimaldates1,ss_estimates,pref);



%% BLOCK 4: IRFs

% compute IRFs if the option has been retained

if IRF==1
% run the Gibbs sampler to obtain posterior draws
[irf_record]=irf(beta_gibbs,It,Bu,IRFperiods,n,m,p,k);

   % If IRFs have been set to an unrestricted VAR (IRFt=1):
   if IRFt==1
   % run a pseudo Gibbs sampler to obtain records for D and gamma (for the trivial SVAR)
   [D_record gamma_record]=irfunres(n,It,Bu,sigma_gibbs);
   % compute posterior estimates
   [irf_estimates,D_estimates,gamma_estimates]=irfestimates(irf_record,n,IRFperiods,IRFband,IRFt,[],[]);
   % display the results
   irfdisp(n,endo,IRFperiods,IRFt,irf_estimates,[],[],pref,[]);
   
   % If IRFs have been set to an SVAR with Choleski identification (IRFt=2):
   elseif IRFt==2
   % run the Gibbs sampler to transform unrestricted draws into orthogonalised draws
   [struct_irf_record D_record gamma_record]=irfcholstvol(F_gibbs,sbar,irf_record,It,Bu,IRFperiods,n);
   % compute posterior estimates
   [irfchol_estimates,D_estimates,gamma_estimates]=irfestimates(struct_irf_record,n,IRFperiods,IRFband,IRFt,D_record,gamma_record);
   % display the results
   irfdisp(n,endo,IRFperiods,IRFt,irfchol_estimates,D_estimates,gamma_estimates,pref,[]);
   
   % If IRFs have been set to an SVAR with triangular factorisation (IRFt=3):
   elseif IRFt==3
   % run the Gibbs sampler to transform unrestricted draws into orthogonalised draws
   [struct_irf_record D_record gamma_record]=irftrigstvol(F_gibbs,sbar,irf_record,It,Bu,IRFperiods,n);
   % compute posterior estimates
   [irftrig_estimates,D_estimates,gamma_estimates]=irfestimates(struct_irf_record,n,IRFperiods,IRFband,IRFt,D_record,gamma_record);
   % display the results
   irfdisp(n,endo,IRFperiods,IRFt,irftrig_estimates,D_estimates,gamma_estimates,pref,[]);
   
   % If IRFs have been set to an SVAR with sign restrictions (IRFt=4):
   elseif IRFt==4
   % run the Gibbs sampler to transform unrestricted draws into orthogonalised draws
   [struct_irf_record D_record gamma_record]=irfres(beta_gibbs,sigma_gibbs,It,Bu,IRFperiods,n,m,p,k,signrestable,signresperiods);
   % compute posterior estimates
   [irfres_estimates,D_estimates,gamma_estimates]=irfestimates(struct_irf_record,n,IRFperiods,IRFband,IRFt,D_record,gamma_record);
   % display the results
   irfdisp(n,endo,IRFperiods,IRFt,irfres_estimates,D_estimates,gamma_estimates,pref,signreslabels);
   end
   
   % If an SVAR was selected, also compute the structural shock series
   if IRFt==2||IRFt==3||IRFt==4
   % compute first the empirical posterior distribution of the structural shocks
   [strshocks_record]=strshocks(beta_gibbs,D_record,Y,X,n,k,It,Bu);
   % compute posterior estimates
   [strshocks_estimates]=strsestimates(strshocks_record,n,T,IRFband);
   % display the results
   strsdisp(n,decimaldates1,strshocks_estimates,endo,stringdates1,pref,IRFt,signreslabels);
   end

end




%% BLOCK 5: FORECASTS

% compute forecasts if the option has been retained
if F==1
% run the Gibbs sampler to obtain draws from the posterior predictive distribution
   % if the model is the standard model
   if stvol==1
   [forecast_record]=forecaststvol1(data_endo_a,data_exo_p,It,Bu,beta_gibbs,F_gibbs,phi_gibbs,L_gibbs,gamma,sbar,Fstartlocation,Fperiods,n,p,k,const);
   % if the model is the random inertia model
   elseif stvol==2
   [forecast_record]=forecaststvol2(data_endo_a,data_exo_p,It,Bu,beta_gibbs,F_gibbs,phi_gibbs,L_gibbs,gamma_gibbs,sbar,Fstartlocation,Fperiods,n,p,k,const);
   % if the model is the large BVAR model
   elseif stvol==3
   [forecast_record]=forecaststvol3(data_endo_a,data_exo_p,It,Bu,beta_gibbs,F_gibbs,phi_gibbs,L_gibbs,gamma,sbar,Fstartlocation,Fperiods,n,p,k,const);
   end
% compute posterior estimates
[forecast_estimates]=festimates(forecast_record,n,Fperiods,Fband);
% display the results for the forecasts
fdisp(Y,n,T,endo,stringdates2,decimaldates2,Fstartlocation,Fendlocation,forecast_estimates,pref);
% finally, compute forecast evaluation if the option was selected
   if Feval==1 
   [Forecasteval]=bvarfeval(data_endo_c,data_endo_c_lags,data_exo_c,stringdates3,Fstartdate,Fcenddate,Fcperiods,Fcomp,const,n,p,k,It,Bu,beta_gibbs,sigma_gibbs,forecast_record,forecast_estimates,names,endo,pref);
   end
   
   
end



%% BLOCK 6: FEVD

% compute FEVD if the option has been retained
if FEVD==1
% run the Gibbs sampler to compute posterior draws
[fevd_record]=fevd(struct_irf_record,gamma_record,It,Bu,IRFperiods,n);
% compute posterior estimates
[fevd_estimates]=fevdestimates(fevd_record,n,IRFperiods,FEVDband);
% display the results
fevddisp(n,endo,IRFperiods,fevd_estimates,pref,IRFt,signreslabels);
end



%% BLOCK 7: historical decomposition

% compute historical decomposition if the option has been retained
if HD==1
% run the Gibbs sampler to compute posterior draws 
[hd_record]=hdecomp(beta_gibbs,D_record,strshocks_record,It,Bu,Y,X,n,m,p,k,T);
% compute posterior estimates
[hd_estimates]=hdestimates(hd_record,n,T,HDband);
% display the results
hddisp(n,endo,Y,decimaldates1,hd_estimates,stringdates1,T,pref,IRFt,signreslabels);
end



%% BLOCK 8: conditional forecasts

% compute conditional forecasts if the option has been retained
if CF==1
   % if the type of conditional forecasts corresponds to the standard methodology
   if CFt==1||CFt==2
   % run the Gibbs sampler to obtain draws from the posterior predictive distribution of conditional forecasts
   [cforecast_record]=cforecast(data_endo_a,data_exo_a,data_exo_p,It,Bu,Fperiods,cfconds,cfshocks,cfblocks,CFt,const,beta_gibbs,D_record,gamma_record,n,m,p,k,q);
   % if the type of conditional forecasts corresponds to the tilting methodology
   elseif CFt==3||CFt==4
   [cforecast_record]=tcforecast(forecast_record,Fperiods,cfconds,cfintervals,CFt,n,Fband,It,Bu);
   end
   % compute posterior estimates
   [cforecast_estimates]=festimates(cforecast_record,n,Fperiods,Fband);
   % display the results for the forecasts
   cfdisp(Y,n,T,endo,stringdates2,decimaldates2,Fstartlocation,Fendlocation,cforecast_estimates,pref);
end



% here finishes grand loop 5
% if the model selected is not a stochastic volatility BVAR, this part will not be run
end










%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Grand loop 6: Time-varying BVAR model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if the selected model is the time-varying BVAR, run this part
if VARtype==6



%% BLOCK 1: OLS ESTIMATES AND PRELIMINARY ELEMENTS

% preliminary OLS VAR and univariate AR estimates
[Bhat betahat sigmahat X Xbar Y y EPS eps n m p T k q]=olsvar(data_endo,data_exo,const,lags);
[arvar]=arloop(data_endo,const,p,n);
[yt y Xt Xbart Xbar]=tvbvarmat(Y,X,n,q,T);


%% BLOCK 2: POSTERIOR DERIVATION

% if the model is the time-varying coefficients only
if tvbvar==1
   % obtain prior elements
   [chi psi kappa S H I_tau]=tvbvar1prior(arvar,n,q,T);
   % run the Gibbs sampling algorithm to recover the posterior distributions
   [beta_gibbs omega_gibbs sigma_gibbs]=tvbvar1gibbs(S,sigmahat,T,chi,psi,kappa,betahat,q,n,It,Bu,I_tau,H,Xbar,y);
   % compute posterior estimates
   [beta_t_median beta_t_std beta_t_lbound beta_t_ubound omega_median sigma_median sigma_t_median sigma_t_lbound sigma_t_ubound]=tvbvar1estimates(beta_gibbs,omega_gibbs,sigma_gibbs,q,T,cband);

   
% if the model is the general time-varying
elseif tvbvar==2
   % obtain prior elements
   [chi psi kappa S H I_tau G I_om f0 upsilon0]=tvbvar2prior(arvar,n,q,T,gamma);
   % run the Gibbs sampling algorithm to recover the posterior distributions
   [beta_gibbs omega_gibbs F_gibbs L_gibbs phi_gibbs sigma_gibbs lambda_t_gibbs sigma_t_gibbs sbar]=tvbvar2gibbs(G,sigmahat,T,chi,psi,kappa,betahat,q,n,It,Bu,I_tau,I_om,H,Xbar,y,alpha0,yt,Xbart,upsilon0,f0,delta0,gamma,pick,pickf);
   % compute posterior estimates
   [beta_t_median beta_t_std beta_t_lbound beta_t_ubound omega_median sigma_median sigma_t_median sigma_t_lbound sigma_t_ubound]=tvbvar2estimates(beta_gibbs,omega_gibbs,F_gibbs,L_gibbs,phi_gibbs,sigma_gibbs,lambda_t_gibbs,sigma_t_gibbs,n,q,T,cband);
end   
   
   


%% BLOCK 3: MODEL EVALUATION
  
% display the VAR results
tvbvardisp(beta_t_median,beta_t_std,beta_t_lbound,beta_t_ubound,sigma_median,sigma_t_lbound,sigma_t_median,sigma_t_ubound,Xbart,Y,yt,n,m,p,k,q,T,tvbvar,gamma,alpha0,IRFt,const,endo,exo,startdate,enddate,stringdates1,decimaldates1,pref)




%% BLOCK 4: IRFs

% compute IRFs if the option has been retained

if IRF==1
% run the Gibbs sampler to obtain posterior draws
[irf_record]=tvbvarirf(beta_gibbs,omega_gibbs,It,Bu,IRFperiods,n,m,p,k,q,T);

   % If IRFs have been set to an unrestricted VAR (IRFt=1):
   if IRFt==1
   % run a pseudo Gibbs sampler to obtain records for D and gamma (for the trivial SVAR)
   [D_record gamma_record]=irfunres(n,It,Bu,sigma_gibbs);
   % compute posterior estimates
   [irf_estimates,D_estimates,gamma_estimates]=irfestimates(irf_record,n,IRFperiods,IRFband,IRFt,[],[]);
   % display the results
   irfdisp(n,endo,IRFperiods,IRFt,irf_estimates,[],[],pref,[]);
   
   % If IRFs have been set to an SVAR with Choleski identification (IRFt=2):
   elseif IRFt==2
   % run the Gibbs sampler to transform unrestricted draws into orthogonalised draws
   [struct_irf_record D_record gamma_record]=irfchol(sigma_gibbs,irf_record,It,Bu,IRFperiods,n);
   % compute posterior estimates
   [irfchol_estimates,D_estimates,gamma_estimates]=irfestimates(struct_irf_record,n,IRFperiods,IRFband,IRFt,D_record,gamma_record);
   % display the results
   irfdisp(n,endo,IRFperiods,IRFt,irfchol_estimates,D_estimates,gamma_estimates,pref,[]);
   
   % If IRFs have been set to an SVAR with triangular factorisation (IRFt=3):
   elseif IRFt==3
   % run the Gibbs sampler to transform unrestricted draws into orthogonalised draws
   [struct_irf_record D_record gamma_record]=irftrig(sigma_gibbs,irf_record,It,Bu,IRFperiods,n);
   % compute posterior estimates
   [irftrig_estimates,D_estimates,gamma_estimates]=irfestimates(struct_irf_record,n,IRFperiods,IRFband,IRFt,D_record,gamma_record);
   % display the results
   irfdisp(n,endo,IRFperiods,IRFt,irftrig_estimates,D_estimates,gamma_estimates,pref,[]);
   
   % If IRFs have been set to an SVAR with sign restrictions (IRFt=4):
   elseif IRFt==4
   % run the Gibbs sampler to transform unrestricted draws into orthogonalised draws
   [struct_irf_record D_record gamma_record]=tvirfres(beta_gibbs,omega_gibbs,sigma_gibbs,It,Bu,IRFperiods,n,m,p,k,q,T,signrestable,signresperiods);
   % compute posterior estimates
   [irfres_estimates,D_estimates,gamma_estimates]=irfestimates(struct_irf_record,n,IRFperiods,IRFband,IRFt,D_record,gamma_record);
   % display the results
   irfdisp(n,endo,IRFperiods,IRFt,irfres_estimates,D_estimates,gamma_estimates,pref,signreslabels);
   end
   
   % If an SVAR was selected, also compute the structural shock series
   if IRFt==2||IRFt==3||IRFt==4
   % compute first the empirical posterior distribution of the structural shocks
   [strshocks_record]=tvstrshocks(beta_gibbs,D_record,y,Xbar,n,T,It,Bu);
   % compute posterior estimates
   [strshocks_estimates]=strsestimates(strshocks_record,n,T,IRFband);
   % display the results
   strsdisp(n,decimaldates1,strshocks_estimates,endo,stringdates1,pref,IRFt,signreslabels);
   end

% if the option IRFs for all period is selected
if alltirf==1
   % if no stochastic volatility
   if tvbvar==1
   % gibbs sampling
   [irf_record_allt]=tvbvarirf2(beta_gibbs,D_record,It,Bu,IRFperiods,n,m,p,k,T);
   % if stochastic volatility, and the model is not defined by sign restrictions
   elseif tvbvar==2 && IRFt~=4
   % recover the structural decomposition matrix for each period
   [irf_record_allt]=tvbvarirf3(beta_gibbs,sigma_t_gibbs,IRFt,It,Bu,IRFperiods,n,m,p,k,T);
   % if stochastic volatility, and the model is defined by sign restrictions
   elseif tvbvar==2 && IRFt==4
   % recover the structural decomposition matrix for each period  
   [irf_record_allt]=tvbvarirf4(beta_gibbs,sigma_t_gibbs,It,Bu,IRFperiods,n,m,p,k,T,signresperiods,signrestable);
   end
   % point estimates
[irf_estimates_allt]=irfestimates2(irf_record_allt,n,T,IRFperiods,IRFband,endo,stringdates1,pref);
% plot
irfdisp2(n,T,decimaldates1,endo,IRFperiods,IRFt,irf_estimates_allt,pref,signreslabels);
end
   
end




%% BLOCK 5: FORECASTS

% compute forecasts if the option has been retained
if F==1
% run the Gibbs sampler to obtain draws from the posterior predictive distribution
   % if the model is the VAR coefficients only model
   if tvbvar==1
   [forecast_record]=forecasttv1(data_endo_a,data_exo_p,It,Bu,beta_gibbs,omega_gibbs,sigma_gibbs,Fstartlocation,Fperiods,n,p,k,q,const);
   % if the model is the general model
   elseif tvbvar==2
   [forecast_record]=forecasttv2(data_endo_a,data_exo_p,It,Bu,beta_gibbs,omega_gibbs,F_gibbs,phi_gibbs,L_gibbs,gamma,sbar,Fstartlocation,Fperiods,n,p,k,q,const);
   end
% compute posterior estimates
[forecast_estimates]=festimates(forecast_record,n,Fperiods,Fband);
% display the results for the forecasts
fdisp(Y,n,T,endo,stringdates2,decimaldates2,Fstartlocation,Fendlocation,forecast_estimates,pref);
% finally, compute forecast evaluation if the option was selected
   if Feval==1
   [Forecasteval]=tvbvarfeval(data_endo_c,data_endo_c_lags,data_exo_c,stringdates3,Fstartdate,Fcenddate,Fcperiods,Fcomp,const,n,p,k,It,Bu,beta_gibbs,sigma_gibbs,forecast_record,forecast_estimates,names,endo,pref);
   end
end



%% BLOCK 6: FEVD

% compute FEVD if the option has been retained
if FEVD==1
% run the Gibbs sampler to compute posterior draws
[fevd_record]=fevd(struct_irf_record,gamma_record,It,Bu,IRFperiods,n);
% compute posterior estimates
[fevd_estimates]=fevdestimates(fevd_record,n,IRFperiods,FEVDband);
% display the results
fevddisp(n,endo,IRFperiods,fevd_estimates,pref,IRFt,signreslabels);
end



%% BLOCK 7: historical decomposition

% compute historical decomposition if the option has been retained
if HD==1
% run the Gibbs sampler to compute posterior draws 
[hd_record]=tvhdecomp(beta_gibbs,D_record,strshocks_record,It,Bu,Y,n,m,p,k,T);
% compute posterior estimates
[hd_estimates]=hdestimates(hd_record,n,T,HDband);
% display the results
hddisp(n,endo,Y,decimaldates1,hd_estimates,stringdates1,T,pref,IRFt,signreslabels);
end



%% BLOCK 8: conditional forecasts

% compute conditional forecasts if the option has been retained
if CF==1
   % if the type of conditional forecasts corresponds to the standard methodology
   if CFt==1||CFt==2
   % run the Gibbs sampler to obtain draws from the posterior predictive distribution of conditional forecasts
   [cforecast_record]=tvcforecast(n,m,p,k,q,cfconds,cfshocks,cfblocks,It,Bu,Fperiods,const,data_exo_p,beta_gibbs,omega_gibbs,sigma_gibbs,D_record,gamma_record,Fstartlocation,Y,CFt);
   % if the type of conditional forecasts corresponds to the tilting methodology
   elseif CFt==3||CFt==4
   [cforecast_record]=tcforecast(forecast_record,Fperiods,cfconds,cfintervals,CFt,n,Fband,It,Bu);
   end
   % compute posterior estimates
   [cforecast_estimates]=festimates(cforecast_record,n,Fperiods,Fband);
   % display the results for the forecasts
   cfdisp(Y,n,T,endo,stringdates2,decimaldates2,Fstartlocation,Fendlocation,cforecast_estimates,pref);
end




% here finishes grand loop 6
% if the model selected is not a time-varying BVAR, this part will not be run
end











% End of forecasting loop
end %iteration

% forecast evaluation
if numt>1
    rolling_evaluation;
% output: Rolling
end 

save([pref.datapath '\results\' pref.results_sub '.mat']); % Save Workspace








