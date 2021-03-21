%% HISTORICAL DECOMPOSITION WITH CONSTANT PARAMETERS
addpath(genpath('files\'))

%% Estimate model to obtain parameter estimates
run files/bear_toolbox_main_code; %Runs basic model with settings as chosen in files/settings
%% Make new input
startdate= '2000q1'; %start of fitting period (no earlier than first period of data sample)
enddate = '2020q2'; %end of fitting period (might be later than end of estimation sample, but no later than end of data sample)
[decimaldates1,decimaldates2,stringdates1,stringdates2,stringdates3,Fstartlocation,Fendlocation]=gendates(names,lags,frequency,startdate,enddate,Fstartdate,Fenddate,Fcenddate,Fendsmpl,F,CF);

[names data data_endo data_endo_a data_endo_c data_endo_c_lags data_exo data_exo_a data_exo_p data_exo_c data_exo_c_lags Fperiods Fcomp Fcperiods Fcenddate]...
=gensample(startdate,enddate,VARtype,Fstartdate,Fenddate,Fendsmpl,endo,exo,frequency,lags,F,CF,pref);

[Bhat betahat sigmahat X Xbar Y y EPS eps n m p T k q]=olsvar(data_endo,data_exo,const,lags) %OLS regression, but also makes relevant datasample

%% Calculate new strucural shocks
% compute first the empirical posterior distribution of the structural shocks
   [strshocks_record2]=strshocks(beta_gibbs,D_record,Y,X,n,k,It,Bu);
   % compute posterior estimates
   [strshocks_estimates2]=strsestimates(strshocks_record2,n,T,IRFband);

%% Calculate new historical decomposition
% run the Gibbs sampler to compute posterior draws 
[hd_record2]=hdecomp(beta_gibbs,D_record,strshocks_record2,It,Bu,Y,X,n,m,p,k,T);
% compute posterior estimates
[hd_estimates2]=hdestimates(hd_record2,n,T,HDband);
% display the results
hddisp(n,endo,Y,decimaldates1,hd_estimates2,stringdates1,T,pref,IRFt,signreslabels);
