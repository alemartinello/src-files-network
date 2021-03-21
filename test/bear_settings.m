%% BEAR 4.0
%  Using the BEAR toolbox implies acceptance of the End User Licence  %
%  Agreement and appropriate acknowledgement should be made.          %                                                        %

%%

% general data and model information
% VAR model selected (1=OLS VAR, 2=BVAR, 3=mean-adjusted BVAR, 4=panel Bayesian VAR, 5=Stochastic volatility BVAR, 6=Time varying)
VARtype=2;
% data frequency (1=yearly, 2= quarterly, 3=monthly, 4=weekly, 5=daily, 6=undated)
frequency=2;
% sample start date; must be a string consistent with the date formats of the toolbox
startdate='2000q1';
% sample end date; must be a string consistent with the date formats of the toolbox
enddate='2020q1';
% endogenous variables; must be a single string, with variable names separated by a space
varendo='lnfeu ep_all lnfierhv lnq_yy lnpcpdk_yy imm';
%varendo='lnfeu, dlnpraoli, lnpeu, int_un, lnfierhv, lnq_yy, dlnpcpdk, imm, dom_un';
%varendo='lnfeu lnpeu lnpraoli_yy lnfy lnq_yy lnlna lnpcpdk ib30';
% exogenous variables, if any; must be a single string, with variable names separated by a space
varexo='trend';
% number of lags
lags=1;
%lags=lagnumbers(1,ppp);
% inclusion of a constant (1=yes, 0=no)
const=1;
% path to data; must be a single string
pref.datapath='G:\Modelenheden\Big_dataproject_newpaperindex\VAR\Quarterly';
% excel results file name
pref.results_sub='Final_ny';%'BSVAR_bloom';
% matlab output file name
Name='Final_ny.mat';
% to output results in excel
pref.results=1;
% output charts
pref.plot=0;
% pref: useless by itself, just here to avoid code to crash
pref.pref=0;
disp('Bruges denne?');






% BVAR specific information: will be read only if VARtype=2

if VARtype==2
% selected prior
% 11=Minnesota (univariate AR), 12=Minnesota (diagonal VAR estimates), 13=Minnesota (full VAR estimates)
% 21=Normal-Wishart(S0 as univariate AR), 22=Normal-Wishart(S0 as identity)
% 31=Independent Normal-Wishart(S0 as univariate AR), 32=Independent Normal-Wishart(S0 as identity)
% 41=Normal-diffuse
% 51=Dummy observations
prior=31;
% hyperparameter: autoregressive coefficient
ar=0.8;

% hyperparameter: lambda1
%lambda1=priortests(kkk,1);
% hyperparameter: lambda2
%lambda2=priortests(kkk,2);
% hyperparameter: lambda3
%lambda3=priortests(kkk,3);
% hyperparameter: lambda4
%lambda4=priortests(kkk,4);
% hyperparameter: lambda5
%lambda5=priortests(kkk,5);

% hyperparameter: lambda1
lambda1=0.1;
% hyperparameter: lambda2
lambda2=1;
% hyperparameter: lambda3
lambda3=1;
% hyperparameter: lambda4
lambda4=100;
% hyperparameter: lambda5
lambda5=0.001;
% hyperparameter: lambda6
lambda6=1;
% hyperparameter: lambda7
lambda7=0.1;
% Overall tightness on the long run prior
lambda8=1;
% total number of iterations for the Gibbs sampler
It=10000;
% number of burn-in iterations for the Gibbs sampler
Bu=5000;
% hyperparameter optimisation by grid search (1=yes, 0=no)
hogs=0;
% block exogeneity (1=yes, 0=no)
bex=1;
% sum-of-coefficients application (1=yes, 0=no)
scoeff=0;
% dummy initial observation application (1=yes, 0=no)
iobs=0;
% Long run prior option
lrp=0;
% create H matrix for the long run priors 
% now taken from excel loadH.m
% H=[1 1 0 0;-1 1 0 0;0 0 1 1;0 0 -1 1];


% Mean-adjusted BVAR specific information: will be read only if VARtype=3

elseif VARtype==3
% hyperparameter: autoregressive coefficient
ar=0.5;
% hyperparameter: lambda1
lambda1=0.1;
% hyperparameter: lambda2
lambda2=0.5;
% hyperparameter: lambda3
lambda3=1;
% hyperparameter: lambda4
lambda4=100;
% hyperparameter: lambda5
lambda5=0.001;
% total number of iterations for the Gibbs sampler
It=2000;
% number of burn-in iterations for the Gibbs sampler
Bu=1000;
% block exogeneity (1=yes, 0=no)
bex=0;
% Scale up the variance of the prior of factor f
priorf=100;




% panel Bayesian VAR specific information: will be read only if VARtype=4

elseif VARtype==4
% choice of panel model 
% 1=OLS mean group estimator, 2=pooled estimator
% 3=random effect (Zellner and Hong), 4=random effect (hierarchical)
% 5=static factor approach, 6=dynamic factor approach
panel=2;
% units; must be single sstring, with names separated by a space
unitnames='US EA UK';
% total number of iterations for the Gibbs sampler
It=2000;
% number of burn-in iterations for the Gibbs sampler
Bu=1000;
% choice of retaining only one post burn iteration over 'pickf' iterations (1=yes, 0=no)
pick=0;
% frequency of iteration picking (e.g. pickf=20 implies that only 1 out of 20 iterations will be retained)
pickf=20;
% hyperparameter: autoregressive coefficient
ar=0.8;
% hyperparameter: lambda1
lambda1=0.1;
% hyperparameter: lambda2
lambda2=0.5;
% hyperparameter: lambda3
lambda3=1;
% hyperparameter: lambda4
lambda4=100;
% hyperparameter: s0
s0=0.001;
% hyperparameter: v0
v0=0.001;
% hyperparameter: alpha0
alpha0=1000;
% hyperparameter: delta0
delta0=1;
% hyperparameter: gama
gama=0.85;
% hyperparameter: a0
a0=1000;
% hyperparameter: b0
b0=1;
% hyperparameter: rho
rho=0.75;
% hyperparameter: psi
psi=0.1;





% Stochastic volatility BVAR information: will be read only if VARtype=5

elseif VARtype==5
% choice of stochastic volatility model 
% 1=standard, 2=random scaling, 3=large BVAR
stvol=1;
% total number of iterations for the Gibbs sampler
It=2000;
% number of burn-in iterations for the Gibbs sampler
Bu=1000;
% choice of retaining only one post burn iteration over 'pickf' iterations (1=yes, 0=no)
pick=0;
% frequency of iteration picking (e.g. pickf=20 implies that only 1 out of 20 iterations will be retained)
pickf=20;
% block exogeneity (1=yes, 0=no)
bex=0;
% hyperparameter: autoregressive coefficient
ar=0.8;
% hyperparameter: lambda1
lambda1=0.1;
% hyperparameter: lambda2
lambda2=0.5;
% hyperparameter: lambda3
lambda3=1;
% hyperparameter: lambda4
lambda4=100;
% hyperparameter: lambda5
lambda5=0.001;
% hyperparameter: gama
gamma=0.85;
% hyperparameter: alpha0
alpha0=0.001;
% hyperparameter: delta0
delta0=0.001;
% hyperparameter: gamma0
gamma0=0;
% hyperparameter: zeta0
zeta0=10000;




% Time-varying BVAR information: will be read only if VARtype=6

elseif VARtype==6
% choice of time-varying BVAR model 
% 1=time-varying coefficients, 2=general time-varying
tvbvar=1;
% total number of iterations for the Gibbs sampler
It=2000;
% number of burn-in iterations for the Gibbs sampler
Bu=1000;
% choice of retaining only one post burn iteration over 'pickf' iterations (1=yes, 0=no)
pick=0;
% frequency of iteration picking (e.g. pickf=20 implies that only 1 out of 20 iterations will be retained)
pickf=20;
% calculate IRFs for every sample period (1=yes, 0=no)
alltirf=0;
% hyperparameter: gama
gamma=0.85;
% hyperparameter: alpha0
alpha0=0.001;
% hyperparameter: delta0
delta0=0.001;




end





% Model options
% activate impulse response functions (1=yes, 0=no)
IRF=1;
% activate unconditional forecasts (1=yes, 0=no)
F=0;
% activate forecast error variance decomposition (1=yes, 0=no)
FEVD=0;
% activate historical decomposition (1=yes, 0=no)
HD=1;
% activate conditional forecasts (1=yes, 0=no)
CF=0;
% structural identification (1=none, 2=Choleski, 3=triangular factorisation, 4=sign restrictions)
IRFt=3;
% activate forecast evaluation (1=yes, 0=no)
Feval=0;
% type of conditional forecasts 
% 1=standard (all shocks), 2=standard (shock-specific)
% 3=tilting (median), 4=tilting (interval)
CFt=1;
% number of periods for impulse response functions
IRFperiods=40;
% start date for forecasts (has to be an in-sample date; otherwise, ignore and set Fendsmpl=1)
Fstartdate='2014q1';
% end date for forecasts
Fenddate='2016q4';
% start forecasts immediately after the final sample period (1=yes, 0=no)
% has to be set to 1 if start date for forecasts is not in-sample
Fendsmpl=0;
% step ahead evaluation
hstep=1;
% window_size for iterative forecasting 0 if no iterative forecasting
window_size=0; 
% evaluation_size as percent of window_size                                      <                                                                                    -
evaluation_size=0.5;                          
% confidence/credibility level for VAR coefficients
cband=0.95;
% confidence/credibility level for impusle response functions
IRFband=0.67;
% confidence/credibility level for forecasts
Fband=0.95;
% confidence/credibility level for forecast error variance decomposition
FEVDband=0.95;
% confidence/credibility level for historical decomposition
HDband=0.95;
