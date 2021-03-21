clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIGURE WITH MANY IRF'S
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print = 1; %Do you want to make a graph?
export = 1; %Do you want to export to Excel?
savee = 1; %Do you want to save your results in a .mat file?

%% 

respons = ["lnfierhv", "lnq_yy"] %JVariables of interest
stod = ["ep_all"] %Shock of interest

varname = {'Uncertainty, Investments','Employment', '', ''} %

% Specification of different models
varendo_base= 'lnfeu ep_all lnfierhv lnq_yy lnpcpdkeu_yy imm'; %Baseline
varendo_omx= 'lnfeu lnomx ep_all lnfierhv lnq_yy lnpcpdkeu_yy imm lnomx'; %Inclusive C25
varendo_inter = 'lnfeu lnpeu lnpraoli ep_all lnfierhv lnq_yy lnpcpdkeu_yy imm'; %Larger foreing block
varendo_eplast = 'lnfeu  lnfierhv lnq_yy lnpcpdkeu_yy imm ep_all'; %Uncertainty last
varendo_immfirst = 'lnfeu imm ep_all lnfierhv lnq_yy lnpcpdkeu_yy'; %Interest rate first
varendo_vix = 'lnfeu VIX ep_all lnfierhv lnq_yy lnpcpdkeu_yy imm'; %Inclusive VIX

endogen = ["base","omx", "inter", "eplast", "immfirst", "vix"] %Names

%Choose lag length
lag_base =1; %Baseline
lag_et =2 %2 lags
lag_to =3 %3 lags

lag = ["base","et","to"] %Names

%Trend
varexo_base = 'trend'; %Baseline
varexo_notrend = ''; %No trend

exogen = ["base", "notrend"] %Names

%% Combines

s = ["varendo_", "lag_", "varexo_"]


spec(1,:) = [strcat(s(1,1),endogen(1,1)), strcat(s(1,2),lag(1,1)) , strcat(s(1,3),exogen(1,1))] %Baseline
spec(2,:) = [strcat(s(1,1),endogen(1,2)), strcat(s(1,2),lag(1,1)) , strcat(s(1,3),exogen(1,1))] %Inclusive C25
spec(3,:) = [strcat(s(1,1),endogen(1,3)), strcat(s(1,2),lag(1,1)) , strcat(s(1,3),exogen(1,1))] %Larger foreing block
spec(4,:) = [strcat(s(1,1),endogen(1,4)), strcat(s(1,2),lag(1,1)) , strcat(s(1,3),exogen(1,1))] %Uncertianty last
spec(5,:) = [strcat(s(1,1),endogen(1,5)), strcat(s(1,2),lag(1,1)) , strcat(s(1,3),exogen(1,1))] %Interest rate first
spec(6,:) = [strcat(s(1,1),endogen(1,1)), strcat(s(1,2),lag(1,2)) , strcat(s(1,3),exogen(1,1))] %Model with two lags
spec(7,:) = [strcat(s(1,1),endogen(1,2)), strcat(s(1,2),lag(1,3)) , strcat(s(1,3),exogen(1,1))] %Model with three lags
spec(8,:) = [strcat(s(1,1),endogen(1,2)), strcat(s(1,2),lag(1,1)) , strcat(s(1,3),exogen(1,2))] %Model without trend
spec(9,:) = [strcat(s(1,1),endogen(1,6)), strcat(s(1,2),lag(1,1)) , strcat(s(1,3),exogen(1,1))] %Inclusive VIX

%% Run model

for kk=2:length(spec(:,1));
    run files/bear_toolbox_main_code;

    resp_index = zeros(1,length(respons));
    stod_index = zeros(1,length(stod));

    for jj = 1:length(respons)
        count_res = 1;
            for ii = 1:length(endo)
                if endo(ii,1)==respons(1,jj)
                    resp_index(1,jj)=count_res
                else count_res=count_res+1;
                end;
            end;
    end;

    for jj = 1:length(stod)
        count_stod = 1;
            for ii = 1:length(endo)
                if endo(ii,1)==stod(1,jj)
                    stod_index(1,jj)=count_stod
                else count_stod=count_stod+1;
                end;
            end;
    end;            
    
IRFoutput.resone.ep(kk,:)=irftrig_estimates{resp_index(1,1),stod_index(1,1)}(2,:) 
IRFoutput.restwo.ep(kk,:)=irftrig_estimates{resp_index(1,2),stod_index(1,1)}(2,:)
end;

%Saves output
if savee == 1;
save('output_IRFs.mat','IRFoutput');
end;

%Export output
if export == 1;
xlswrite('output_IRFs.xls',IRFoutput.resone.ep,'lnerhv_ep')
xlswrite('output_IRFs.xls',IRFoutput.restwo.ep,'lnq_ep')
end;

if print == 1;
    
objects=zeros(length(spec),IRFperiods,length(respons)*length(stod))
objects(:,:,1)=IRFoutput.resone.ep %Gemmer responsevariabel 1 til int stød
objects(:,:,2)=IRFoutput.restwo.ep %Gemmer responsevariabel 2 til int stød
         
    set(0,'defaulttextinterpreter','latex')
    fig = figure;
    fig.Units = 'inches';
    fig.PaperSize = fig.Position(3:4);
    subfig={};
        for ii=1:length(respons)*length(stod);
            subfig{ii}=subplot(2,2,ii);
            subfig{ii}.YLabel.Interpreter='latex';
            subfig{ii}.XLabel.Interpreter='latex';
            hold on
            plotdata=zeros(length(spec),IRFperiods);
            plotdata(1:length(spec),:)=objects(:,:,ii)*100
            ax=plot(1:IRFperiods,plotdata);             
            %ax(1).LineStyle = '--';
            ax(1).LineWidth = 1.2;      
            ax(1).Color = [0.9290 0.6940 0.1250];
            %ax(2).LineStyle = '--';
            ax(2).LineWidth = 1.2;
            ax(2).Color = 'g';
            ax(3).LineWidth = 1.0;
            ax(3).Color = 'b';
            ax(4).LineWidth = 1.0;
            ax(4).Color = 'r';
            ax(5).LineWidth = 0.5;
            ax(5).Color = 'k';
            %ax(5).LineStyle = ':';
            ax(6).Color = 'y'
            ax(7).LineStyle = ':';
            ax(7).Color = 'b'
            ax(8).LineStyle = ':';
            ax(8).Color = 'r'
            ax(9).LineStyle = ':';
            ax(9).Color = 'b'
            title(varname{ii})
            hold off       
    
            minband=min(min(plotdata));
            maxband=max(max(plotdata));
            space=maxband-minband;
            absolut = max(abs(minband),abs(maxband));
            Ymin=minband-0.2*space;
            Ymax=maxband+0.2*space;
            
            subfig{ii}.YLim = [Ymin Ymax];
end;


subfig{1}.YLabel.String = 'Investment response, %';
%subfig{1}.YLabel.FontSize = 9;
%subfig{2}.YLabel.String = 'Employment response, %';
%subfig{2}.YLabel.FontSize = 9;
subfig{3}.YLabel.String = 'Employment response, %';
%subfig{3}.YLabel.FontSize = 9;
%subfig{4}.YLabel.String = '';
%subfig{4}.YLabel.FontSize = 9;


%suptitle('Impulse responses to liquidity shock');
h=legend({'Baseline','Including C25','Larger international block','Uncertainty last', 'Interest rate first', 'Two lag', 'Three lags','No trend'});
set(h,'Orientation','horizontal');
set(h,'Position',[0.1174    0.0018    0.7945    0.0450]);
%set(h,'Position',[0.1174    0.0018    0.3    0.01]);
h.Interpreter = 'latex';
    
end;    
    
    