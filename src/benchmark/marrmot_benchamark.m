% Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
% This file is part of the Modular Assessment of Rainfall-Runoff Models
% Toolbox (MARRMoT).
% MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
% WARRANTY. See <https://www.gnu.org/licenses/> for details.

% Contact:  l.trotter@unimelb.edu.au

% This example workflow  contains an example application of a single model 
% to a single catchment.
% It includes 5 steps:
%
% 1. Data preparation
% 2. Model choice and setup
% 3. Model solver settings
% 4. Model generation and set-up
% 5. Model runs
% 6. Output vizualization

%% 1. Prepare data
% 设置数据长度限制
data_length = 500;  % 可以根据需要修改为2000或其他值

% 读取CSV文件并计时
tic;
data = readtable('data/gr4j/sample.csv');
read_time = toc;
fprintf('读取CSV文件耗时: %.3f 秒\n', read_time);

% 限制数据长度
if height(data) > data_length
    data = data(1:data_length, :);
end

% 创建气候数据输入结构
input_climatology.precip   = data.prec;                    % 日降水量 [mm/d]
input_climatology.temp     = data.temp;                    % 日平均温度 [degree C]
input_climatology.pet      = data.pet;                     % 潜在蒸散发 [mm/d]
input_climatology.delta_t  = 1;                           % 时间步长: 1 [d]

% 保存观测流量数据用于后续比较
data_MARRMoT_examples.streamflow = data.qobs;
data_MARRMoT_examples.dates_as_datenum = datenum(data.time);

fprintf('数据处理完成，总数据长度: %d\n', height(data));

%% 2. Define the model settings
% NOTE: this example assumes that parameter values for this combination of
% model and catchment are known. 

% Model name 
% NOTE: these can be found in the Model Descriptions
model = 'm_07_gr4j_4p_2s';                     

% Parameter values
% NOTE: descriptions of these parameters can be found in the Model
% descriptions (supplementary materials to the main paper). Alternatively,
% the parameters are described in each model function. Right-click the
% model name above (i.e. 'm_29_hymod_5p_5s') and click 
% "Open 'm_29_hymod_5p_5s'". Parameters are listed on lines 44-50.

input_theta     = [ 50;                                                     % Soil moisture depth [mm]
                     0.1;                                                   % Soil depth distribution parameter [-]
                     20;                                                   % Fraction of soil moisture excess that goes to fast runoff [-]
                     3.5];                                                 % Runoff coefficient of the lower store [d-1]

% Initial storage values
% NOTE: see the model function for the order in which stores are given. For
% HyMOD, this is on lines 86-91.

input_s0       = [0;                                                       % Initial soil moisture storage [mm]
                   0];                                                   % Initial fast flow 1 storage [mm]
                                                                        % Initial slow flow storage [mm]
%% %% 3. Define the solver settings  
% Create a solver settings data input structure. 
% NOTE: the names of all structure fields are hard-coded in the model class.
%  These should not be changed.
input_solver_opts.resnorm_tolerance = 0.001;                                       % Root-finding convergence tolerance
input_solver_opts.resnorm_maxiter   = 6;                                           % Maximum number of re-runs
% these are the same settings that run by default if no settings are given
              
%% 4. Create a model object
% Create a model object
m = feval(model);

% Set up the model
m.theta         = input_theta;
m.input_climate = input_climatology;
%m.delta_t       = input_climatology.delta_t;         % unnecessary if input_climate already contains .delta_t
m.solver_opts   = input_solver_opts;
m.S0            = input_s0;

%% 5. Run the model and extract all outputs
% This process takes ~6 seconds on a i7-4790 CPU 3.60GHz, 4 core processor.
[output_ex,...                                                             % Fluxes leaving the model: simulated flow (Q) and evaporation (Ea)
 output_in,...                                                             % Internal model fluxes
 output_ss,...                                                             % Internal storages
 output_waterbalance] = ...                                                % Water balance check              
                        m.get_output();                            
    
%% 6. Analyze the outputs                   
% Prepare a time vector
t = data_MARRMoT_examples.dates_as_datenum;

% Compare simulated and observed streamflow by calculating the Kling-Gupta
% Efficiency (KGE). Other objective functions provided are inverse KGE and
% multi-objective average KGE (0.5*(KGE(Q) + KGE(1/Q))
tmp_obs  = data_MARRMoT_examples.streamflow;
tmp_sim  = output_ex.Q;
tmp_kge  = of_KGE(tmp_obs,tmp_sim);                                         % KGE on regular flows
tmp_kgei = of_inverse_KGE(tmp_obs,tmp_sim);                                 % KGE on inverse flows
tmp_kgem = of_mean_hilo_KGE(tmp_obs,tmp_sim);                               % Average of KGE(Q) and KGE(1/Q)

figure('color','w'); 
    box on;
    hold on; 
    
    h1 = plot(t,tmp_obs);
    h2 = plot(t,tmp_sim);
    
    legend('Observed','Simulated')
    title(['Kling-Gupta Efficiency = ',num2str(tmp_kge)])
    ylabel('Streamflow [mm/d]')
    xlabel('Time [d]')
    datetick;
    set(h1,'LineWidth',2)
    set(h2,'LineWidth',2)
    set(gca,'fontsize',16);

clear hi h2 
    
% Investigate internal storage changes
figure('color','w');
    
    p1 = subplot(311);
        hold on;
        h1 = plot(t,output_ss.S1);
        title('Simulated storages')
        ylabel('Soil moisture [mm]')
        datetick;
        
    p2 = subplot(312);
        box on;
        hold all;
        h2 = plot(t,output_ss.S2);
        legend('Fast store 1','Fast store 2','Fast store 3')
        ylabel('Fast stores [mm]')
        datetick;
        

    set(p1,'fontsize',16)
    set(p2,'fontsize',16)

        
    set(h1,'LineWidth',2)
    set(h2,'LineWidth',2)


clear p* h* t