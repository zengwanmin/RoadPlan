clc;
clear all;
close all;
warning off
%%
noRng=2;
rng('default')
rng(noRng)
%%
global data
filename=fullfile(pwd,'敏感性分析','900_2500.xlsx');
data.Line=xlsread(filename,1);
data.TimeList=xlsread(filename,2);
data.Demand=xlsread(filename,3);
data.Type=xlsread(filename,4);

data.t1=1/60; %上下乘客时间
data.t2=10/60; %装卸20个集装箱时间
data.numBus=6; %车辆数
data.T=[55/60,50/60]; %每条线的运行时间。
data.D=[40.2,37.3];   %每条线的运行距离
data.v=data.D./data.T; %每条线的平均速度

data.maxT=70/60; % 最大人均时间约束 
data.minLambda=0.1; % 最低客仓容量

data.sumDemandOfP=sum(data.Demand(data.Demand(:,7)==1,6)); % 客运总需求
data.sumDemandOfC=sum(data.Demand(data.Demand(:,7)==2,6)); % 货运总需求
data.wz=0.313; %上层模型Z的熵权法权重
data.wt=0.687; %下层模型T的熵权法权重
%%

%%
lb=0;
ub=1;
Xdim=12*data.numBus; % 调度方案x的维度
Ydim=data.numBus; % 车型y的维度
Lambdadim=data.numBus; % 客仓占比λ的维度
option.lb=lb;
option.ub=ub;
option.dim=Xdim+Ydim+Lambdadim;
if length(option.lb)==1
    option.lb=ones(1,option.dim)*option.lb;
    option.ub=ones(1,option.dim)*option.ub;
end
option.fobj=@aimFcn_1;
option.showIter=0;

%% 算法参数设置 Parameters
% 基本参数
option.numAgent=20;        %种群个体数 size of population
option.maxIteration=100;    %最大迭代次数 maximum number of interation
% option.maxIteration=100;    %最大迭代次数 maximum number of interation
%% 遗传算法
option.p1_GA=0.8;
option.p2_GA=0.2;
%% 粒子群
option.w_pso=0.8;
option.c1_pso=2;
option.c2_pso=2;
%%
%str_legend=[{'GA'},{'PSO'},{'GWO'},{'JS'},{'IJS'}];
%aimFcn=[{@GA},{@PSO},{@GWO},{@myJS},{@myIJS}];
str_legend=[{'IJS'}];
aimFcn=[{@myIJS}];
%% 
%% 初始化
rng(noRng)
x=ones(option.numAgent,option.dim);
y=ones(option.numAgent,1);
for i=1:option.numAgent
    x(i,:)=rand(size(option.lb)).*(option.ub-option.lb)+option.lb;
    y(i)=option.fobj(x(i,:),option,data);
end
%% 使用算法求解
bestX=x;
for i=1:length(aimFcn)
    rng(noRng)
    tic
    [bestY(i,:),bestX(i,:),recording(i)]=aimFcn{i}(x,y,option,data);
    tt(i)=toc;
    disp([str_legend{i}, ' 求解时间: ', num2str(tt(i)), ' 秒']);
    disp([recording(i).bestFit(option.maxIteration+1)]);
end
%% Iterative curve:MeanFit
figure
hold on
for i=1:length(aimFcn)
    switch i
        case 1
            style = '-'; 
    %        mycolor = [93 157 200]/255;
        case 2
            style = ':'; 
    %        mycolor = [255 158 73]/255;
        case 3
            style = '-.';
     %       mycolor = [99 185 99]/255;
        case 4
            style = ':';
        otherwise
            style = '--'; 
     %       mycolor = [225 104 105]/255;
    end
    %plot(-recording(i).meanFit,style,'LineWidth',2,color=mycolor) %
    plot(recording(i).meanFit,'-.','LineWidth',2)
end
%legend([{'JS'},{'IJS'},{'GA'},{'PSO'}])
legend(str_legend)
xlabel('Iteration')
ylabel('Fit')
title('Iterative curve:MeanFit') % ：meanFit
%% 绘制迭代曲线:bestFit
figure
hold on
for i=1:length(aimFcn)
    plot(recording(i).bestFit,'LineWidth',2)
end
legend(str_legend)
xlabel('Iteration')
ylabel('Fit')
title('Iterative curve:BestFit')
%% 计算结果
for i=1:length(str_legend)
    str=[str_legend{i},'优化后方案'];
    [~,result0(i)]=option.fobj(bestX(i,:),option,data);
    drawPC(str,result0(i),data,option)
    disp([' ']);
end
