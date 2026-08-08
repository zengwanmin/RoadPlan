%-----------------------------------------------------------------------------------------------------------------%
%  JS optimizer                                                                                                   %
%  Jellyfish Search Optimizer (JS) source codes demo version 1.0  Developed in MATLAB R2016a                      %
%  Author and programmer:                                                                                         %
%         Professor        Jui-Sheng Chou                                                                         %
%         Ph.D. Candidate  Dinh- Nhat Truong                                                                      %
%  Paper: A Novel Metaheuristic Optimizer Inspired By Behavior of Jellyfish in Ocean,                             %
%         Applied Mathematics and Computation. Computation, Volume 389, 15 January 2021, 125535.                  %
%  DOI:   https://doi.org/10.1016/j.amc.2020.125535                                                               %
%                                     PiM Lab, NTUST, Taipei, Taiwan, July-2020                                   %
%-----------------------------------------------------------------------------------------------------------------%
function [u,fval,fbestvl,recording]=js(x,CostFunction,Lb,Ub,nd,para);

%% Problem Definition
nVar=nd;                           % Number of Decision Variables
VarSize=[1 nVar];                  % Size of Decision Variables Matrix
if length(Lb)==1
    VarMin=Lb.*ones(1,nd);         % Lower Bound of Variables
    VarMax=Ub.*ones(1,nd);         % Upper Bound of Variables
else
    VarMin=Lb;
    VarMax=Ub;
end
%% AJS Parameters
MaxIt = para(1);%1000;        % Maximum Number of Iterations
nPop = para(2);%50;           % Population Size
% initialization using Logistic map using Eq. (18)
popi=x;
% Evaluate population
for i=1:nPop
    popCost(i) = CostFunction(popi(i,:));
end
%% 初始化recording
recording.bestFit = zeros(MaxIt+1, 1);
recording.meanFit = zeros(MaxIt+1, 1);
%% AJS Main Loop
for it=1:MaxIt
    Meanvl=mean(popi,1);
    [value,index]=sort(popCost);
    BestSol=popi(index(1),:);
    BestCost=popCost(index(1));

    % 这里添加记录bestFit和meanFit的代码
    recording.bestFit(it+1) = BestCost;
    recording.meanFit(it+1) = mean(popCost);
    
    for i=1:nPop
        % Calculate time control c(t) using Eq. (17);
        Ar=(1-it*((1)/MaxIt))*(2*rand-1);
        if abs(Ar)>=0.5
            %% Folowing to ocean current using Eq. (11)
            newsol = popi(i,:)+ rand(VarSize).*(BestSol - 3*rand*Meanvl);
            % Check the boundary using Eq. (19)
            newsol = simplebounds(newsol,VarMin,VarMax);
            % Evaluation
            newsolCost = CostFunction(newsol);
            % Comparison
            if newsolCost<popCost(i)
                popi(i,:) = newsol;
                popCost(i)=newsolCost;
                if popCost(i) < BestCost
                    BestCost=popCost(i);
                    BestSol = popi(i,:);
                end
            end
        else
            %% Moving inside swarm
            if rand<=(1-Ar)
                % Determine direction of jellyfish by Eq. (15)
                j=i;
                while j==i
                    j=randperm(nPop,1);
                end
                Step = popi(i,:) - popi(j,:);
                if popCost(j) < popCost(i)
                    Step = -Step;
                end
                % Active motions (Type B) using Eq. (16)
                newsol = popi(i,:) + rand(VarSize).*Step;
            else
                % Passive motions (Type A) using Eq. (12)
                newsol = popi(i,:) + 0.1*(VarMax-VarMin)*rand;
            end
            % Check the boundary using Eq. (19)
            newsol = simplebounds(newsol, VarMin,VarMax);
            % Evaluation
            newsolCost = CostFunction(newsol);
            % Comparison
            if newsolCost<popCost(i)
                popi(i,:) = newsol;
                popCost(i)=newsolCost;
                if popCost(i) < BestCost
                    BestCost=popCost(i);
                    BestSol = popi(i,:);
                end
            end
        end
    end
    %% Store Record for Current Iteration
    fbestvl(it)=BestCost;
    if it>=2000
        if abs(fbestvl(it)-fbestvl(it-100))<1e-350
            break;
        end
    end
end
u=BestSol;
fval=fbestvl(it);
NumEval=it*nPop;
end



%% This function is for checking boundary by using Eq. 19
function s=simplebounds(s,Lb,Ub)
ns_tmp=s;
I=ns_tmp<Lb;
% Apply to the lower bound
while sum(I)~=0
    ns_tmp(I)=Ub(I)+(ns_tmp(I)-Lb(I));
    I=ns_tmp<Lb;
end
% Apply to the upper bound
J=ns_tmp>Ub;
while sum(J)~=0
    ns_tmp(J)=Lb(J)+(ns_tmp(J)-Ub(J));
    J=ns_tmp>Ub;
end
% Check results
s=ns_tmp;
end