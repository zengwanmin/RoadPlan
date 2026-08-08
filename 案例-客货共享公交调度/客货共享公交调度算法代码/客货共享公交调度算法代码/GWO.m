function [bestY, bestX, recording] = GWO(x, y, option, data)
    %% 灰狼优化算法
    %% 初始化
    recording.bestFit = zeros(option.maxIteration+1, 1);
    recording.meanFit = zeros(option.maxIteration+1, 1);
    numAgents = option.numAgent;
    dim = option.dim;
    fobj = option.fobj;

    %% 初始化Alpha, Beta, Delta狼群
    Alpha_Pos = zeros(1, dim);
    Beta_Pos = zeros(1, dim);
    Delta_Pos = zeros(1, dim);
    Alpha_Score = inf;
    Beta_Score = inf;
    Delta_Score = inf;

    %% 根据适应度初始化Alpha, Beta, Delta狼群
    [y_g, position] = sort(y);
    y_p = y;
    Alpha_Pos = x(position(1), :);
    Alpha_Score = y_g(1);
    Beta_Pos = x(position(2), :);
    Beta_Score = y_g(2);
    Delta_Pos = x(position(3), :);
    Delta_Score = y_g(3);
    %% 记录初始最优解
    y_g = Alpha_Score;
    x_g = Alpha_Pos;
    recording.bestFit = y_g;
    recording.meanFit = mean(y_p);

    %% 开始更新
    for iter = 1:option.maxIteration
        a = 2 - iter * (2 / option.maxIteration); % 线性调整a的值
        for i = 1:numAgents % 更新每个狼的位置
            for j = 1:dim
                % 更新位置X1, X2, X3
                r1 = rand;
                r2 = rand;
                A1 = 2 * a * r1 - a;
                C1 = 2 * r2;
                D_Alpha = abs(C1 * Alpha_Pos(j) - x(i, j));
                X1 = Alpha_Pos(j) - A1 * D_Alpha;
                
                r1 = rand;
                r2 = rand;
                A2 = 2 * a * r1 - a;
                C2 = 2 * r2;
                D_Beta = abs(C2 * Beta_Pos(j) - x(i, j));
                X2 = Beta_Pos(j) - A2 * D_Beta;
                
                r1 = rand;
                r2 = rand;
                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;
                D_Delta = abs(C3 * Delta_Pos(j) - x(i, j));
                X3 = Delta_Pos(j) - A3 * D_Delta;
                
                % 更新狼只位置
                x(i, j) = (X1 + X2 + X3) / 3;
            end
            % 确保解在界限内
            x(i,:)=checkX(x(i,:),option,data);
            % 计算新解的适应度
            newFit=fobj(x(i,:),option,data);
            y(i)=newFit;
            if y(i) < y_p(i)
                y_p(i)=y(i);
            end
            
            % 更新Alpha, Beta, Delta狼群
            if newFit < Alpha_Score
                Delta_Score = Beta_Score;
                Delta_Pos = Beta_Pos;
                Beta_Score = Alpha_Score;
                Beta_Pos = Alpha_Pos;
                Alpha_Score = newFit;
                Alpha_Pos = x(i, :);
                if Alpha_Score < y_g
                    y_g = Alpha_Score;
                    x_g = Alpha_Pos;
                end
            elseif newFit < Beta_Score
                Delta_Score = newFit;
                Delta_Pos = x(i, :);
                Beta_Score = newFit;
                Beta_Pos = x(i, :);
            elseif newFit < Delta_Score
                Delta_Score = newFit;
                Delta_Pos = x(i, :);
            end
        end
        
        %% 更新记录
        recording.bestFit(iter+1) = y_g;
        recording.meanFit(iter+1) = mean(y_p);
        
    end
    bestY=y_g;
    bestX=x_g;
end
