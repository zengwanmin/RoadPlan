function [bestY, bestX, recording] = ACO(x, y, option, data)
    %% 蚁群算法
    %% 初始化
    numAgents = option.numAgent; % 蚂蚁数量
    dim = option.dim; 
    LB = option.lb; 
    UB = option.ub; 
    fobj = option.fobj; 
    
    recording.bestFit=zeros(option.maxIteration+1,1);
    recording.meanFit=zeros(option.maxIteration+1,1);
    [y_g, position] = min(y);
    x_g = x(position(1), :);
    y_p=y;
    x_p=x;
    recording.bestFit = y_g;
    recording.meanFit = mean(y_p);

    tau0 = 1; % 初始信息素浓度
    rho = 0.5; % 信息素的挥发率/蒸发系数
    Q = 10; % 信息素的重要程度
    %x = x; % 蚂蚁的位置
    %y = y; % 蚂蚁的适应度
    tau = tau0 * ones(numAgents, 1); % 初始化信息素矩阵
    
    %% 开始迭代
    for iter = 1:option.maxIteration
        % 更新蚂蚁位置
        for i = 1:numAgents 
            % 选择下一个位置，这里使用轮盘赌选择
            probabilities = tau ./ (sum(tau) + eps);
            randNumbers = rand(1, length(probabilities));
            nextIdx = find(cumsum(probabilities) > randNumbers(1), 1);
            if ~isempty(nextIdx) % 检查nextIdx是否有效
                disp(i)
                x(i, :) = x_p(nextIdx, :); % 更新位置
            else
                
                x(i, :) = x_p(i, :); % 没有有效的选择，保持当前位置
            end
            x(i, :) = checkX(x(i, :), option, data); % 确保解在界限内
            %disp(x(i, :))
            
            y(i) = fobj(x(i, :), option, data);
            if y(i)<y_p(i)
                y_p(i)=y(i);
                x_p(i,:)=x(i,:);
                if y_p(i)<y_g
                    y_g=y_p(i);
                    x_g=x_p(i,:);
                end
            end
            tau(i) = (1 - rho) * tau(i) + Q / y(i);
            %disp(tau(i))
        end
        %% 更新记录
        recording.bestFit(1+iter)=y_g;
        recording.meanFit(1+iter)=mean(y_p);
       
        
    end
    bestY=y_g;
    bestX=x_g;
end