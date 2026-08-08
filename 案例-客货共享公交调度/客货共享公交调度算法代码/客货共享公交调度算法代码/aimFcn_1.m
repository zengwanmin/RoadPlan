function [fit,result]=aimFcn_1(x,option,data)
%%
global data
x=reshape(x,12+1+1,data.numBus);
%%
nowT_Bus=zeros(data.numBus,1);
demand=data.Demand;
timelist=data.TimeList;
type=data.Type;
recording=[];

% 下层模型 T
Tcru=0; % 总乘客乘车时间
Tdwell=0; % 总装卸耽误时间
Tdet=0; % 总乘客滞留时间
Tw=0; % 总乘客等待时间 在站点
wt=data.wt; %下层模型T的熵权法权重

% 上层模型 Z
Ctoll=0; % 道路通行费
Cidle=0; % 站点停滞总成本
Ckm=0; % 线路运行成本
Cfix=0; % 车队配置成本
Eu=0; % 客运总收入
Ef=0; % 货运总收入
wz=data.wz; %上层模型Z的熵权法权重

sumP=0; % 总乘客数
sumC=0; % 总货运数
maxT=data.maxT; % 最大人均时间约束
minLambda=data.minLambda; % 最低客仓容量
nowMinLambda=1;

detention=[]; % 滞留矩阵: 站点编号，滞留人数，滞留开始时间，行程长度，类型
sumDemandOfP=data.sumDemandOfP; % 客运总需求
sumDemandOfC=data.sumDemandOfC; % 货运总需求
sumPCapacity=0; % 客运能力
sumCCapacity=0; % 货运能力


for noB=1:data.numBus % 循环遍历每辆公交车
    % 调度方案x
    %[~,S]=sort(x(1:12,noB)); % 排序：行程顺序
    S=[1,2,3,4,5,6,7,8,9,10,11,12];
    % x(1:12, noB) = [1, 0.5, 0.3, 0.8, 0.2] 排序后：[0.2, 0.3, 0.5, 0.8, 1] S = [5, 3, 2, 4, 1]  (这些是原始位置的索引)
    % 车型y
    busTypeValue=x(1,noB);
    if busTypeValue<1/3 % 映射到三种车型
        busType=0; % 0-小型
    elseif busTypeValue<2/3 
        busType=1; % 1/3-中型
    else 
        busType=2; % 2/3-大型
    end

    t=find(type(:,1)==busType);
    eCost=type(t,2); % 电耗成本
    bCost=type(t,3); % 购置成本
    sCost=15; % 公交车单位停留时间成本 15元/小时
    pIncome=type(t,4); % 每公里客运价
    cIncome=0.05; % 每公里货运价 
    pBase=2.5; % 客运起步价
    cBase=0.1; % 货运起步价 元/件 1
    V=type(t,5); % 车厢容量
    Lambda=x(2,noB); % 客仓占比
    nowMinLambda=min(Lambda,nowMinLambda); % 更新现在的最低客仓比
    %disp(nowMinLambda)
    vp=0.696; % 客仓内设置一个座位的平均空间面积 0.696
    PCapacity=floor(V*Lambda/vp); % 客仓容量（座位数）
    oneC=0.3*0.3; % 每件货物的占地面积 平方米
    totalC=(1-Lambda)*V; % 货仓面积 平方米
    CCapacity=floor(totalC/oneC)*(1.5/0.3); % 货仓容量（货物数）：规定车内货物限高1.5米
    sumPCapacity=sumPCapacity+PCapacity*2;
    sumCCapacity=sumCCapacity+CCapacity*2;

    %disp(sumPCapacity)
    Cfix=Cfix+bCost; % 公交购置成本
   
    %起程
    no=1; % 用于遍历data.TimeList的索引
    flag=0; % 检测是否找到了匹配的需求行程
    while 1 % 处理noB车的所有行程（在TimeList），只要找到匹配需求/超出TimeList范围，break
        tempR=[]; 
        
        if no>length(timelist(:,1)) % 如果当前的索引no超出了data.TimeList的范围
            recording=[recording;tempR]; % 添加到总记录，退出
            break
        end
        noLine=timelist(S(no),5); % 获取线路编号
        if timelist(S(no),4)==2 || timelist(S(no),6)==1 % 如果当前班次是返程，或已执行
            no=no+1; 
            continue; % 跳过这次循环
        end

        % 初始化公交车状态
        nowT_Bus(noB)=timelist(S(no),2)*24; % 当前时间，为出发时间（转换为24小时制）
        Pcount=0; % 实时乘客数
        Ccount=0; % 实时货物数
        
        p=find(data.Line(:,4)==noLine); % 从Line表中找到所有与当前线路noLine匹配的线路（1 or 2）

        for i=1:length(p) % 遍历该线路上的每个站点
            ST=nowT_Bus(noB); % 公交车当前时间
            index=p(i); % Line中的行的索引，代表每个from-to
            nowP=data.Line(index,1); % 当前站点编号
            nextP=data.Line(index,2); % 下一个站点编号
            if nextP==11 % 现在是站点9
                a=1; % 代表要到终点站了
            else
                a=0;
            end
            D=data.Line(index,3); % 当前路段的距离km
            AT=ST+D/data.v(noLine); % 预计抵达下一个站点的时间
            needT=D/data.v(noLine); % 当前路段所需的时间
            
            % 下客/下货
            % 处理需求，从demand表查找需求：当前站点下客/下货、正在执行中、路线与方向正确
            pp=find(demand(:,3)==nowP & demand(:,6)>0 & demand(:,9)==1 & demand(:,10)==noLine & demand(:,11)==1  & demand(:,6)>0);
            if isempty(pp) % 如果没有需求，不需要停留
                DT=0; % 卸货时间0
                typeWork=0; % 类型0
                WT=0; % 等待时间0
                need=0; % 载量0
            else
                %disp(1);
                % 处理匹配到的每个需求
                for j=1:length(pp)
                    ppp=pp(j); % 每行需求
                    typeWork=demand(ppp,7); % 装卸类型
                    need=demand(ppp,6); % 载量
                    len=demand(ppp,8); % 线路长度
                    if typeWork==1 % 客运
                        DT=data.t1; 
                        Pcount=Pcount-need; 
                        Tcru=Tcru+need*(len/data.v(noLine)); % 累计总乘客乘车时间：a上车b下车的人数*ab间路程时间
                        if len>=20          
                            Eu=Eu+need*((len-20)*pIncome+pBase); % 计算客运收入
                        else
                            Eu=Eu+need*pBase;
                        end
                        
                        %disp(len*pIncome+pBase);  % 计算客运收入
                    else % 货运
                        DT=data.t2; 
                        Ccount=Ccount-need;
                        if len>=20          
                            Ef=Ef+need*((len-20)*cIncome+cBase); % 计算货运收入
                        else
                            Ef=Ef+need*cBase;
                        end
                    end
                    Tdwell=Tdwell+DT; % 累计装卸时间
                    Cidle=Cidle+DT*sCost; % 累计停滞成本
                    WT=0;
                    demand(ppp,6)=0; % 更新需求表，将已处理的需求量设为0
                    demand(ppp,9)=0;
                end
            end
           
            
            % 处理滞留
            if ~isempty(detention)
                matchingIndices=find(detention(:,1)==nowP); % nowP的滞留信息
                if ~isempty(matchingIndices) % 找到匹配的滞留信息
                    for j=1:length(matchingIndices) % 遍历所有匹配的滞留信息
                        idx=matchingIndices(j);
                        currentDetention=detention(idx,:); % 获取当前滞留信息
                        if currentDetention(5)==1 % 客运
                            canBoard=min(currentDetention(2),PCapacity-Pcount); % 检查是否有滞留乘客可以上车
                            if canBoard>0 % 人数允许上车
                                detentionTime=ST-currentDetention(3);
                                if detentionTime>=0 % 时间允许上车
                                    % 更新滞留人数
                                    detention(idx,2)=detention(idx,2)-canBoard;
                                    sumP=sumP+canBoard;
                                    %Pcount=Pcount+canBoard;
                                    % 计算总滞留时间、总乘车时间、Eu
                                    Tdet=Tdet+canBoard*detentionTime;
                                    len=currentDetention(4);
                                    Tcru=Tcru+canBoard*(len/data.v(noLine));
                                    if len>=20          
                                        Eu=Eu+need*((len-20)*pIncome+pBase); % 计算客运收入
                                    else
                                        Eu=Eu+need*pBase;
                                    end
                                    %Eu=Eu+canBoard*(len*pIncome+pBase);  % 计算客运收入
                                    if detention(idx,2)==0 % 如果所有滞留乘客都上了车  
                                        detention(idx,:)=[0,0,0,0,0]; % 从列表中删除这一行
                                    end
                                 end                  
                            end
                        else % 货运
                            canBoard=min(currentDetention(2),CCapacity-Ccount); % 检查是否有滞留乘客可以上车
                            if canBoard>0 % 人数允许上车
                                detentionTime=ST-currentDetention(3);
                                if detentionTime>=0 % 时间允许上车
                                    detention(idx,2)=detention(idx,2)-canBoard;
                                    sumC=sumC+canBoard;
                                    len=currentDetention(4);
                                    if len>=20          
                                        Ef=Ef+canBoard*((len-20)*cIncome+cBase); % 计算货运收入
                                    else
                                        Ef=Ef+canBoard*cBase;
                                    end
                                    %Ef=Ef+canBoard*(len*cIncome+cBase);
                                    if detention(idx,2)==0 % 如果所有滞留乘客都上了车  
                                        detention(idx,:)=[0,0,0,0,0]; % 从列表中删除这一行
                                    end
                                 end                  
                            end
                        end
                    end
                end
            end
            
            
            % 上客/上货
            % 处理需求，从demand表查找需求：当前站点上客/上货、所在时间窗内、需求未被处理、未被执行、路线方向正确
            pp=find(demand(:,2)==nowP & demand(:,4)*24-0.5<=ST & demand(:,5)*24>=ST & demand(:,6)>0 & demand(:,9)==0 & demand(:,10)==noLine & demand(:,11)==1);
            if isempty(pp) % 如果没有需求，不需要停留
                DT=0; % 卸货时间0
                typeWork=0; % 类型0
                WT=0; % 等待时间0
                need=0; % 载量0
            else
                flag=1; % 代表找到了匹配的需求行程
                %disp(1)
                % 处理匹配到的每个需求
                for j=1:length(pp)
                    ppp=pp(j); % 每行需求
                    demand(ppp,9)=1; % 该需求执行中
                    typeWork=demand(ppp,7); % 装卸类型
                    need=demand(ppp,6); % 载量
                    len=demand(ppp,8); % 线路长度
                    if typeWork==1 % 客运
                        DT=data.t1; 
                        % 计算乘客等待时间
                        Tw=Tw+abs(demand(ppp,4)*24-ST);
                        if Pcount+need <= PCapacity % 能上车
                            Pcount=Pcount+need;
                            sumP=sumP+need;  
                        else % 存在滞留
                            stayPassengers=Pcount+need-PCapacity; 
                            % 记录滞留信息：站点编号，滞留人数，滞留开始时间，行程长度，类型
                            detention=[detention;nowP,stayPassengers,ST,len,1];
                            sumP=sumP+PCapacity-Pcount;
                            Pcount=PCapacity; % 满员
                        end
                    else % 货运
                        DT=data.t2; 
                        if Ccount+need <= CCapacity % 能上车
                            Ccount=Ccount+need;
                            sumC=sumC+need;
                        else % 存在滞留
                            stayCargos=Ccount+need-CCapacity; 
                            detention=[detention;nowP,stayCargos,ST,len,2];
                            sumC=sumC+CCapacity-Ccount;
                            Ccount=CCapacity; % 满员
                        end
                    end
                    Tdwell=Tdwell+DT; % 累计装卸时间
                    Cidle=Cidle+DT*sCost; % 累计停滞成本
                    % 计算等待时间
                    WT=max(demand(ppp,4)*24-ST,0);
                end
            end

            ET=ST+WT+DT+needT; % 到达下一站的时间
            
            if a==1 % 单独处理终点站的下客/下货
                % 处理需求，从demand表查找需求：当前站点下客/下货、正在执行中、路线与方向正确
                pp=find(demand(:,3)==nextP & demand(:,6)>0 & demand(:,9)==1 & demand(:,10)==noLine & demand(:,11)==1);
                if ~isempty(pp) % 如果没有需求，不需要停留
                    % 处理匹配到的每个需求
                    for j=1:length(pp)
                        ppp=pp(j); % 每行需求
                        typeWork=demand(ppp,7); % 装卸类型
                        need=demand(ppp,6); % 载量
                        len=demand(ppp,8); % 线路长度
                        if typeWork==1 % 客运
                            DT=data.t1; 
                            Pcount=Pcount-need; 
                            Tcru=Tcru+need*(len/data.v(noLine)); % 累计总乘客乘车时间：a上车b下车的人数*ab间路程时间
                            %Eu=Eu+need*(len*pIncome+pBase);  % 计算客运收入
                            if len>=20          
                                Eu=Eu+need*((len-20)*pIncome+pBase); % 计算客运收入
                            else
                                Eu=Eu+need*pBase;
                            end
                        else % 货运
                            DT=data.t2; 
                            Ccount=Ccount-need;
                            %Ef=Ef+need*(len*cIncome+cBase); % 计算货运收入
                            if len>=20          
                                Ef=Ef+need*((len-20)*cIncome+cBase); % 计算货运收入
                            else
                                Ef=Ef+need*cBase;
                            end
                        end
                        Tdwell=Tdwell+DT; % 累计装卸时间
                        Cidle=Cidle+DT*sCost; % 累计停滞成本
                        WT=0;
                        demand(ppp,6)=0; % 更新需求表，将已处理的需求量设为0
                        demand(ppp,9)=0;
                    end
                end
                if flag==1 % 该线路计入方案
                    Ckm=Ckm+data.D(noLine)*eCost; % 线路运行成本：线路总长*电耗成本
                end
            end
            % 记录本次行程
            tempR=[tempR;noB,noLine,nowP,nextP,typeWork,need,ST,AT,ET,WT,DT,0,0,0,needT,1,busType,Lambda,Tcru,Tdwell,Tdet,Tw,sumP,sumC,PCapacity,CCapacity,Ctoll,Cidle,Ckm,Cfix,Eu,Ef,sumPCapacity,sumCCapacity];
            % 1公交车编号，2线路编号 3出发节点 4抵达节点 5装卸类型 6装卸数量 7出发时间 8抵达时间 9离开时间 10等待时间 
            % 11装卸时间 12出发时SOC 13抵达时SOC 14离开时SOC 15路途时间 16往返 
            % 17车型y 18客仓占比λ 19Tcru 20Tdwell 21Tdet 22Tw
            % 23总乘客sumP 24总货物sumC 25客仓座位 26货仓容量
            % 27Ctoll 28Cidle 29Ckm 30Cfix 31Eu 32Ef
            % 33客运能力 34货运能力

            nowT_Bus(noB)=ET; % 更新当前时间
           
        end
        % timelist(S(no),6)=1; % 该行程已执行
        if flag==1
            timelist(S(no),6)=1;
        end
        
        %
        %timelist(S(no),6)=1;
        %recording=[recording;tempR];
        %break;
        %

        no=no+1; % 处理下一个行程
        if flag==1 || no>length(timelist(:,1)) % 找到了匹配的行程\no大于TimeList第一列长度（即处理完所有行程），退出循环
            recording=[recording;tempR];
            break
        end
        
    end

    %返程
    no=1; % 用于遍历data.TimeList的索引
    flag=0; % 检测是否找到了匹配的需求行程
    nowT_Bus0(noB)=nowT_Bus(noB); % 在返程开始时的时间  
    
    while 1 % 处理noB车的所有行程
        tempR=[];
        if no>length(timelist(:,1))
            recording=[recording;tempR];
            break
        end
        noLine=timelist(S(no),5); % 获取返程线路编号
        if timelist(S(no),4)==1 || timelist(S(no),6)==1 %&& noLine0==noLine % 如果当前班次是起程且已执行
            no=no+1;
            continue; % 跳过这个行程
        end
        if timelist(S(no),2)*24<nowT_Bus0(noB)  % 如果该行程时间早于返程时间
            no=no+1;
            continue; % 跳过这个行程
        end

        % 初始化公交车状态
        nowT_Bus(noB)=timelist(S(no),2)*24; % 当前时间为行程的出发时间
        Pcount=0; % 实时乘客数
        Ccount=0; % 实时货物数

        p=find(data.Line(:,4)==noLine); % 从Line表中找到所有与当前线路noLine匹配的线路（1 or 2）
        p=p(end:-1:1); % 并反转顺序，以便按逆序处理站点
        for i=1:length(p)
            ST=nowT_Bus(noB);
            index=p(i);
            nowP=data.Line(index,2);
            nextP=data.Line(index,1);
            if nextP==10
                a=1;
            else
                a=0;
            end
            D=data.Line(index,3);
            AT=ST+D/data.v(noLine);
            needT=D/data.v(noLine);
            
            % 下客/下货
            % 处理需求，从demand表查找需求：当前站点下客/下货、正在执行中、路线与方向正确
            pp=find(demand(:,3)==nowP & demand(:,9)==1 & demand(:,10)==noLine & demand(:,11)==2 & demand(:,6)>0);
            if isempty(pp) % 如果没有需求，不需要停留
                DT=0; % 卸货时间0
                typeWork=0; % 类型0
                WT=0; % 等待时间0
                need=0; % 载量0
            else
                % 处理匹配到的每个需求
                %disp(2)
                for j=1:length(pp)
                    ppp=pp(j); % 每行需求
                    typeWork=demand(ppp,7); % 装卸类型
                    need=demand(ppp,6); % 载量
                    len=demand(ppp,8); % 线路长度
                    if typeWork==1 % 客运
                        DT=data.t1; 
                        Pcount=Pcount-need;
                        Tcru=Tcru+need*(len/data.v(noLine)); % 累计总乘客乘车时间：a上车b下车的人数*ab间路程时间
                        if len>=20          
                            Eu=Eu+need*((len-20)*pIncome+pBase); % 计算客运收入
                        else
                            Eu=Eu+need*pBase;
                        end
                        %Eu=Eu+need*(len*pIncome+pBase);  % 计算客运收入
                    else % 货运
                        DT=data.t2; 
                        Ccount=Ccount-need;
                        %Ef=Ef+need*(len*cIncome+cBase); % 计算货运收入
                        if len>=20          
                            Ef=Ef+need*((len-20)*cIncome+cBase); % 计算货运收入
                        else
                            Ef=Ef+need*cBase;
                        end
                    end
                    Tdwell=Tdwell+DT; % 累计装卸时间
                    Cidle=Cidle+DT*sCost; % 累计停滞成本
                    % 计算等待时间
                    WT=0;
                    demand(ppp,6)=0; % 更新需求表，将已处理的需求量设为0
                    demand(ppp,9)=0; % 该需求完成
                end
            end

            % 处理滞留
            if ~isempty(detention)
                matchingIndices=find(detention(:,1)==nowP); % nowP的滞留信息
                if ~isempty(matchingIndices) % 找到匹配的滞留信息
                    for j=1:length(matchingIndices) % 遍历所有匹配的滞留信息
                        idx=matchingIndices(j);
                        currentDetention=detention(idx,:); % 获取当前滞留信息
                        if currentDetention(5)==1 % 客运
                            canBoard=min(currentDetention(2),PCapacity-Pcount); % 检查是否有滞留乘客可以上车
                            if canBoard>0 % 人数允许上车
                                detentionTime=ST-currentDetention(3);
                                if detentionTime>=0 % 时间允许上车
                                    % 更新滞留人数
                                    detention(idx,2)=detention(idx,2)-canBoard;
                                    sumP=sumP+canBoard;
                                    %Pcount=Pcount+canBoard;
                                    % 计算总滞留时间、总乘车时间、Eu
                                    Tdet=Tdet+canBoard*detentionTime;
                                    len=currentDetention(4);
                                    Tcru=Tcru+canBoard*(len/data.v(noLine)); 
                                    if len>=20          
                                        Eu=Eu+need*((len-20)*pIncome+pBase); % 计算客运收入
                                    else
                                        Eu=Eu+need*pBase;
                                    end
                                    %Eu=Eu+canBoard*(len*pIncome+pBase);  % 计算客运收入
                                    if detention(idx,2)==0 % 如果所有滞留乘客都上了车  
                                        detention(idx,:)=[0,0,0,0,0]; % 从列表中删除这一行
                                    end
                                 end                  
                            end
                        else % 货运
                            canBoard=min(currentDetention(2),CCapacity-Ccount); % 检查是否有滞留乘客可以上车
                            if canBoard>0 % 人数允许上车
                                detentionTime=ST-currentDetention(3);
                                if detentionTime>=0 % 时间允许上车
                                    detention(idx,2)=detention(idx,2)-canBoard;
                                    sumC=sumC+canBoard;
                                    len=currentDetention(4);
                                    %Ef=Ef+canBoard*(len*cIncome+cBase);
                                    if len>=20          
                                        Ef=Ef+canBoard*((len-20)*cIncome+cBase); % 计算货运收入
                                    else
                                        Ef=Ef+canBoard*cBase;
                                    end
                                    if detention(idx,2)==0 % 如果所有滞留乘客都上了车  
                                        detention(idx,:)=[0,0,0,0,0]; % 从列表中删除这一行
                                    end
                                 end                  
                            end
                        end
                    end
                end
            end

            % 上客/上货
            % 处理需求，从demand表查找需求：当前站点上客/上货、所在时间窗内、需求未被处理、路线方向正确
            pp=find(demand(:,2)==nowP & demand(:,4)*24-0.5<=ST & demand(:,5)*24>=ST & demand(:,6)>0 & demand(:,9)==0 & demand(:,10)==noLine & demand(:,11)==2);
            if isempty(pp) % 如果没有需求，不需要停留
                DT=0; % 卸货时间0
                typeWork=0; % 类型0
                WT=0; % 等待时间0
                need=0; % 载量0
            else
                flag=1; % 代表找到了匹配的需求行程
                %disp(2)
                % 处理匹配到的每个需求
                for j=1:length(pp) 
                    ppp=pp(j); % 每行需求
                    demand(ppp,9)=1; % 该需求执行中
                    typeWork=demand(ppp,7); % 装卸类型
                    need=demand(ppp,6); % 载量
                    len=demand(ppp,8); % 线路长度
                    if typeWork==1 % 客运
                        DT=data.t1; 
                        % 计算乘客等待时间
                        Tw=Tw+abs(demand(ppp,4)*24-ST);
                        if Pcount+need <= PCapacity % 能上车
                            Pcount=Pcount+need;
                            sumP=sumP+need;  
                        else % 存在滞留
                            stayPassengers=Pcount+need-PCapacity; 
                            % 记录滞留信息：站点编号，滞留人数，滞留开始时间，行程长度，类型
                            detention=[detention;nowP,stayPassengers,ST,len,1];
                            sumP=sumP+PCapacity-Pcount;
                            Pcount=PCapacity; % 满员
                        end
                    else % 货运
                        DT=data.t2; 
                        if Ccount+need <= CCapacity % 能上车
                            Ccount=Ccount+need;
                            sumC=sumC+need;
                        else % 存在滞留
                            stayCargos=Ccount+need-CCapacity; 
                            detention=[detention;nowP,stayCargos,ST,len,2];
                            sumC=sumC+CCapacity-Ccount;
                            Ccount=CCapacity; % 满员
                        end
                    end
                    Tdwell=Tdwell+DT; % 累计装卸时间
                    Cidle=Cidle+DT*sCost; % 累计停滞成本
                    % 计算等待时间
                    WT=max(demand(ppp,4)*24-ST,0);
                end
            end

            ET=ST+WT+DT+needT; % 到达下一站的时间

            if a==1 % 单独处理终点站的下客/下货
                % 处理需求，从demand表查找需求：当前站点下客/下货、正在执行中、路线与方向正确
                pp=find(demand(:,3)==nextP & demand(:,6)>0 & demand(:,9)==1 & demand(:,10)==noLine & demand(:,11)==2);
                if ~isempty(pp) % 如果没有需求，不需要停留
                    % 处理匹配到的每个需求
                    for j=1:length(pp)
                        ppp=pp(j); % 每行需求
                        typeWork=demand(ppp,7); % 装卸类型
                        need=demand(ppp,6); % 载量
                        len=demand(ppp,8); % 线路长度
                        if typeWork==1 % 客运
                            DT=data.t1; 
                            Pcount=Pcount-need; 
                            Tcru=Tcru+need*(len/data.v(noLine)); % 累计总乘客乘车时间：a上车b下车的人数*ab间路程时间
                            if len>=20          
                                Eu=Eu+need*((len-20)*pIncome+pBase); % 计算客运收入
                            else
                                Eu=Eu+need*pBase;
                            end
                            %Eu=Eu+need*(len*pIncome+pBase);  % 计算客运收入
                        else % 货运
                            DT=data.t2; 
                            Ccount=Ccount-need;
                            %Ef=Ef+need*(len*cIncome+cBase); % 计算货运收入
                            if len>=20          
                                Ef=Ef+need*((len-20)*cIncome+cBase); % 计算货运收入
                            else
                                Ef=Ef+need*cBase;
                            end
                        end
                        Tdwell=Tdwell+DT; % 累计装卸时间
                        Cidle=Cidle+DT*sCost; % 累计停滞成本
                        WT=0;
                        demand(ppp,6)=0; % 更新需求表，将已处理的需求量设为0
                        demand(ppp,9)=0;
                    end
                end
                if flag==1 % 该线路计入方案
                    Ckm=Ckm+data.D(noLine)*eCost; % 线路运行成本：线路总长*电耗成本
                end
            end


            tempR=[tempR;noB,noLine,nowP,nextP,typeWork,need,ST,AT,ET,WT,DT,0,0,0,needT,2,busType,Lambda,Tcru,Tdwell,Tdet,Tw,sumP,sumC,PCapacity,CCapacity,Ctoll,Cidle,Ckm,Cfix,Eu,Ef,sumPCapacity,sumCCapacity];
            % 1公交车编号，2线路编号 3出发节点 4抵达节点 5装卸类型 6装卸数量 7出发时间 8抵达时间 9离开时间 10等待时间 
            % 11装卸时间 12出发时SOC 13抵达时SOC 14离开时SOC 15路途时间 16往返 
            % 17车型y 18客仓占比λ 19Tcru 20Tdwell 21Tdet 22Tw
            % 23总乘客sumP 24总货物sumC 25客仓座位 26货仓容量
            % 27Ctoll 28Cidle 29Ckm 30Cfix 31Eu 32Ef
            % 33客运能力 34货运能力
            nowT_Bus(noB)=ET; % 更新当前时间

        end
        %timelist(S(no),6)=1; % 该行程已执行
        if flag==1
            timelist(S(no),6)=1;
        end
        
        %timelist(S(no),6)=1;
        %recording=[recording;tempR];
        %break;
        
        no=no+1;
        if flag==1 || no>length(timelist(:,1)) % 找到了匹配的行程\no大于TimeList第一列长度（即处理完所有行程），退出循环
            recording=[recording;tempR]; 
            break
        end
    end
    
    if sum(demand(:,6))==0 % 所有需求都已满足，退出循环
        disp(1);
        break;
    end   
end

% 1公交车编号，2线路编号 3出发节点 4抵达节点 5装卸类型 6装卸数量 7出发时间 8抵达时间 9离开时间 10等待时间 
% 11装卸时间 12出发时SOC 13抵达时SOC 14离开时SOC 15路途时间 16往返 
% 17车型y 18客仓占比λ 19Tcru 20Tdwell 21Tdet 22Tw
% 23总乘客sumP 24总货物sumC 25客仓座位 26货仓容量
% 27Ctoll 28Cidle 29Ckm 30Cfix 31Eu 32Ef
% 33客运能力 34货运能力
%fit=Tcru+Tdwell+Tdet+Tw;
%Cfix=3211.08;
Z=Ctoll+Cidle+Ckm+Cfix-Eu-Ef; %上层模型Z
T=Tcru+Tdwell+Tdet+Tw; %下层模型T
fit=wz*Z+wt*T; % 用熵权法转化为单目标优化

% 处理约束，施加惩罚
penalty=100000000000; % 惩罚数
meanT=(Tcru+Tdwell+Tdet+Tw)/sumP; % 乘客人均时间
Flag=0; % 是否违反约束
% 约束条件：人均时间<=时间约束；客运能力>=客运总需求；货运能力>=货运总需求
% || minLambda>nowMinLambda || 
if meanT>maxT || sumDemandOfP>sumPCapacity || sumDemandOfC>sumCCapacity 
    fit=fit+penalty;  % 施加惩罚
    %Flag=1;
end
%disp(sumPCapacity)
%if sumDemandOfP<=sumPCapacity
%    disp(1)
%end

if nargout>1 %返回总成本与行程记录详细数据
    result.fit=fit; 
    result.recording=recording; 
end
end