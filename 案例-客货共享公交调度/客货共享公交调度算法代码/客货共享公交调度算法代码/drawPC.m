function drawPc(str,result,data,option)
disp(str)
recording=result.recording;
% 1公交车编号，2线路编号 3出发节点 4抵达节点 5装卸类型 6装卸数量 7出发时间 8抵达时间 9离开时间 10等待时间 
% 11装卸时间 12出发时SOC 13抵达时SOC 14离开时SOC 15路途时间 16往返 
% 17车型y 18客仓占比λ 19Tcru 20Tdwell 21Tdet 22Tw
% 23总乘客sumP 24总货物sumC 25客仓座位 26货仓容量
% 27Ctoll 28Cidle 29Ckm 30Cfix 31Eu 32Ef
% 33客运能力 34货运能力

for noB=1:data.numBus % 公交编号从1-numBus
    p=find(recording(:,1)==noB & recording(:,16)==1);
    if ~isempty(p)
        noL=recording(p(1),2); % 路线
        busType=recording(p(1),17); % 车型
        Lambda=recording(p(1),18); % 客仓占比
        PCapacity=recording(p(1),25); 
        disp(['公交车',num2str(noB),' 车型(',num2str(busType),') 客仓占比(',num2str(Lambda*100,'%.2f'),'%) 座位数(',num2str(PCapacity),')']);

        %disp(num2str(recording(p(1),21)));

        %strPath=['出发路线(路线',num2str(noL),')：'];
        %for i=1:length(p)
        %    nowP=recording(p(i),3);
         %   nextP=recording(p(i),4);
        %    strPath=[strPath,num2str(nowP),'->'];
        %    %disp([num2str(recording(p(i),19))]);
        %end
        %ST=sprintf('%02d:%02d', floor(recording(p(1),7)), round(mod(recording(p(1),7),1)*60));
        %ET=sprintf('%02d:%02d', floor(recording(p(length(p)),9)), round(mod(recording(p(length(p)),9),1)*60));
        %strPath=[strPath,num2str(nextP),' 时间：',ST,'-',ET];
        %disp(strPath)
        %p=find(recording(:,1)==noB & recording(:,16)==2);
        %noL=recording(p(1),2);
        %strPath=['返回路线(路线',num2str(noL),')：'];
        %flag=0;
        %for i=1:length(p)
        %    nowP=recording(p(i),3);
        %    nextP=recording(p(i),4);
        %    strPath=[strPath,num2str(nowP),'->'];
        %    flag=1;
        %end
        %if flag==1
        %    ST=sprintf('%02d:%02d', floor(recording(p(1),7)), round(mod(recording(p(1),7),1)*60));
        %    ET=sprintf('%02d:%02d', floor(recording(p(length(p)),9)), round(mod(recording(p(length(p)),9),1)*60));
        %    strPath=[strPath,num2str(nextP),' 时间：',ST,'-',ET];
        %else 
        %    strPath=['无'];
        %end
        %disp(strPath)
        
    end
end
Tcru=recording(end,19); % 总乘客乘车时间
Tdwell=recording(end, 20); % 总装卸耽误时间
Tdet=recording(end, 21); % 总乘客滞留时间
Tw=recording(end, 22); % 总乘客等待时间
T=Tcru+Tdwell+Tdet+Tw; % 乘客出行总时间
wt=data.wt; %上层模型T的熵权法权重
sumP=recording(end,23); % 总乘客
sumC=recording(end,24); % 总货物
meanT=T/sumP; % 乘客人均时间


Ctoll=recording(end,27); % 道路通行费
Cidle=recording(end,28); % 站点停滞总成本
Ckm=recording(end,29); % 线路运行成本
Cfix=recording(end,30); % 车队配置成本
Eu=recording(end,31); % 客运总收入
Ef=recording(end,32); % 货运总收入
Z=Ctoll+Cidle+Ckm+Cfix-Eu-Ef; % 公交总运营成本
wz=data.wz; %上层模型Z的熵权法权重
sumPCapacity=recording(end,33); % 客运能力
sumCCapacity=recording(end,34); % 货运能力


fit=wz*Z+wt*T; % 目标函数
disp(['目标函数：',num2str(fit,'%.2f')])
disp(['利润:',num2str(-Z,'%.2f'),'元'])
disp(['Ctoll:',num2str(Ctoll),'元 Cidle:',num2str(Cidle,'%.2f'),'元 Ckm:',num2str(Ckm,'%.2f'),'元 Cfix:',num2str(Cfix),'元 Eu:',num2str(Eu,'%.2f'),'元 Ef:',num2str(Ef,'%.2f'),'元'])
disp(['乘客出行总时间:',num2str(T,'%.2f'),'h'])
disp(['乘客人均时间:',num2str(meanT*60,'%.2f'),'min 约束时间:',num2str(data.maxT*60,'%.2f'),'min']);
disp(['Tcru:',num2str(Tcru/sumP*60,'%.2f'),'min Tdwell:',num2str(Tdwell/sumP*60,'%.2f'),'min Tw:',num2str(Tw/sumP*60,'%.2f'),'min Tdet:',num2str(Tdet/sumP*60,'%.2f'),'min'])
disp(['总乘客:',num2str(sumP),' 总货物:',num2str(sumC),' 公交客运能力:',num2str(sumPCapacity),' 公交货运能力:',num2str(sumCCapacity)])

end