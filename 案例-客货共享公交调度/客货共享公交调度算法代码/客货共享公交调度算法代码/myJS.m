function [bestY,bestX,recording]=myJS(x,y,option,data)
  
%% JS算法
    SearchAgents_no=option.numAgent;
    Max_iter=option.maxIteration;
    lb=option.lb;
    ub=option.ub;
    dim=option.dim;
    fobj=option.fobj;

    [Alpha_pos,Alpha_score,Convergence_curve,recording]=js(x,fobj,lb,ub,dim,[Max_iter,SearchAgents_no]);
    %recording.bestFit=[min(y),Convergence_curve];

    [y_g,position]=min(y);
    x_g=x(position(1),:);
    y_p=y;
    x_p=x;
    recording.bestFit(1)=y_g;
    recording.meanFit(1)=mean(y_p);

    bestY=Alpha_score;
    bestX=Alpha_pos;
end