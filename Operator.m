function [Offspring1,Offspring2,Offspring3] = Operator(Loser1,Loser2,Winner)
    % The competitive swarm optimizer of LMOCSO

    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    % for evolutionary multi-objective optimization [educational forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------

    %% Parameter setting
    Loser1Dec  = Loser1.decs;
    Loser2Dec  = Loser2.decs;
    WinnerDec = Winner.decs;
    [N,D]     = size(Loser1Dec);
    Loser1Vel  = Loser1.adds(zeros(N,D));
    Loser2Vel  = Loser2.adds(zeros(N,D));
    WinnerVel = Winner.adds(zeros(N,D));
    [N,D]     = size(Loser1.decs);
    Problem   = PROBLEM.Current();

    [WinOffVel, WinOffDec] = DV_Func(Winner,Winner,Problem);
    if rand > 0.35
        [Loser1OffVel, Loser1OffDec] = CV_Func(Loser1,Winner,Problem);
    else
        [Loser1OffVel, Loser1OffDec] = DV_Func(Loser1, Winner,Problem);
    end
    [Loser2OffVel, Loser2OffDec] = CV_Func(Loser2,Winner,Problem);

    %% Add the winners
    Loser1OffDec = [Loser1OffDec;WinnerDec];
    Loser1OffVel = [Loser1OffVel;WinnerVel];

    Loser2OffDec = [Loser2OffDec;WinnerDec];
    Loser2OffVel = [Loser2OffVel;WinnerVel];

    WinOffDec = [WinOffDec;WinnerDec];
    WinOffVel = [WinOffVel;WinnerVel];
    
    Offspring1 = Polynomial_mutation(Problem,Loser1OffDec,Loser1OffVel,N,D);
    Offspring2 = Polynomial_mutation(Problem,Loser2OffDec,Loser2OffVel,N,D);
    Offspring3 = Polynomial_mutation(Problem,WinOffDec,WinOffVel,N,D);
end

function [OutVel, OutDec] = CV_Func(Loser, Winner,Problem)
    LoserDec = Loser.decs;
    [N,D]     = size(LoserDec);
    LoserVel = Loser.adds(zeros(N,D));
    WinnerDec = Winner.decs;
    if size(LoserVel, 1) == 0
        OutVel = LoserVel;
        OutDec = LoserDec;
        return;
    end

    c1 = 0.125
    r1     = repmat(rand(N,1),1,D);
    r2     = repmat(rand(N,1),1,D);
    r3     = repmat(rand(N,1),1,D);

    OutVel = r1.*LoserVel + r2.*(WinnerDec-LoserDec);
    OutDec = LoserDec + OutVel;
    OutDec = OutDec + c1*r3.*(WinnerDec-OutDec);

end

function [OutVel, OutDec] = DV_Func(Loser, Winner,Problem)
    LoserDec = Loser.decs;
    [N,D]     = size(LoserDec);
    LoserVel = Loser.adds(zeros(N,D));
    WinnerDec = Winner.decs;
    if size(LoserVel, 1) == 0
        OutVel = LoserVel;
        OutDec = LoserDec;
        return;
    end

    LoserObjs = Loser.objs;
    [FrontNo,MaxFNo] = NDSort(LoserObjs,inf);
    
    for i = 1:MaxFNo
        tmpNo = find(i==FrontNo);
        dis = pdist2(LoserObjs(tmpNo,:), LoserObjs(tmpNo,:));
        dis(dis==0)=inf;
        [~, tmpIndex] = min(dis);
        FrontNo(tmpNo) = tmpIndex;
    end

    TmpDec = repmat(LoserDec(FrontNo,:), 1, 1);
    
    c1=1.3;
    c2=1.2;
    c3=0.13;
    c4=1.28;

    r1     = repmat(rand(N,1),1,D);
    r2     = repmat(rand(N,1),1,D);
    r3     = repmat(rand(N,1),1,D);
    max_tmp = max(LoserDec+0.5,[],2);
    min_tmp = min(LoserDec.*0.1,[],2);
    max_tmp = min(max_tmp,Problem.upper);
    min_tmp = max(min_tmp,Problem.lower);
    
    xr = unifrnd(repmat(min_tmp,1,1),repmat(max_tmp,1,1));
    OffVel = c1*r1.*LoserVel + c2*r2.*(WinnerDec-LoserDec) + c3.*( xr - LoserDec)+c4*r3.*(LoserDec-TmpDec);
    OffDec = LoserDec + OffVel;
    
    OutDec = OffDec;
    OutVel = OffVel;
end