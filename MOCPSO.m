classdef MOCPSO < ALGORITHM
% <multi> <real/binary/permutation> <constrained/none>
% MOCPSO: A Multi-Objective Cooperative Particle Swarm Optimization Algorithm with Dual Search Strategies
% Zhang, Yan and Li, Bingdong and Hong, Wenjing and Zhou, Aimin
%
%------------------------------- Reference --------------------------------
% Zhang, Y., Li, B., Hong, W., & Zhou, A.
% "MOCPSO: A multi-objective cooperative particle swarm optimization algorithm 
% with dual search strategies." 
% Neurocomputing 562 (2023): 126892.
% https://github.com/ilog-ecnu/MOCPSO.git
%
%------------------------------- Copyright --------------------------------
% Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
methods
    function main(Algorithm,Problem)
        %% Generate random population
        [V,Problem.N] = UniformPoint(Problem.N,Problem.M);
        Population = Problem.Initialization();
        Population = EnvironmentalSelection(Population,V,(Problem.FE/Problem.maxFE)^2);

        % Optimization
        while Algorithm.NotTerminated(Population)
            if size(Population,2) < 3
                disp(size(Population));
                [N,D]     = size(Population.decs);
                PopVel  = Population.adds(zeros(N,D));
                Offspring = Polynomial_mutation(Problem,Population.decs,PopVel,N/2,D);
                Population = EnvironmentalSelection([Population, Offspring],V,(Problem.FE/Problem.maxFE)^2);
                continue;
            end

            FitValue = calFitness(Population.objs);
            Rank = randperm(length(Population),floor((length(Population))/3)*3);
            Loser1 = Rank(1:end/3);
            Loser2 = Rank(end/3+1:end/3*2);
            Winner = Rank(end/3*2+1:end);
            [Loser1, Loser2] = swapWL(Loser1,Loser2,FitValue);
            [Winner, Loser1] = swapWL(Winner,Loser1,FitValue);
            [Loser1, Loser2] = swapWL(Loser1,Loser2,FitValue);
            [Offspring1,Offspring2, Offspring3]      = Operator(Population(Loser1),Population(Loser2),Population(Winner));
            Population     = EnvironmentalSelection([Population,Offspring1,Offspring2,Offspring3],V,(Problem.FE/Problem.maxFE)^2);
        end
    end
end
end

function [Winner,Loser] = swapWL(Winner, Loser,FitValue)
    Change = FitValue(Loser) >= FitValue(Winner);
    Temp   = Winner(Change);
    Winner(Change) = Loser(Change);
    Loser(Change)  = Temp;
end