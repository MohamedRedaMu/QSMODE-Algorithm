function [goalReached, GlobalBest, countFE, Q, replay_memory] = QSMODE_algorithm()

    %% *********************************** Citation **********************
    %{
    Citation (Please cite the following paper paper):
        If you use this code in your research, please cite the following paper:
        %% Bibtex:
        @article{reda2025novel,
          title={A novel reinforcement learning-based multi-operator differential evolution with cubic spline for the path planning problem},
          author={Reda, Mohamed and Onsy, Ahmed and Haikal, Amira Y and Ghanbari, Ali},
          journal={Artificial Intelligence Review},
          volume={58},
          number={5},
          pages={142},
          year={2025},
          publisher={Springer}
        }
        %% APA Style
        Reda, M., Onsy, A., Haikal, A. Y., & Ghanbari, A. (2025). A novel reinforcement learning-based multi-operator differential evolution with cubic spline for the path planning problem. Artificial Intelligence Review, 58(5), 142.
    
        %% Chicago/Turabian Style
        Reda, Mohamed, Ahmed Onsy, Amira Y. Haikal, and Ali Ghanbari. "A novel reinforcement learning-based multi-operator differential evolution with cubic spline for the path planning problem." Artificial Intelligence Review 58, no. 5 (2025): 142.        
    
    Code citation:
        If you use this code in your research, please cite the following code as follows:

        Mohamed Reda. (2025). MATLAB implementation of the Q-Spline Multi-Operator Differential Evolution (QSMODE) Algorithm. Available at: https://github.com/MohamedRedaMu/QSMODE-Algorithm
    
        Mohamed Reda. (2025). MATLAB implementation of the Reda Path Planning Benchmark Library 2024 (RP2B24). Available at: https://github.com/MohamedRedaMu/RP2B24-Benchmark

    Contact Information:
        For questions or further information, please contact:
        Author Name:
            Mohamed Reda
    
        Affiliation:
            1- School of Engineering, University of Central Lancashire, Preston, PR1 2HE, UK
            2- Computers and Control Systems Engineering Department, Faculty of Engineering, Mansoura University, Mansoura, 35516, Egypt
    
        Emails:
            mohamed.reda.mu@gmail.com;
            mohamed.reda@mans.edu.eg;
            mramohamed@uclan.ac.uk

    %}

    %% Begin intialiazation
    Q = [] ; % for reinforecement learning 
    replay_memory = [] ;  % for reinforecement learning 

    %% Model paramters
    modelType = 4;
    modelNum = 8; 
    boundaries.xmin = -10;
    boundaries.xmax = 10;
    boundaries.ymin = -10;
    boundaries.ymax = 10;
    fNo = 1; 
    model = CreateModel(modelType, modelNum, boundaries, fNo);

    %% Cost Function
    beta_penalty = 100; % stepness factor for the penalty
    costFunction = @(sol, model, nSplinePoints, beta) MyCost_linearSharpPenalty(sol, model, nSplinePoints, beta);
      
    %% Algorithm paramters
    AlgName = 'QSMODE Algorithm' ;
    n_opr = 3 ; % number of operators 
    arch_rate = 2.6; % 2.6 (A) rate of the archive control the size of it
    mem_size_scale = 20 ; % memory size (H) = 20 * nd, where 20 is the memeory scale
    init_CR = 0.2; % Initial CR
    init_F = 0.2; % initial F
    maxfe_ls_ratio = 0.85; % ration at which the local search starts (after 85% of maxfes)
    max_ls_iterations_rate = 20.0000e-003; % CFE_ls max iterations for ls (ration of maxfe)
    prob_ls=0.1;  % porabibility of quadratic programming optimization using fmincon
    prob_ls_min = 0.01 ; % 0.0001 in the paper 0.01 in code set this min probability if the LS not improving the global vest
    ls_optimizar_name = 'sqp' ; % sequantial quadratic programming 'sqp' or  'active-set' or 'trust-region-reflective' or 'interior-point'
    MinPopSize = 6 ; % min pop size (linear reduction for population)
    nPopRate = 12 ; % initial population is set to nPopRate * nd * nd (not used here)

    isSpline = true ; % smooth path planning
    nSplinePoints = 100; % number of way points in the spline curve
    nWayPointsRate = 1 ; % no of way points = no of dim = no of obstacles * nWayPointsRate

    is_RL_activated = true; 
    num_actions = 4; % number of actions (angles) to move from obe state to another
    learning_rate = 0.6;
    discount_factor = 0.9;
    RL_exploration_prob = 0.1;
    RL_exploration_decay = 0.995;
    RL_min_exploration_prob = 0.01;
    gridResolution = 0.15; % less than this needs more iteration , more time, goal may not be gauranteed
    maxSteps = 500;  % to update the q table (pure q learning)
    maxCount = 10 ; % to construct the path from the q table

    %% dim size and number of way points and no of obstacles
    nd = max(round( model.n * nWayPointsRate),1); 

    %% Q learning intialization 
    %% get the set of actiosn (angles) based on the number of actions
    action_angles = linspace(0, 360, num_actions + 1);
    action_angles = action_angles(1:end-1); % Remove the last element to avoid duplicating 0 and 360 degree


    %% Create a grid map based on the model
    gridMap = createGridMap(model, gridResolution);

    %% intialize the Q table  based on the grid and the number of actions
    Q = zeros(size(gridMap, 1), size(gridMap, 2), num_actions);

    %% start and end points
    start_state = round([(model.xs - model.xmin) / gridResolution, (model.ys - model.ymin) / gridResolution]) + 1;
    goal_state = round([(model.xt - model.xmin) / gridResolution, (model.yt - model.ymin) / gridResolution]) + 1;
    
    %% intial pop size 
    nPop = nPopRate * nd  ; % Initial Population Size
    InitPop = nPop ;
    % prob. of each DE operator
    probDE1= 1./n_opr .* ones(1,n_opr);

    %% Initialize Archive Data 
    archive.NP = arch_rate * nPop; % the maximum size of the archive
    cand = create_empty_individual(); 
    cand.Position.x = zeros(1, nd); % empty solution of nd dim
    cand.Position.y = zeros(1, nd); % empty solution of nd dim
    cand.Cost = inf(1, 1); % empty cost value
    archPop = repmat(cand, 1, 1);  % popualtion of one individual
    archive.pop = archPop; 

    %% Initialize Adaptive Archive for CR and F
    hist_pos=1;
    memory_size=mem_size_scale*nd;
    archive_f = ones(1,memory_size).* init_F;
    archive_Cr = ones(1,memory_size).* init_CR;

    %% Stopping criteria
    tol = 10^-8;
    max_iterations = 1000; 
    deltaThreshold = 1e-5;  % Small threshold value for the change in solution quality
    consecIterationsThreshold = 500;  % Number of consecutive iterations with minimal improvement
    
    %% Display iteration prompt
    print_flag = true;

    %%  Global variable to count number of function evaluations
    global countFE;
    countFE = 0 ;

    %% Initialize iteration counter 
    N_iter = 0;

    %% Goal reached flag
    goalReached = false; 

    %% Initialize Global best
    GlobalBest = initializeGlobalBest(); 

    %% Set the seed for random number generator
    rng('default');  % Resets to the default settings
    rng('shuffle'); % set it to shuffle
    
    %% Initialize population and update the global best
    population = repmat(create_empty_individual(), nPop, 1);

    % Generate initial positions using Latin Hypercube Sampling
    for i = 1:nPop
        % Initialize Position
        if i == 1 % For the first nest, initialize with a straight-line path
            x = linspace(model.xs, model.xt, nd+2);
            y = linspace(model.ys, model.yt, nd+2);
            population(i).Position.x = x(2:end-1);
            population(i).Position.y = y(2:end-1);
        else
            % generate random path withim the range (min, max) 
            population(i).Position.x = (model.xmax - model.xmin) * rand(1, nd) + model.xmin;
            population(i).Position.y = (model.ymax - model.ymin) * rand(1, nd) + model.ymin;
        end
        
        % Evaluation
        [population(i).Cost, population(i).Sol] = costFunction(population(i).Position, model, nSplinePoints, beta_penalty);

        % Update Global Best
        if i == 1 || population(i).Cost < GlobalBest.Cost
            GlobalBest = population(i);
        end
    end

    % Initial Error (Global Best Cost)
    InitError = GlobalBest.Cost ; % used for population size reduction
    % create a random permuatation of the popualtion
    pop_old = population(randperm(nPop),:);

   %% begin algorithm loop 
   consecIterationsCount = 0;  % Initialize the consecutive iteration count
   previousGlobalBestCost = GlobalBest.Cost;  % Store the initial global best cost
   tolReached = consecIterationsCount >= consecIterationsThreshold;
   
   while (~tolReached)  && (N_iter <= max_iterations)
        %% update the generation
        N_iter=N_iter+1; 

        %% Update popuation size
        FERate = (N_iter / max_iterations) ; 
        newPopSize= round(MinPopSize + ((InitPop - MinPopSize) * (1 - ( FERate).^1)));

        %% Update the popuation according to the new popuation size
        nPop = numel(population); % current popsize 
        if nPop > newPopSize
            % Calculate the number of individuals to remove
            reduction_ind_num = nPop - newPopSize;
            if nPop - reduction_ind_num < MinPopSize
                reduction_ind_num = nPop - MinPopSize;
            end
        
            % Remove the worst individuals
            for r = 1 : reduction_ind_num
                % Sort population based on Cost
                [~, sortedIdx] = sort([population.Cost], 'descend');
                % Remove the worst individual
                population(sortedIdx(1)) = []; % it removed from the original popualtion
            end

            % update the current popsize
            nPop = numel(population);
        
            %% Update archive size based on the new population size
            archive.NP = round(arch_rate * nPop);

            % If archive size exceeds its limit, randomly remove some individuals
            current_archive_NP = numel(archive.pop);
            if current_archive_NP > archive.NP
                rndpos = randperm(current_archive_NP);
                rndpos = rndpos(1 : archive.NP);
                archive.pop = archive.pop(rndpos);
            end
        end

        %% Initialize the archive of the CR and F
        mem_rand_index = ceil(memory_size * rand(nPop, 1));
        mu_sf = archive_f(mem_rand_index);
        mu_cr = archive_Cr(mem_rand_index);
        
        %%  generate CR   
        cr = normrnd(mu_cr, 0.1);
        term_pos = find(mu_cr == -1);
        cr(term_pos) = 0;
        cr = min(cr, 1);
        cr = max(cr, 0);
        % sort the cr
        [cr,~]=sort(cr);

        %% for generating scaling factor
        F = mu_sf + 0.1 * tan(pi * (rand(1,nPop) - 0.5));
        pos = find(F <= 0);
        
        while ~ isempty(pos)
            F(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(1,length(pos)) - 0.5));
            pos = find(F <= 0);
        end
        
        F = min(F, 1);
        F=F';

         %% Sort the popuation     
        Costs = [population.Cost]; % original costs of the original popuation
        [Costs, SortOrder] = sort(Costs);
        population = population(SortOrder);  

        %% **** Mutation Phase ****
        % combine the popuation with the archive population  
        popAll = [population; archive.pop];  
        
        %% generate mutation operator probablities for each individual in the population
        % Randomly decide the mutation strategy for each individual
        bb = rand(nPop, 1);
        
        % Retrieve probabilities for each strategy
        probiter = probDE1(1, :);
        l2 = sum(probDE1(1:2));
        % Determine which strategy to apply for each individual
        op_1 = bb <= probiter(1) * ones(nPop, 1);
        op_2 = (bb > probiter(1)) & (bb <= l2);
        op_3 = (bb > l2) & (bb <= 1);

        %% generate random integer numbers
        r0 = 1 : nPop;
        [r1, r2,r3] = gnR1R2(nPop, size(popAll, 1), r0);

        %% Choose top individuals (at least one) for DE operator 1 and 2
        pNP12 = max(round(0.25 * nPop), 1); % At least one or 25% of the population size
        randindex = ceil(rand(1, nPop) .* pNP12); % Select indices from the best subset
        randindex = max(1, randindex); % Ensuring indices are valid (not less than 1)
        phix12 = population(randindex, :);
        
        %%  Choose top individuals (at least two) for DE operator 3
        pNP3 = max(round(0.5 * nPop), 2); %% choose at least two best solutions
        randindex = ceil(rand(1, nPop) .* pNP3); %% select from [1, 2, 3, ..., pNP]
        randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
        phix3 = population(randindex, :);

        %% Initialize mutation vector
        cand = create_empty_individual();
        cand.Position.x = zeros(1, nd);
        cand.Position.y = zeros(1, nd);
        newPop = repmat(cand, nPop, 1);

        for i = 1:nPop
            % apply mutation rule
            x_curr = population(i).Position;
            x_r1 = population(r1(i)).Position;
            x_r3 = population(r3(i)).Position;
            xx_r2 = popAll(r2(i)).Position;
            x_phi12 = phix12(i).Position;
            x_phi3 = phix3(i).Position;

            if op_1(i)
                % Strategy 1: Mutation based on the difference between two individuals
                newPop(i).Position.x = x_curr.x + F(i) * (x_phi12.x - x_curr.x + x_r1.x - xx_r2.x);
                newPop(i).Position.y = x_curr.y + F(i) * (x_phi12.y - x_curr.y + x_r1.y - xx_r2.y);
            elseif op_2(i)
                % Strategy 2 : Mutation using three random individuals
                newPop(i).Position.x = x_curr.x + F(i) * (x_phi12.x - x_curr.x + x_r1.x - x_r3.x);
                newPop(i).Position.y = x_curr.y + F(i) * (x_phi12.y - x_curr.y + x_r1.y - x_r3.y);

            elseif op_3(i)
                % Strategy 3 (DE3)
                newPop(i).Position.x = F(i) * x_r1.x + F(i) * (x_phi3.x - x_r3.x);
                newPop(i).Position.y = F(i) * x_r1.y + F(i) * (x_phi3.y - x_r3.y);

            end
             % Applying boundary check for each individual
             newPop(i).Position.x = han_boun_individual(newPop(i).Position.x, model.xmin, model.xmax , x_curr.x);
             newPop(i).Position.y = han_boun_individual(newPop(i).Position.y, model.ymin, model.ymax , x_curr.y);
        end

        %% *** Crossover ***
        % Initialize ui as an array of individuals
        cand = create_empty_individual();
        cand.Position.x = zeros(1, nd);
        cand.Position.y = zeros(1, nd);
        newPop2 = repmat(cand, nPop, 1);

        for i = 1:nPop
            % Generate a random number to decide the crossover method
            if rand < 0.4
                % Binomial Crossover
                mask = rand(1, nd) > cr(i);
                jrand = floor(rand * nd) + 1; % Ensure at least one dimension is inherited from vi
                mask(jrand) = false;
                newPop2(i).Position = population(i).Position; % Start with parent position from origial population
                
                newPop2(i).Position.x(~mask) = newPop(i).Position.x(~mask); % Inherit from newPop where mask is false
                newPop2(i).Position.y(~mask) = newPop(i).Position.y(~mask); % Inherit from newPop where mask is false
            else
                % Exponential Crossover
                startLoc = randi(nd);
                L = 0;
                while (rand < cr(i) && L < nd)
                    L = L + 1;
                end
                idx = mod(startLoc-1:startLoc+L-2, nd) + 1; % Ensuring wrapping around dimensions
                newPop2(i).Position = population(i).Position; % Start with parent position
                newPop2(i).Position.x(idx) = newPop(i).Position.x(idx); % Inherit from vi for selected indices
                newPop2(i).Position.y(idx) = newPop(i).Position.y(idx); 
            end

            %% evaluate the new cost
            [newPop2(i).Cost, newPop2(i).Sol] = costFunction(newPop2(i).Position, model, nSplinePoints, beta_penalty);
        end

        %% *** Update the archives ****
        %% get the I label for the improved individuals
        newCosts = [newPop2.Cost] ;
        I = (newCosts < Costs ); % Logical index of improved solutions

        %% update the archive with the old bad solutions in the population
        archive = updateArchive_basic(archive, population(I == 1)) ; 

        %% update probDE1 (operators probabilities). of each DE
        diff2 = max(0, (Costs - newCosts))./abs(Costs + eps); % Improvement metric, adding eps for stability

        % Calculate performance scores for this iteration
        count_S = zeros(1, n_opr);
        count_S(1)=max(0,mean(diff2(op_1==1)));
        count_S(2)=max(0,mean(diff2(op_2==1)));
        count_S(3)=max(0,mean(diff2(op_3==1)));

        % Check if there is any significant improvement across all operators
        if all(count_S <= eps) % If no significant improvement, reset to equal probabilities
            probDE1 = ones(1, n_opr) / n_opr;
        else
            % Optionally enforce minimum and maximum probabilities (e.g., between 0.1 and 0.9)
            % This step is typically not necessary right after softmax, but included for completeness
            probDE1 = max(0.1, min(0.9, probDE1));

        end
        %% calc. imprv. for Cr and F archives
        goodCR = cr(I == 1);
        goodF = F(I == 1);
        diff = abs(Costs - newCosts);
        if size(goodF,1)==1
            goodF=goodF';
        end
        if size(goodCR,1)==1
            goodCR=goodCR';
        end
        num_success_params = numel(goodCR);
        if num_success_params > 0
            weightsDE = diff(I == 1)./ sum(diff(I == 1));
            %% for updating the memory of scaling factor
            archive_f(hist_pos) = (weightsDE * (goodF .^ 2))./ (weightsDE * goodF);
            
            %% for updating the memory of crossover rate
            if max(goodCR) == 0 || archive_Cr(hist_pos)  == -1
                archive_Cr(hist_pos)  = -1;
            else
                archive_Cr(hist_pos) = (weightsDE * (goodCR .^ 2)) / (weightsDE * goodCR);
            end
            
            hist_pos= hist_pos+1;
            if hist_pos > memory_size;  hist_pos = 1; end
        else
            archive_Cr(hist_pos)=0.5;
            archive_f(hist_pos)=0.5;
        end

        %% update population with the good solution and update the oldPop with the bad old solutions
        pop_old(I == 1) = population(I == 1); %save the bad individual in popualtion in the old popuation
        population(I == 1) = newPop2(I == 1); % relace the bad individuals in popuation with the better individuals in newPop2

        %% sort the population and old population
        [~, sortedIndices] = sort([population.Cost]); % the new updated merged costs
        population = population(sortedIndices);
        pop_old = pop_old(sortedIndices);

        %% update global best
        localBest = population(1); % the sorted population, the top solution is the best cost solution
        if localBest.Cost < GlobalBest.Cost
            goalReached = true; % we guarnatee a path found
            GlobalBest = localBest;
        end


        %% *** Local Search ***
        if N_iter > maxfe_ls_ratio * max_iterations && N_iter<max_iterations
            if rand<prob_ls
                % set the maxfes for the fmincon algorithm (small ratio of the original maxfes)
                LS_maxStop = min(ceil(max_ls_iterations_rate * max_iterations), (max_iterations - N_iter)) ; 
                [GlobalBest, succ] = LS2_pathPlanning(GlobalBest, model,N_iter, ls_optimizar_name, LS_maxStop, isSpline, nSplinePoints, nd, costFunction, beta_penalty );

                if succ==1 %% if LS2 was successful
                    % replace the worst solution in the popualtion, with the
                    % new global best, because the new global best is not in  the orgignial population
                    population(nPop) = GlobalBest;
                    goalReached = true ; % we guantee a path found in case of success

                    % sort the population and old population
                    [~, sortedIndices] = sort([population.Cost]); % the new updated merged costs
                    population = population(sortedIndices);
                    pop_old = pop_old(sortedIndices);
    
                    prob_ls = prob_ls;
                else
                    prob_ls=prob_ls_min; %% set p_LS to a small value it  LS was not successful
                end    
            end
        end

        %% *** Q learning update ****
        if  is_RL_activated
            %% Update exploration probability
            RL_exploration_prob = RL_exploration_prob * RL_exploration_decay;
            RL_exploration_prob = max(RL_exploration_prob, RL_min_exploration_prob);
    
            if rand() < RL_exploration_prob
                %% update based on pure q learning
                [Q] = UpdateQLearning_basedOn_PureQLearning(Q, model, gridResolution, gridMap, action_angles, learning_rate, discount_factor, costFunction, beta_penalty, maxSteps, RL_exploration_prob);
    
            else
                 %% update the Q table based in popuation( and the histpy of expanded (updated) current states points)
                 for i = 1:nPop 
                     %% update the Q -table
                    [Q]  = UpdateQLearning_basedOnPopulation(Q, population(i), model, gridResolution,gridMap, action_angles, learning_rate, discount_factor, costFunction, beta_penalty);
                 end
    
            end
        
             %% Q learning path generation
            % Parameters for path generation
            nWayPoints = nd;  % Or any specific number you desire
            
            % Generate the new path from the Q-table
            newPath = GeneratePathFromQ(Q, start_state, goal_state, gridResolution, model, nWayPoints, num_actions, maxCount);
            if ~isempty(newPath.Position.x) 
                goalReached = true ;
              % Re-evaluate the new solution's cost
               [newPath.Cost, newPath.Sol] = costFunction(newPath.Position, model, nSplinePoints, beta_penalty);
            
                 % Assuming 'population' and 'costs' are already defined
                [~, worstIndex] = max([population.Cost]);  % Assuming each individual in population has a 'Cost' field
                 worsSol = population(worstIndex); 
        
                % replace the worst individual in the popuation
                if newPath.Cost < worsSol.Cost  
                    GlobalBest = newPath;
                    population(worstIndex) = newPath;
                end
            end
        end

        %% check if maxfes is exceeded 
        if N_iter > max_iterations 
            break;
        end

        %% Check the improvement in the global best cost
        bestCostChange = abs(previousGlobalBestCost - GlobalBest.Cost);
        
        % Update the count of consecutive iterations with minimal improvement
        if bestCostChange < deltaThreshold
            consecIterationsCount = consecIterationsCount + 1;
        else
            consecIterationsCount = 0;  % Reset if there is significant improvement
        end
        
        % Update the previous global best cost for the next iteration
        previousGlobalBestCost = GlobalBest.Cost;

        tolReached = consecIterationsCount >= consecIterationsThreshold;

        %% print the iteration number
        if print_flag            
            fprintf('%s | Model_Type%d_Config%d | Iteration %d | nPop %d | FEs %d | Error %d\n', AlgName, model.modelId, model.modelNum , N_iter, nPop, countFE, GlobalBest.Cost);
        end
   end % end of main loop
end

function individual = create_empty_individual()
    individual.Position.x = [];
    individual.Position.y = [];
    individual.Cost = Inf;
    individual.Sol = [];
end



function archive = updateArchive_basic(archive, newPop)
    if archive.NP == 0, return; end

     % Combine existing archive population with new population
     combinedPop = [archive.pop; newPop];

    % Randomly remove solutions if necessary to maintain archive size
    nA = numel(combinedPop); 
    Asize = archive.NP;
    if nA > Asize

        combinedPop = combinedPop(floor(nA-Asize+1):nA);     
    end
    % return the uprated archive
    archive.pop = combinedPop;
end

function [GlobalBest, succ] = LS2_pathPlanning(GlobalBest, model, iter, ls_optimizar_name, LS_maxStop, isSpline, nSplinePoints, nd, costFunction, beta_penalty)
    global countFE; 
    %newIndividual = initializeGlobalBest(); % create intial global best
    % Combine x and y from the position to create a vector for optimization
    initialPosition = [GlobalBest.Position.x, GlobalBest.Position.y];

    % set the upper and lower bounds in the size of (nd * 2)
    ub = [repmat(model.xmax, 1, nd), repmat(model.ymax, 1, nd)];  % Upper bounds for x and y
    lb = [repmat(model.xmin, 1, nd), repmat(model.ymin, 1, nd)];  % Lower bounds for x and y


    % Set the option for the optimizer
    options = optimset('Display', 'off', 'algorithm', ls_optimizar_name, 'UseParallel', 'never', 'MaxFunEvals', LS_maxStop);

    % Adapted to use the new cost function format
    costFuncWrapper = @(solVec) WrapperCost(solVec, model);

    % run the algorithm
    [newPositionVec, newCost, ~, details] = fmincon(costFuncWrapper, initialPosition, [], [], [], [], lb, ub, [], options);

    %unpack the solution to x and y
    halfLen = length(newPositionVec) / 2;
    newIndividual.Position.x = newPositionVec(1:halfLen);
    newIndividual.Position.y = newPositionVec(halfLen+1:end);

    % evaualte 
    [newIndividual.Cost, newIndividual.Sol, newIndividual.CostNoSpline, newIndividual.SolNoSpline] = costFunction(newIndividual.Position, model, nSplinePoints, beta_penalty);

    % Update GlobalBest if there is an improvement
    if (GlobalBest.Cost - newIndividual.Cost) > 0
        succ = 1;
        GlobalBest = newIndividual; % from the local search
    else
        succ = 0;
    end
end

function [z, sol] = WrapperCost(solVec, model)
    sol = create_empty_individual();
    % Split solVec into x and y components
    halfLen = length(solVec) / 2;
    sol.Position.x = solVec(1:halfLen);
    sol.Position.y = solVec(halfLen+1:end);
    
    [z, sol] = cost_LS(sol, model);  % Use MyCost2 alwayse becyae we want sharp cost of the pos
    
end

function [r1, r2,r3] = gnR1R2(NP1, NP2, r0)
  
    NP0 = length(r0);
    r1 = floor(rand(1, NP0) * NP1) + 1;
    
    for i = 1 : 99999999
        pos = (r1 == r0);
        if sum(pos) == 0
            break;
        else % regenerate r1 if it is equal to r0
            r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000, % this has never happened so far
            error('Can not genrate r1 in 1000 iterations');
        end
    end
    
    r2 = floor(rand(1, NP0) * NP2) + 1;
    %for i = 1 : inf
    for i = 1 : 99999999
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos)==0
            break;
        else % regenerate r2 if it is equal to r0 or r1
            r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
        end
        if i > 1000, % this has never happened so far
            error('Can not genrate r2 in 1000 iterations');
        end
    end
    
    r3= floor(rand(1, NP0) * NP1) + 1;
    %for i = 1 : inf
    for i = 1 : 99999999
        pos = ((r3 == r0) | (r3 == r1) | (r3==r2));
        if sum(pos)==0
            break;
        else % regenerate r2 if it is equal to r0 or r1
             r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000, % this has never happened so far
            error('Can not genrate r2 in 1000 iterations');
        end
    end
end

function x = han_boun_individual(x, ub, lb, x2)
    if isscalar(ub)
        ub = repmat(ub, 1, numel(x2));
    end
    if isscalar(lb)
        lb = repmat(lb, 1, numel(x2));
    end

    x_L = lb;
    pos = x < x_L;
    x(pos) = (x2(pos) + x_L(pos)) / 2;

    x_U = ub;
    pos = x > x_U;
    x(pos) = (x2(pos) + x_U(pos)) / 2;
end

function newPos = chaoticMap(pos, mapType)
    % Chaotic Map Function
    % Inputs:
    %   pos - Current position
    %   mapType - Type of chaotic map ('logistic', 'sine', 'tent')
    % Output:
    %   newPos - New position after applying chaotic map

    if nargin < 2
        mapType = 'logistic'; % Default map
    end

    switch mapType
        case 'logistic'
            % Logistic Map
            r = rand * (3.57- 0.1) + 4; % Random r in [3.57, 4]
            newPos = r .* pos .* (1 - pos);
        case 'sine'
            % Sine Map
            r = rand * (1 - 0.9) + 0.9; % Random r in [0.9, 1]
            newPos = r .* sin(pi .* pos);
        case 'tent'
            % Tent Map
            r = rand * (5 - 1.5) + 1.5; % Random r in [1.5, 2]
            if pos < 0.5
                newPos = r .* pos;
            else
                newPos = r .* (1 - pos);
            end
        otherwise
            error('Unknown map type');
    end
end


function GlobalBest = initializeGlobalBest()
    % Initializing GlobalBest structure and all its fields and subfields
    % Position
    GlobalBest.Position.x = [];
    GlobalBest.Position.y = [];
    % Cost
    GlobalBest.Cost = inf;
    % Sol
    GlobalBest.Sol.TS = []; % Parameter values for the path
    GlobalBest.Sol.XS = []; % The x-coordinates of the path including start and end
    GlobalBest.Sol.YS = []; % The y-coordinates of the path including start and end
    GlobalBest.Sol.tt = []; % A linear space from 0 to 1 with 100 points for interpolation
    GlobalBest.Sol.xx = []; % Interpolated x-coordinates using cubic spline
    GlobalBest.Sol.yy = []; % Interpolated y-coordinates using cubic spline
    GlobalBest.Sol.dx = []; % Difference between consecutive x-coordinates of interpolated path
    GlobalBest.Sol.dy = []; % Difference between consecutive y-coordinates of interpolated path
    GlobalBest.Sol.L = [];  % Total length of the interpolated path
    GlobalBest.Sol.Violation = []; % Path's violation value
    GlobalBest.Sol.IsFeasible = []; % Boolean indicating path's feasibility
end

function [Q] = UpdateQLearning_basedOnPopulation(Q, individual, model, gridResolution, gridMap,action_angles, learning_rate, discount_factor, costFunction, beta_penalty)
    % Iterate through the path points from GlobalBest
    for idx = 1:length(individual.Position.x)-1

        %% Convert current and next path points to grid states
        currentState = round([(individual.Position.x(idx) - model.xmin) / gridResolution, (individual.Position.y(idx) - model.ymin) / gridResolution]) + 1;
        nextState = round([(individual.Position.x(idx + 1) - model.xmin) / gridResolution, (individual.Position.y(idx + 1) - model.ymin) / gridResolution]) + 1;

        % Ensure nextState is within the bounds of the grid
        currentState = max(min(currentState, [size(Q, 1), size(Q, 2)]), [1, 1]);

        %% --------------------------- Check if new state is valid
        if (nextState(1) < 1 || nextState(1) > size(gridMap, 1) || ...
           nextState(2) < 1 || nextState(2) > size(gridMap, 2) || ...
           gridMap(nextState(1), nextState(2)) == 1)
    
            reward = -200;  % Increased penalty for invalid move
            nextState = max(min(nextState, [size(Q, 1), size(Q, 2)]), [1, 1]);
            %nextState = currentState; % revert to the original state if the move is invalid
        else

            %% Get the reward associated with moving to the nextState
            reward = CalculateReward(currentState, nextState, model, gridResolution, costFunction, beta_penalty);
        end

        % Determine the action taken to move from currentState to nextState
        action = DetermineAction(currentState, nextState, action_angles);

        % Update the Q-table
        maxQ = max(Q(nextState(1), nextState(2), :));
        Q(currentState(1), currentState(2), action) = (1 - learning_rate) * Q(currentState(1), currentState(2), action) + learning_rate * (reward + discount_factor * maxQ);
   

    end
end

function [Q] = UpdateQLearning_basedOn_PureQLearning(Q, model, gridResolution, gridMap,action_angles, learning_rate, discount_factor, costFunction, beta_penalty, maxSteps, RL_exploration_prob)
    %% start by start state as current state
    start_state = round([(model.xs - model.xmin) / gridResolution, (model.ys - model.ymin) / gridResolution]) + 1;
    currentState = start_state;
    num_actions = length(action_angles);

    for idx = 1:maxSteps

         %% Choose action using epsilon-greedy policy
        if rand() < RL_exploration_prob
           
            action = randi(num_actions);
        else
            [~, action] = max(Q(currentState(1), currentState(2), :));
        end

        %% take action: update the state based on the selected action
        movement = ActionToMovement(action, num_actions);  % Get movement from action
        nextState = currentState + movement;  % Calculate next state, ensure it remains within bounds

        %nextState = max(min(nextState, [size(Q, 1), size(Q, 2)]), [1, 1]);  % Clamp to grid boundaries
        %% --------------------------- Check if new state is valid
        if nextState(1) < 1 || nextState(1) > size(gridMap, 1) || ...
           nextState(2) < 1 || nextState(2) > size(gridMap, 2) || ...
           gridMap(nextState(1), nextState(2)) == 1
    
            reward = -200;  % Increased penalty for invalid move
            nextState = currentState; % revert to the original state if the move is invalid
        else

            %% Get the reward associated with moving to the nextState
            reward = CalculateReward(currentState, nextState, model, gridResolution, costFunction, beta_penalty);
        end

        %% Update the Q-table
        maxQ = max(Q(nextState(1), nextState(2), :));
        Q(currentState(1), currentState(2), action) = (1 - learning_rate) * Q(currentState(1), currentState(2), action) + learning_rate * (reward + discount_factor * maxQ);
 
         %% Update current state
        currentState = nextState;  
    end
end

function action = DetermineAction(currentState, nextState, action_angles)
    % Calculate the angle between the two states
    deltaY = nextState(2) - currentState(2);
    deltaX = nextState(1) - currentState(1);
    angle = atan2(deltaY, deltaX); % Result is in radians

    % Convert angle from radians to degrees
    angle = mod(rad2deg(angle), 360); % Use mod to ensure the angle is between 0 and 360

    % Find the nearest action angle
    [~, action] = min(abs(action_angles - angle));
end


function reward = CalculateReward(currentState, nextState, model, gridResolution, costFunction, beta_penalty)
    
    % Convert states to actual coordinates
    currentPos.x = (currentState(1)-1) * gridResolution + model.xmin;
    currentPos.y = (currentState(2)-1) * gridResolution + model.ymin;
    nextPos.x = (nextState(1)-1) * gridResolution + model.xmin;
    nextPos.y = (nextState(2)-1) * gridResolution + model.ymin;

    % Convert goal state to grid indices
    goalState = round([(model.xt - model.xmin) / gridResolution, (model.yt - model.ymin) / gridResolution]) + 1;

    % Check if the nextState is the goal state
    if isequal(nextState, goalState)
        % Assign a large positive reward for reaching the goal
        reward = 1000; % You can adjust this value as needed
        return; % Return immediately as we don't need to calculate other penalties or rewards
    end

    % If not goal state, calculate penalties and rewards as before
    sol1.x = [currentPos.x, nextPos.x];
    sol1.y = [currentPos.y, nextPos.y];

    % Calculate the cost based on the MyCost_linearSharpPenalty function
    nSplinePoints = 5; % this is only between two states
    [cost, ~] = costFunction(sol1, model, nSplinePoints, beta_penalty);

    % Calculate the distance from the nextState to the goal
    goalPos = [model.xt, model.yt];
    distanceToGoal = norm([(nextPos.x - goalPos(1)), (nextPos.y - goalPos(2))]);

    % Adjust the reward based on the distance to the goal
    distance_factor = 1; % Change this to scale the effect of the distance

    % Combine the penalty from the path cost and the reward for moving closer to the goal
    reward = -cost  -distance_factor * distanceToGoal;  % Negative cost since lower cost is better, add distance reward

    % Ensure the reward is not too positive to keep the learning process consistent
    reward = max(reward, -1000);  % Setting a lower bound for the reward can be useful in some scenarios
end


function path = GeneratePathFromQ(Q, start_state, goal_state, gridResolution, model, nWayPoints, num_actions, maxCount)
    currentState = start_state;
    rawPath = [];  % Initialize path as empty

    count = 0 ;
    % Generate path using the Q-table
    while ~isequal(currentState, goal_state) && size(rawPath, 1) <= size(Q, 1) * size(Q, 2)
        count = count + 1 ; 
        [~, action] = max(Q(currentState(1), currentState(2), :));  % Choose the best action from the current state
        rawPath = [rawPath; currentState];  % Add the current state to the path
        movement = ActionToMovement(action, num_actions);  % Get movement from action
        nextState = currentState + movement;  % Calculate next state, ensure it remains within bounds
        nextState = max(min(nextState, [size(Q, 1), size(Q, 2)]), [1, 1]);  % Clamp to grid boundaries
        currentState = nextState;  % Update current state

        %check max count
        if (count > maxCount) 
            % Return empty solution
            path.Position.x = [];
            path.Position.y = [];
            %disp('No path found');
            %disp(count);
            return ;
        end
    end
    rawPath = [rawPath; goal_state];  % Ensure the goal is included in the path if not already

    % remove start and endpoints
        % Check if rawPath contains more than two points before removing start and goal
    if size(rawPath, 1) > 2
        % Remove the first and last points (start and goal states)
        rawPath = rawPath(2:end-1, :);  % Remove start and goal
    else
        rawPath = rawPath;  % Keep original if it's too short
    end
    
    % Convert path from grid indices to actual coordinates
    pathCoordinates = (rawPath - 1) * gridResolution + [model.xmin, model.ymin];

    % Interpolate to get the specific number of waypoints
    if size(rawPath, 1) > 1 % Only interpolate if we have more than one point
        xq = linspace(1, size(pathCoordinates, 1), nWayPoints);
        interpX = interp1(1:size(pathCoordinates, 1), pathCoordinates(:, 1), xq, 'linear');
        interpY = interp1(1:size(pathCoordinates, 1), pathCoordinates(:, 2), xq, 'linear');
        path.Position.x = interpX;
        path.Position.y = interpY;
    else
        % In case the raw path is a single point (start is the goal)
        path.Position.x = pathCoordinates(:, 1)';
        path.Position.y = pathCoordinates(:, 2)';
    end
end

function movement = ActionToMovement(action, num_actions)
    % Convert action index to angle
    angle = (action - 1) * (360 / num_actions); % Convert action to angle
    angle_rad = deg2rad(angle); % Convert angle to radians
    
    % Define movement based on angle
    dx = cos(angle_rad);
    dy = sin(angle_rad);
    
    % Assuming a standard grid where each cell is one step away,
    % the movement should still be rounded to the nearest whole number to ensure we stay within grid cells.
    movement = round([dx, dy]);
end


function [cost, sol]=MyCost_linearSharpPenalty(sol1,model, nSplinePoints, beta)
    global countFE;
    countFE = countFE + 1 ; % inc function evaluations

    if nargin <3 || isempty(nSplinePoints)
        nSplinePoints = 100; 
    end
    if nargin <4 || isempty(beta)
        beta = 100;  % stepness factor for the penalty 
    end

    %% spline case
    sol=ParseSolution(sol1,model, nSplinePoints);
    cost=sol.L*(1+beta*sol.Violation);
end

function sol2=ParseSolution(sol1,model, nSplinePoints)

    if nargin <3 || isempty(nSplinePoints)
        nSplinePoints = 100; 
    end

    x=sol1.x;
    y=sol1.y;

    xs=model.xs;
    ys=model.ys;
    xt=model.xt;
    yt=model.yt;
    xobs=model.xobs;
    yobs=model.yobs;
    robs=model.robs;
    width_obs = model.width_obs;
    height_obs = model.height_obs;

    XS=[xs x xt];
    YS=[ys y yt];
    k=numel(XS);
    TS=linspace(0,1,k);
    
    tt=linspace(0,1,nSplinePoints);
    xx=spline(TS,XS,tt);
    yy=spline(TS,YS,tt);
    
    dx=diff(xx);
    dy=diff(yy);
    
    L=sum(sqrt(dx.^2+dy.^2));
    
    nobs = numel(xobs); % Number of Obstacles
    Violation = 0;

    %% Precompute the boundaries for all rectangles
    xmin = xobs - width_obs / 2;
    xmax = xobs + width_obs / 2;
    ymin = yobs - height_obs / 2;
    ymax = yobs + height_obs / 2;

    %% calcualte the vialoation based on the model type (circle/rectangular)
    for k=1:nobs
         switch model.modelId
            %% circle obstacle cases
            case {1, 2, 3} % Circle visualization
                d=sqrt((xx-xobs(k)).^2+(yy-yobs(k)).^2);
                v=max(1-d/robs(k),0);

                 %% Check for line segment intersections with the circle
                for j = 1:length(xx) - 1
                    if fast_line_circle_intersection(xx(j), yy(j), xx(j + 1), yy(j + 1), xobs(k), yobs(k), robs(k))
                        v(j) = 1; % Maximum violation for intersecting line segments
                    end
                end
                Violation=Violation+mean(v);

             case {4, 5} % Rectangle visualization
                %% Rectangle collision detection
                %% Check if points are inside the rectangle
                inside_x = (xx >= xmin(k)) & (xx <= xmax(k));
                inside_y = (yy >= ymin(k)) & (yy <= ymax(k));
                inside = inside_x & inside_y;

                %% Calculate distances to the four sides of the rectangle
                dist_to_left = (xx - xmin(k));
                dist_to_right = (xmax(k) - xx);
                dist_to_bottom = (yy - ymin(k));
                dist_to_top = (ymax(k) - yy);

                %% Calculate the distance to the nearest borders dx and dy 
                dx = min(dist_to_left, dist_to_right);
                dy = min(dist_to_bottom, dist_to_top);

                %% Normalize these distances by the respective half-width or half-height
                normalized_dx = dx / (width_obs(k) / 2);
                normalized_dy = dy / (height_obs(k) / 2);

                %% Calculate the overall normalized distance to the nearest border
                normalized_min_dist_to_border = min(normalized_dx, normalized_dy);

                %% Use the normalized minimum distance as the violation value
                v = normalized_min_dist_to_border;

                %% Set violation to zero for points outside the rectangle
                v(~inside) = 0;

                %% Check for line segment intersections with the rectangle
                for j = 1:length(xx) - 1
                    if fast_line_intersects_rectangle(xx(j), yy(j), xx(j + 1), yy(j + 1), xmin(k), xmax(k), ymin(k), ymax(k))
                        v(j) = 1; % Maximum violation for intersecting line segments
                    end
                end

                %% Accumulate violation over all obstacles
                Violation = Violation + mean(v);
            otherwise
                error('Invalid modelId');
         end
    end
    
    sol2.TS=TS;          % These are the parameter values corresponding to the x and y values of the provided path (with start and end points). It represents a linear space from 0 to 1 with the number of elements equal to the path length + start + end points.
    sol2.XS=XS;          % The x-coordinates of the path including the start and end points.
    sol2.YS=YS;          % The y-coordinates of the path including the start and end points.
    sol2.tt=tt;          % A linear space from 0 to 1 with 100 points. This is used to interpolate the path using spline for smoother visualization and calculation.
    sol2.xx=xx;          % The interpolated x-coordinates of the path using cubic spline interpolation. 
    sol2.yy=yy;          % The interpolated y-coordinates of the path using cubic spline interpolation.
    sol2.dx=dx;          % The difference between consecutive x-coordinates of the interpolated path. Helps in calculating the length of the path.
    sol2.dy=dy;          % The difference between consecutive y-coordinates of the interpolated path. Helps in calculating the length of the path.
    sol2.L=L;            % The total length of the interpolated path.
    sol2.Violation=Violation; % The violation indicates how much the path goes into the obstacles. A violation of 0 means the path doesn't touch any obstacles.
    sol2.IsFeasible=(Violation==0); % A boolean flag indicating if the path is feasible (true) or not (false). A path is feasible if it doesn't violate any obstacle constraints.
end

% Optimized Helper function to check if a line segment intersects with a circular obstacle
function intersects = fast_line_circle_intersection(x1, y1, x2, y2, cx, cy, r)
    % Bounding box check
    if max(x1, x2) < cx - r || min(x1, x2) > cx + r || max(y1, y2) < cy - r || min(y1, y2) > cy + r
        intersects = false;
        return;
    end
    
    % Vector from point 1 to point 2
    dx = x2 - x1;
    dy = y2 - y1;
    
    % Vector from point 1 to circle center
    fx = x1 - cx;
    fy = y1 - cy;
    
    a = dx * dx + dy * dy;
    b = 2 * (fx * dx + fy * dy);
    c = (fx * fx + fy * fy) - r * r;
    
    discriminant = b * b - 4 * a * c;
    
    if discriminant < 0
        intersects = false; % No intersection
    else
        discriminant = sqrt(discriminant);
        t1 = (-b - discriminant) / (2 * a);
        t2 = (-b + discriminant) / (2 * a);
        
        intersects = (t1 >= 0 && t1 <= 1) || (t2 >= 0 && t2 <= 1);
    end
end

% Optimized Helper function to check if a line segment intersects with a rectangle
function intersects = fast_line_intersects_rectangle(x1, y1, x2, y2, xmin, xmax, ymin, ymax)
    % Check if the bounding box of the line segment intersects the rectangle
    if max(x1, x2) < xmin || min(x1, x2) > xmax || max(y1, y2) < ymin || min(y1, y2) > ymax
        intersects = false;
        return;
    end

    % If bounding box intersects, check the edges
    intersects = ...
        line_intersects_edge(x1, y1, x2, y2, xmin, ymin, xmax, ymin) || ... % bottom edge
        line_intersects_edge(x1, y1, x2, y2, xmin, ymax, xmax, ymax) || ... % top edge
        line_intersects_edge(x1, y1, x2, y2, xmin, ymin, xmin, ymax) || ... % left edge
        line_intersects_edge(x1, y1, x2, y2, xmax, ymin, xmax, ymax);       % right edge
end

% Helper function to check if a line segment intersects with a rectangle
function intersects = line_intersects_rectangle(x1, y1, x2, y2, xmin, xmax, ymin, ymax)
    intersects = ...
        line_intersects_edge(x1, y1, x2, y2, xmin, ymin, xmax, ymin) || ... % bottom edge
        line_intersects_edge(x1, y1, x2, y2, xmin, ymax, xmax, ymax) || ... % top edge
        line_intersects_edge(x1, y1, x2, y2, xmin, ymin, xmin, ymax) || ... % left edge
        line_intersects_edge(x1, y1, x2, y2, xmax, ymin, xmax, ymax);       % right edge
end

% Helper function to check if two line segments intersect
function intersects = line_intersects_edge(x1, y1, x2, y2, x3, y3, x4, y4)
    % Calculate the direction of the lines
    d1 = direction(x3, y3, x4, y4, x1, y1);
    d2 = direction(x3, y3, x4, y4, x2, y2);
    d3 = direction(x1, y1, x2, y2, x3, y3);
    d4 = direction(x1, y1, x2, y2, x4, y4);

    % Check if the line segments straddle each other
    intersects = (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ...
                  ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) || ...
                  (d1 == 0 && on_segment(x3, y3, x4, y4, x1, y1)) || ...
                  (d2 == 0 && on_segment(x3, y3, x4, y4, x2, y2)) || ...
                  (d3 == 0 && on_segment(x1, y1, x2, y2, x3, y3)) || ...
                  (d4 == 0 && on_segment(x1, y1, x2, y2, x4, y4));
end

% Helper function to calculate the direction of the turn
function d = direction(xi, yi, xj, yj, xk, yk)
    d = (xk - xi) * (yj - yi) - (yk - yi) * (xj - xi);
end

% Helper function to check if a point is on a segment
function onSeg = on_segment(xi, yi, xj, yj, xk, yk)
    onSeg = (min(xi, xj) <= xk && xk <= max(xi, xj) && ...
             min(yi, yj) <= yk && yk <= max(yi, yj));
end

function gridMap = createGridMap(model, resolution)
    %% Calculate the number of grid cells in x and y directions
    num_cells_x = ceil((model.xmax - model.xmin) / resolution);
    num_cells_y = ceil((model.ymax - model.ymin) / resolution);
    
    %% Initialize the grid with zeros (no obstacles)
    gridMap = zeros(num_cells_x, num_cells_y);
    
    %% Number of obstacles
    nobs = numel(model.xobs);
    for i = 1:nobs       
        %% Convert obstacle positions (centers) to grid cell indices
        obstacle_center_x = ceil((model.xobs(i) - model.xmin) / resolution);
        obstacle_center_y = ceil((model.yobs(i) - model.ymin) / resolution);
        
       switch model.modelId
            case {1, 2, 3} % Circular obstacles

                 %% Calculate the radius in grid cells
                obstacle_radius_cells = ceil(model.robs(i) / resolution);

                %% Determine the range of cells to be marked as obstacles
                x_start = max(obstacle_center_x - obstacle_radius_cells, 1);
                x_end = min(obstacle_center_x + obstacle_radius_cells, num_cells_x);
                y_start = max(obstacle_center_y - obstacle_radius_cells, 1);
                y_end = min(obstacle_center_y + obstacle_radius_cells, num_cells_y);       
                %% Mark the cells within the circular obstacle's area
                for x = x_start:x_end
                    for y = y_start:y_end
                        % Calculate the actual distance from the center in world coordinates
                        dist_x = (x - obstacle_center_x) * resolution;
                        dist_y = (y - obstacle_center_y) * resolution;
                        distance = sqrt(dist_x^2 + dist_y^2);

                        if distance <= model.robs(i)
                            gridMap(x, y) = 1;
                        end
                    end
                end

            case {4, 5} % Rectangular obstacles
                %% Calculate the half-width and half-height in grid cells
                half_width_cells = ceil(model.width_obs(i) / (2 * resolution));
                half_height_cells = ceil(model.height_obs(i) / (2 * resolution));

                %% Calculate the rectangle boundaries in grid cells
                x_start = max(obstacle_center_x - half_width_cells, 1);
                x_end = min(obstacle_center_x + half_width_cells, num_cells_x);
                y_start = max(obstacle_center_y - half_height_cells, 1);
                y_end = min(obstacle_center_y + half_height_cells, num_cells_y);

                %% Mark the cells within the rectangular boundaries as obstacles
                gridMap(x_start:x_end, y_start:y_end) = 1;

            otherwise
                error('Invalid modelId');
        end
    end  
end
