function model = CreateModel_RP2B24_Benchmark(modelType, modelNum, boundaries, fNo)

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
    % Extract boundaries for easier reference
    xmin = boundaries.xmin;
    xmax = boundaries.xmax;
    ymin = boundaries.ymin;
    ymax = boundaries.ymax;

    % Switch based on modelType to set up the environment
    switch modelType
        case 1 % Open Field Model
            % no obstacles to draw only start and end point
            [xs, ys, xt, yt, xobs, yobs, robs,width_obs, height_obs] = Model_openFieldConfig(modelNum);
            
        case 2 % Single Obstacle Model
            % ---------- cricle  
            [xs, ys, xt, yt, xobs, yobs, robs, width_obs, height_obs] = Model_singleObstacleConfig(modelNum);
            
        case 3 % Multiple Small Obstacles Model
            % -----draw circles
            [xs, ys, xt, yt, xobs, yobs, robs, width_obs, height_obs] = Model_multipleSmallObstaclesConfig(modelNum,xmin, xmax, ymin, ymax);
        
        case 4 % Narrow Passage Model
            % ----------- draw as rectangle
             [xs, ys, xt, yt, xobs, yobs, robs,width_obs, height_obs] = Model_narrowPassageConfig(modelNum);
            
        case 5 % Maze-like Model
            % ----- draw rectangle
            [xs, ys, xt, yt,xobs, yobs,robs,  width_obs,height_obs]= Model_mazeLikeConfig(modelNum);
                         
        otherwise
            error('Invalid modelType');
    end

    % Adjust min and max values: 
    % 1. Determine the Minimum and Maximum Coordinates:
    allX = [xs, xt, xobs - width_obs/2, xobs + width_obs/2]; % Include the left and right bounds of rectangles
    allY = [ys, yt, yobs - height_obs/2, yobs + height_obs/2]; % Include the top and bottom bounds of rectangles
    for i = 1:length(xobs) % Including the circumferences of circles
        th = linspace(0, 2*pi, 100);
        xCirc = xobs(i) + robs(i) * cos(th);
        yCirc = yobs(i) + robs(i) * sin(th);
        allX = [allX, xCirc];
        allY = [allY, yCirc];
    end
    minX = min(allX);
    maxX = max(allX);
    minY = min(allY);
    maxY = max(allY);

    % 2. Calculate Margin Values:
    dynamicBoundaryMargine = false;
    dynamicBoundaryMargineValue = 0.1;
    if dynamicBoundaryMargine == true
        xmin = minX - dynamicBoundaryMargineValue;
        xmax = maxX + dynamicBoundaryMargineValue;
        ymin = minY - dynamicBoundaryMargineValue;
        ymax = maxY + dynamicBoundaryMargineValue;
    end
        
    % Number of Obstacles
    n = length(xobs);
    
    % Assign Model parameters
    model.xs = xs;
    model.ys = ys;
    model.xt = xt;
    model.yt = yt;

    model.xobs = xobs;
    model.yobs = yobs;
    model.robs = robs; % for circle shape
    model.width_obs = width_obs; % for rectangle shape
    model.height_obs = height_obs; % for rectangle shape
    model.n = n;

    model.xmin = xmin;
    model.xmax = xmax;
    model.ymin = ymin;
    model.ymax = ymax;

    model.fNo = fNo ; % used for prinitng and naming purposes (like M1, M2, ...)
    model.modelId = modelType; % model type
    model.modelNum = modelNum; % model config inside a type
end


function [xs, ys, xt, yt, xobs, yobs, robs, width_obs, height_obs] = Model_openFieldConfig( modelNum)

    % Initialize empty obstacle arrays
    xobs = [];
    yobs = [];
    robs = [];
    width_obs = [];
    height_obs = [];

    switch modelNum
        case 1
            %% fixed start and end points
            xs = -7; ys = -10;
            xt = -6; yt = 10;
        otherwise
            error('Invalid modelNum for Open Field Model');
    end
end

function [xs, ys, xt, yt, xobs, yobs, robs, width_obs, height_obs] = Model_singleObstacleConfig(modelNum)

    maxX = 10; % Assuming a grid or field of 10x10 for simplicity
    maxY = 10;
    
    switch modelNum
        case 1  
            xs = 0; ys = 0; 
            xt = 4; yt = 6;
            xobs = 2;
            yobs = 3;
            robs = 1.5;
        case 2     
            xs = 4; ys = 2;
            xt = 6; yt = 8;
            xobs = maxX/2; 
            yobs = maxY/2;
            robs = 2.5; 
        otherwise
            error('Invalid modelNum for this modelType');
    end
    [width_obs, height_obs] = Model_circleToRectangleDimensions(robs);
end


function [xs, ys, xt, yt, xobs, yobs, robs, width_obs, height_obs] = Model_multipleSmallObstaclesConfig(modelNum, xmin, xmax, ymin, ymax)

    % Define a range for radii of obstacles based on the overall region
    minR = (xmax-xmin)/30; % Smallest obstacle radius
    maxR = (xmax-xmin)/10; % Largest obstacle radius

    switch modelNum
        case 1
            xs = -8.5; ys = -6; 
            xt = 1; yt = 5;
            xobs = [-5 -2];
            yobs = [-5 2];
            robs = [2.5, 2];
        case 2 
            xs = -2; ys = -2;
            xt = xmax; yt = ymax;
            xobs = [1.5, 6, 6];
            yobs = [2, 2, 7];
            robs = [2.3, 1.5, 2];        
        case 3
            % Standard placement
            xs = 0; ys = 0; 
            xt = 4; yt = 6;
            xobs = [1.5, 4.0, 1.2];
            yobs = [4.5, 3.0, 1.5];
            robs = [1.5, 1.0, 0.8];         
        case 4
            xs = xmin; ys = ymin;
            xt = xmax; yt = ymax;
            xobs = linspace(xmin + (xmax-xmin)/8, xmax - (xmax-xmin)/8, 5);
            yobs = linspace(ymin + (ymax-ymin)/8, ymax - (ymax-ymin)/8, 5);
            robs = repelem(maxR, 5);
        case 5
            xs = 0; ys = 0; 
            xt = 9; yt = 8;
            xobs = [1.5, 3.5, 5.5, 7.5];
            yobs = [1.5, 4.5, 2.5, 5.5];
            robs = [1, 1, 1, 1];
       case 6 % Two obstacles at diagonal corners + center
            xs = xmin; ys = ymin;
            xt = xmax; yt = ymax;
            xobs = [xmax/5, 2*xmax/5, 3*xmax/5, 4*xmax/5];
            yobs = [ymax/5, 4*ymax/5, ymax/5, 4*ymax/5];
            robs = [maxR*0.8, maxR*0.8, maxR*0.8, maxR*0.8];
        case 7 % Previously provided
            xs = -2; ys = -1;
            xt = xmax; yt = ymax;
            xobs = [2, 8, 2, 8];
            yobs = [2, 2, 8, 8];
            robs = [2.5, 1.5, 2, 1.5];
        case 8 % Blocking the direct path
            xs = -6; ys = -5;
            xt = 7; yt = 4;
            xobs = [-4, 4, -4, 4, 0];
            yobs = [-2, -2, 2, 2, 0];
            robs = [1.5, 1.5, 1.5, 1.5, 2];       
        case 9 % Zigzag pattern
              xs = -8; ys = -8;
                xt = 8; yt = 8;
                xobs = [-5, -1, 4, -5, -1, 4];
                yobs = [-3, -3, -3, 3, 3, 3];
                robs = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5];
         case 10 % Horizontal barriers
            xs = xmin; ys = ymin;
            xt = xmax; yt = ymax;
            xobs = [-7.5,   -6.75,  -0.5,    7.2,    6];
            yobs = [-6.5,   0.5,  0.5,   -0.5,    5.5];
            robs = [2.2, 2.6, 3, 2, 3];     
        case 11 % Create barriers with a few gaps and central obstacles
            xs = -8; ys = -8;
            xt = 8; yt = 8;
            xobs = [0, 0, 0, -4, 4, -4, 4];
            yobs = [-6, 0, 6, -2, -2, 2, 2];
            robs = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5];
        case 12 % Vertical barriers
            xs = -8; ys = -8;
            xt = 9; yt = 8;
            xobs = [-5, 1, 7, -5, 1, 7, -5, 1, 7];
            yobs = [-5, -5, -5, 0, 0, 0, 5, 5, 5];
            %robs = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.3];
            robs = [1.4, 1.9, 1.4, 1.9, 1.9, 1.9, 1.4, 1.9, 1.4];
        case 13
            % Creating a 'zigzag' of obstacles, forcing a winding path
            xs = -8; ys = -4.5;
            xt = 8; yt = 8;
            xobs = [6, 6, 6, -6, -6, -6, 0, 0, 2.1, -2.1, 0];
            yobs = [2, 6, -2, 2, 6, -2, -4, 7, 0, 0, 3];
            robs = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5];    
        otherwise
            error('Invalid modelNum for this modelType');
    end
    [width_obs, height_obs] = Model_circleToRectangleDimensions(robs);
end



function [xs, ys, xt, yt, xobs, yobs, robs, width_obs, height_obs] = Model_narrowPassageConfig(modelNum)
    switch modelNum
        % Basic Two Obstacle Configuration
        case 1
            xs = 0; ys = -2.5; 
            xt = 1.5; yt = 0;
            xobs = [1.5, -0.8,3]; 
            yobs = [-2, 0.3,-5];
            width_obs = [1.5, 1.5,1.5];
            height_obs = [3, 3,3];
        case 2
            xs = 2; ys = -9.5;
            xt = 7; yt = 4;
            xobs = [4.5, 2.2, 4.5, 2.2]; 
            yobs = [-8, -4, 0, 4];
            width_obs = [1.5, 1.5, 1.5, 1.5];
            height_obs = [4, 4, 4, 4];         
        case 3
            xs = 2.5; ys = 1.5;
            xt = 2.5; yt = 2.5;
            xobs = [2, 3];
            yobs = [3.3, 2];
            width_obs = [0.2, 2];
            height_obs = [2, 0.4];
        case 4
             xs = 1.85; ys = 6.5;
             xt = 2.35; yt = 4;
            xobs = [2.45, 1.25];
            yobs = [6.75, 5];
            width_obs = [0.4, 1.8];
            height_obs = [3, 1];        
        case 5
            xs = 0; ys = 1.1;
            xt = 2; yt = 6;
            xobs = [0.5, -1, 2];
            yobs = [-1, 3, 4];
            width_obs = [0.5, 3,3];
            height_obs = [6, 0.5,0.5];      
        % Zigzag Passage Configurations
        case 6 % 2 obstacle
            xs = 3.2; ys = 2.8; 
            xt = 4; yt = 6;
            xobs = [4, 2, 3.2,4];
            yobs = [2, 4, 6,8];
            width_obs = [0.5, 3.5, 0.5, 0.5];
            height_obs = [7, 0.5, 2, 2];      
        case 7 % 4 obstacles
            xs = 0; ys = 0; 
            xt = 7; yt = 9;
            xobs = [4, 2, 2, 5];
            yobs = [2, 4, 6, 6];
            width_obs = [0.5, 3, 0.5, 1];
            height_obs = [7, 0.5, 2, 2];
       % Case 8 - Zigzag passage: 6 obstacles
        case 8
           xs = 3; ys = 2;
            xt = 5.25; yt = 6.5;
            xobs = [2, 4, 2, 5, 3];
            yobs = [2, 2, 6, 6, 4];
            width_obs = [0.5, 0.5, 2, 2, 1];
            height_obs = [2, 2, 0.5, 0.5, 1.8];
        case 9
            xs = 0; ys = 0;
            xt = 5; yt = 4;
            xobs = [1, 5.4, 2, 4, 3.5];
            yobs = [2, 2, 5.5, 4.5, 1];
            width_obs = [1, 0.5, 2, 1, 1.5];
            height_obs = [5, 2, 0.5, 2.5, 0.5];
        case 10
            xs = 0; ys = 0; 
            xt = 10; yt = 6;
            xobs = [2, 2, 5, 7, 7]; 
            yobs = [1, 4, 3, 1, 5];
            width_obs = [6, 1, 1, 1, 1];  
            height_obs = [1, 2, 2, 2, 1];
        case 11
            xs = 0; ys = 0; 
            xt = 10; yt = 10;
            xobs = [1, 2, 3, 4, 5, 6, 7, 8, 9]; 
            yobs = [2, 5, 8, 1, 4, 7, 2, 5, 8];
            width_obs = [1, 1, 1, 1, 1, 1, 1, 1, 1];  
            height_obs = [2, 2, 2, 2, 2, 2, 2, 2, 2];
        case 12
              xs = 1.75; ys = 0; 
            xt = 6; yt = 6.5;
            xobs = [1, 2.5, 4, 6, 7.5];
            yobs = [1, 1, 4, 0.5, 6];
            width_obs = [1, 1, 1.5, 1.8, 1];
            height_obs = [7, 3, 7, 8.5, 3];
        case 13   
            xs = 0; ys = 0; 
            xt = 5; yt = 4;
            xobs = [2, 3, 1, 5, 7, 5];
            yobs = [1, 4, 7, 7, 5, 2];
            width_obs = [2, 3, 2, 2, 1, 3];
            height_obs = [3, 3, 2, 2, 3, 1];
        case 14
            xs = 2.5; ys = 0;
            xt = 6; yt = 4;
            xobs = [1.5, 4, 3, 7, 9, 6,7];  % Moved first obstacle's x to 1.5 from 1
            yobs = [1, 1, 4, 4, 2, 7,0];
            width_obs = [1.5, 1, 3, 1, 2, 3,1.5];  % Adjusted first obstacle's width to 1.5 from 2
            height_obs = [8, 4, 2, 5, 6, 2, 2];
        case 15
            xs = 5; ys = -9;  % Start point
            xt = 2; yt = 9;  % End point
            % Obstacle coordinates
            xobs = [2, 5, 2, 5, 2, 5];  % Alternating to create a zigzag pattern
            yobs = [-9, -6, -3, 1, 5, 9];      % Spaced out along the y-axis
            % Obstacle sizes (make sure they fit within the bounds and do not overlap excessively)
            width_obs = [3, 3, 3, 3, 3, 3];    % Width of obstacles
            height_obs = [2, 2, 2, 2, 2, 2];   % Height of obstacles
        case 16
            xs = 5; ys = -9;  % Start point
            xt = 3; yt = 4;  % End point
            % Obstacle coordinates
            xobs = [2, 5, 2, 5, 2, 5, 2, 5];  % Alternating to create a zigzag pattern
            yobs = [-8, -6,-4, -2, 2,4, 6, 8];         % Spaced out along the y-axis
            % Obstacle sizes
            width_obs = [2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8];    % Width of obstacles
            height_obs = [1.3, 1.3,1.3, 1.3, 1.3, 1.3, 1.3,1.3];  % Height of obstacles     
        case 17
            xs = 6.5; ys = -4;
            xt = 8.1; yt =8.8;
            xobs = [6.5,	8.5,	6.5,	8.5,	6.5,	8.5,	6.5,	8.5,	6.5,	8.5,	6.5,	8.5];  % Alternating positions for zigzag
            yobs = [-8,-7, -6, -4, -2, 0, 2, 4, 6, 6.7, 8, 10];      % Evenly spaced along y-axis
            width_obs = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];        % Uniform width for all obstacles
            height_obs = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]; % Uniform height  
        case 18
            xs = -10; ys = -5;
            xt = -2.2; yt = 7.8;
            xobs = [-7, -4, -7, -4, -7, -4, -7, -4, -7, -4];  % Alternating positions for zigzag
            yobs = [-8, -6, -4, -2, 0, 2, 4, 6, 8, 10];      % Evenly spaced along y-axis
            width_obs = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4];      % Uniform width for all obstacles
            height_obs = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]; % Uniform height
        case 19 
           xs = -4; ys = -5; 
            xt = 7; yt = 3.4;
            xobs = [-3	-3	-1.8	-1	-1	0.3	1.5	1.8	3	4	4.1	5.7	5.5	7.5]; 
            yobs = [-1.5	2	-4	0	4	-2	2	-4.5	0	4.5	-2.3	2	-4.2	0];
            width_obs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];  
            height_obs = [2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];      
        case 20 
           xs = 2; ys = 4; 
            xt = 6; yt = 9;
            xobs = [1, 1, 2, 3, 3, 4, 5, 5, 6, 7]; 
            yobs = [3, 7, 1, 5, 9, 3, 7, 1, 5, 9];
            width_obs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1];  
            height_obs = [4, 4, 2, 2, 2, 2, 2, 2, 2, 2];  
        case 21
            xs = 3; ys = 1; 
            xt = 9; yt = 7;
            xobs =       [1, 1, 2, 3.5, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9.2]; 
            yobs =       [1, 5,-1, 3, 7, 1, 5,-1.3, 3, 7, 0.8, 5, -1.5, 2.3] ;
            width_obs =  [1, 1, 1, 1.5, 1,1.3, 1.2, 1, 1.2, 2, 1, 1, 1.5, 1.5];  
            height_obs = [4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        otherwise
            error('Invalid modelNum for Narrow Passage Model');      
    end
    % Calculate the radii of the circles that encompass these rectangles, once for all cases
    robs = Model_rectangleToCircleRadius(width_obs, height_obs);
end



function [xs, ys, xt, yt, xobs, yobs,robs, width_obs, height_obs] = Model_mazeLikeConfig(modelNum)
    % Switch based on modelNum to return different configurations
    switch modelNum
        case 1  % Simple Maze
            xs = 0; ys = 0;
            xt = 8; yt = 8;
            xobs = [2, 6];
            yobs = [2, 6];
            width_obs = [0.1, 0.1];
            height_obs = [6, 6];
        case 2  % Moderate Maze
            xs = 0; ys = 0;
            xt = 10; yt = 10;
            xobs = [2, 4, 4, 6, 8, 8];
            yobs = [2, 6, 2, 4, 2, 8];
            width_obs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
            height_obs = [7, 3, 0.1, 5, 6, 0.1];
        case 3  % Complex Maze
            xs = 0; ys = 0;
            xt = 8; yt = 8;
            xobs = [2, 2, 4, 4, 6, 6];
            yobs = [1, 5, 1, 5, 1, 5];
            width_obs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
            height_obs = [4, 2, 4, 2, 4, 2];
        case 4 % Simple L-shape
           xs = 0; ys = 0;
            xt = 9.5; yt = 8.2;            
            xobs = [1, 1, 3, 3, 5, 5, 7, 7, 9, 9];
            yobs = [2, 6, 2, 6, 2, 6, 2, 6, 2, 6];
            width_obs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
            height_obs = [3.8, 1, 3.8, 1, 3.8, 1, 3.8, 1, 3.8, 1];   
        case 5 % A bit more complexity with more turns
            xs = 0; ys = 0;
            xt = 10; yt = 8;
            xobs = [2, 2, 4, 4, 6, 6, 8, 8];
            yobs = [2, 6, 1, 5, 2, 6, 1, 5];
            width_obs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
            height_obs = [3, 3, 2, 2, 3, 3, 2, 2];
        case 6 % Zigzag path but more evenly distributed
            xs = 0; ys = 0;
            xt = 10; yt = 10;
            xobs = [2, 2, 4, 4, 6, 6, 8, 8];
            yobs = [2, 8, 1, 5, 1, 5, 2, 8];
            width_obs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
            height_obs = [5, 1, 3, 3, 3, 3, 5, 1];
        case 7 % Complex path with an increase in dead-ends and turns
            xs = -5; ys = -5;
            xt = 4; yt = 2;
            xobs = [-4	-4	-2	-2	0	0	2	2	4	4	6	6];
            yobs = [-4	4	-4	2	-4	4	-4	3	-2	4	-4	2];
            width_obs = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1];
            height_obs = [6, 6, 2, 4, 7, 2, 4, 5, 6, 2, 2, 4];
        case 8 % A more advanced maze with narrow and winding paths
            xs = -2; ys = -7;
            xt = 10; yt = 7;
            xobs = [0	0	0	2	2	4	4	5.5	6	6	6	8	8	8];
            yobs = [-4	-0.2	4	-4	3.5	-3.5	1	6	-4	0	3	-2	2	6.2]; 
            width_obs = [0.1, 0.1, 2.5, 0.1, 0.1, 0.1, 2.5, 3, 0.1, 3, 0.1, 0.1, 0.1, 0.1 ];
            height_obs = [4, 1.5, 0.2, 2, 4, 3, 0.2, 0.2, 3, 0.2, 1.5, 2.5, 1.5, 1.5];
        case 9 % Maze with a clear path
            xs = 0; ys = -5;
            xt = 4; yt = 3;
            xobs = [-3	-3	-3	-1	-1	-1	1	1	3	3	3	5	5	5];
            yobs = [-3	1	5	-3	1	5	-3	5	-3	1	5	-3	1	5];
            width_obs = [0.1, 2, 0.1, 3.2, 0.1, 0.1, 0.1, 2, 2, 3, 0.1, 0.1, 0.1, 0.1];
            height_obs = [3, 0.1, 3, 0.1, 3, 3, 7, 0.1, 0.1, 0.1, 4, 3.5, 3, 3];
        case 10 % Maze that requires backtracking
            xs = -4; ys = -4;
            xt = 4; yt = -1;
            xobs = [-5	-5	-3	-3	-3	-1	-1	1	1	2	3	3	5	4.3];
            yobs = [-3	3	-6	0	5	-3.5	4	0	5	-6	-2	2	-3	2];
            width_obs = [0.1,0.1,3,3,3,0.1,0.1,2,4, 4.5, 0.1,0.1,0.1,2];
            height_obs = [8, 4, 0.1, 0.1, 0.2, 6, 4, 0.1, 0.1, 0.1, 3.5, 3, 8, 0.1];
        case 11 % Complex zig-zag path with multiple options
            xs = -5; ys = -5;
            xt = 5; yt = 3.7;
            xobs = [-6	-6	-6	-4	-4	-2	-2	-2	0	0	2	2	2	4	4	6	6];
            yobs = [-3	3	5	-3	1.8	-6	0	5	-5	3	-6	0	5	-5	1.8	-3	5];
            width_obs = [0.1, 2, 0.1, 3, 0.1, 2, 2, 3, 0.1, 0.1, 2, 2, 3, 0.1, 0.1, 0.1, 0.1];
            height_obs = [9, 0.1, 3, 0.1, 2.5, 2.9, 2.9, 0.1, 6, 4, 2.9, 2.9, 0.1, 6, 2.5, 9, 3];
        case 12 % Very complex path with dead ends
 
            xs = -2.5; ys = 2.5;
            xt = 5; yt = 0;
            xobs = [-6	-6	-6	-4	-4	-2	-2	-2	0	0	2	2	2	4	4	6	6];
            yobs = [-3	3	5	-3	1.8	-6	0	5	-5	3	-6	0	5	-5	1.8	-3	5];
            width_obs = [0.1, 2, 0.1, 3, 0.1, 2, 2, 3, 0.1, 0.1, 2, 2, 3, 0.1, 0.1, 0.1, 0.1];
            height_obs = [9, 0.1, 3, 0.1, 2.5, 2.9, 2.9, 0.1, 6, 4, 2.9, 2.9, 0.1, 6, 2.5, 9, 3];
            

        case 13 % Extensive maze with tight pathways
            xs = -5; ys = -4.6;
            xt = 5; yt = 0;
            xobs = [-6	-4	-4	-6	-4	-4	-2	-2	-2	0	0	0	2	2	4	4	4	2	6	6];
            yobs = [-4	0	2	4	-5	-2	-4	0	4	-6	-2	1	-4	0	-6	-2	2	4	-4	4];
            width_obs = [0.1, 0.1, 3, 0.1, 0.1, 3, 2, 2, 2, 0.1, 0.1, 0.1, 2, 2, 3, 3, 3, 2, 0.1, 0.1];
            height_obs = [8, 2, 0.1, 3, 2, 0.1, 2, 2, 2, 2.2, 2.2, 2.2, 2, 2, 0.1, 0.1, 0.1, 2, 8, 4];        
        otherwise
            error('Invalid modelNum for this modelType');
    end
    % Calculate the radii of the circles that encompass these rectangles, once for all cases
    robs = Model_rectangleToCircleRadius(width_obs, height_obs);
end
 

function [widths, heights] = Model_circleToRectangleDimensions(rs)
    % get the outer rectangle
    widths = 2 * rs;
    heights = 2 * rs;
end

function r = Model_rectangleToCircleRadius(widths, heights)
    % get the outer circle
    r = 0.5 * sqrt(widths.^2 + heights.^2);
end
