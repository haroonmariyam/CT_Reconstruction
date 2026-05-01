function [pixels, lengths] = siddon_ray(s, theta, N, center)
% SIDDON_RAY  Compute pixel intersections for a single ray
%
% Inputs:
%   s      - detector position
%   theta  - angle (radians)
%   N      - image size (NxN)
%   center - image center ((N+1)/2)
%
% Outputs:
%   pixels  - [k x 2] indices (i,j)
%   lengths - [k x 1] intersection lengths

    cos_t = cos(theta);
    sin_t = sin(theta);

    % Ray parametrisation: x = ox + t*dx, y = oy + t*dy
    ox = s * cos_t;
    oy = s * sin_t;
    dx = -sin_t;
    dy =  cos_t;

    % Grid boundaries (pixel edges)
    x_planes = (0.5:1:N+0.5) - center;
    y_planes = (0.5:1:N+0.5) - center;

    % Compute intersection parameters 
    if abs(dx) > 1e-12
        tx = (x_planes - ox) / dx;
        tx_min = min(tx);
        tx_max = max(tx);
    else
        tx = [];
        tx_min = -inf;
        tx_max = inf;
    end

    if abs(dy) > 1e-12
        ty = (y_planes - oy) / dy;
        ty_min = min(ty);
        ty_max = max(ty);
    else
        ty = [];
        ty_min = -inf;
        ty_max = inf;
    end

    % Entry and exit points
    t_enter = max(tx_min, ty_min);
    t_exit  = min(tx_max, ty_max);

    % No intersection
    if t_exit <= t_enter
        pixels  = [];
        lengths = [];
        return;
    end

    % All intersection points within segment
    t_all = sort([tx, ty]);
    t_all = t_all(t_all > t_enter & t_all < t_exit);

    % Include entry/exit
    t_all = [t_enter, t_all, t_exit];

    % Preallocate (max possible segments ~ 2N)
    pixels  = zeros(length(t_all), 2);
    lengths = zeros(length(t_all), 1);
    count   = 0;

    % Traverse segments
    for k = 1:length(t_all)-1
        dt = t_all(k+1) - t_all(k);

        if dt < 1e-12
            continue;
        end

        % Midpoint of segment
        t_mid = 0.5 * (t_all(k) + t_all(k+1));
        xm = ox + t_mid * dx;
        ym = oy + t_mid * dy;

        % Convert to pixel indices
        j = floor(xm + center) + 1;
        i = floor(center - ym) + 1;

        % Check bounds
        if i >= 1 && i <= N && j >= 1 && j <= N
            count = count + 1;
            pixels(count,:)  = [i, j];
            lengths(count)   = dt;
        end
    end

    % Trim unused
    pixels  = pixels(1:count,:);
    lengths = lengths(1:count);
end