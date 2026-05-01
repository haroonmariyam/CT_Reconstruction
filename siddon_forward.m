function sinogram = siddon_forward(img, angles, det_positions,N)
% SIDDON_FORWARD  Compute sinogram using Siddon ray tracing
%
% Inputs:
%   img           - NxN image
%   angles        - projection angles (degrees)
%   det_positions - detector positions (same units as image grid)
%
% Output:
%   sinogram      - [num_detectors x num_angles]

    [N, M] = size(img);
    if N ~= M
        error('Image must be square');
    end

    n_angles    = length(angles);
    n_detectors = length(det_positions);
    center      = (N + 1) / 2;

    sinogram = zeros(n_detectors, n_angles);

    for a = 1:n_angles
        theta = angles(a) * pi / 180;

        for d = 1:n_detectors
            s = det_positions(d);

            % Get ray intersections
            [pixels, lengths] = siddon_ray(s, theta, N, center);

            if isempty(pixels)
                continue;
            end

            % Line integral
            val = 0;
            for k = 1:size(pixels,1)
                i = pixels(k,1);
                j = pixels(k,2);
                val = val + img(i,j) * lengths(k);
            end

            sinogram(d,a) = val;
        end

        % Progress print (optional)
        if mod(a,10) == 0
            fprintf('Forward projection: angle %d/%d\n', a, n_angles);
        end
    end
end