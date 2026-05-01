function recon = siddon_backprojection(sinogram, angles, det_positions, N)
% SIDDON_BACKPROJECTION Backprojects a sinogram using Siddon ray tracing
%   Inputs:
%       sinogram      - (n_detectors x n_angles) measured projection
%       angles        - projection angles in degrees
%       det_positions - positions of detectors (length n_detectors)
%       N             - image size (NxN)
%   Output:
%       recon         - reconstructed image (NxN)
%
%   This is the exact adjoint of siddon_forward.

n_angles    = length(angles);
n_detectors = length(det_positions);
center      = (N+1)/2;

recon = zeros(N,N);

for a = 1:n_angles
    theta = angles(a) * pi/180;
    for d = 1:n_detectors
        s   = det_positions(d);
        val = sinogram(d,a);

        % Get pixels crossed by ray and their lengths
        [pixels, lengths] = siddon_ray(s, theta, N, center);

        % Add weighted contribution to pixels
        for k = 1:size(pixels,1)
            i = pixels(k,1);
            j = pixels(k,2);
            recon(i,j) = recon(i,j) + val * lengths(k);
        end
    end
    if mod(a,10)==0
        fprintf('  Backprojected angle %d/%d\n', a, n_angles);
    end
end

% Scale by angle step for consistency with continuous adjoint
recon = recon * (pi / (n_angles));

end