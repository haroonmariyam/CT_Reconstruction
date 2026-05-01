clear all; close all; clc;
rng(42);

%% Setup
% Standard parallel beam geometry - same as used throughout the project
N             = 256;
angles        = 0:1:179;
n_angles      = length(angles);
n_detectors   = 2*ceil(N/sqrt(2)) + 1;
det_positions = linspace(-N/sqrt(2), N/sqrt(2), n_detectors);

%% Base Shepp-Logan ellipse parameters
% Each row is [intensity, a, b, x0, y0, angle_deg]
% These are the standard values from Shepp & Logan (1974)
SHEPP_LOGAN_BASE = [
     1.0,   0.6900, 0.9200,  0.00,   0.0000,   0;
    -0.8,   0.6624, 0.8740,  0.00,  -0.0184,   0;
    -0.2,   0.1100, 0.3100,  0.22,   0.0000, -18;
    -0.2,   0.1600, 0.4100, -0.22,   0.0000,  18;
     0.1,   0.2100, 0.2500,  0.00,   0.3500,   0;
     0.1,   0.0460, 0.0460,  0.00,   0.1000,   0;
     0.1,   0.0460, 0.0460,  0.00,  -0.1000,   0;
    -0.02,  0.0460, 0.0230, -0.08,  -0.6050,   0;
    -0.02,  0.0230, 0.0230,  0.00,  -0.6060,   0;
    -0.02,  0.0230, 0.0460,  0.06,  -0.6050,   0;
];

%% Helper functions

function img = make_shepp_logan_phantom(N, noise_level, base_params)
    % Generates a randomised Shepp-Logan phantom by perturbing ellipse
    % parameters. This gives structural variation across samples while
    % keeping anatomy broadly consistent, acting as a proxy for
    % inter-patient variation within a single body region.
    img  = zeros(N, N);
    half = N / 2;

    for e = 1:size(base_params, 1)
        intensity = base_params(e,1) * (1 + (rand*2-1)*noise_level);
        a         = base_params(e,2) * (1 + (rand*2-1)*noise_level/2);
        b         = base_params(e,3) * (1 + (rand*2-1)*noise_level/2);
        x0        = base_params(e,4) + (rand*2-1)*noise_level/4;
        y0        = base_params(e,5) + (rand*2-1)*noise_level/4;
        angle_deg = base_params(e,6) + (rand*2-1)*5;
        angle_rad = deg2rad(angle_deg);
        cos_a     = cos(angle_rad);
        sin_a     = sin(angle_rad);

        [X, Y] = meshgrid(1:N, 1:N);
        Xn = (X - half) / half;
        Yn = (Y - half) / half;

        % Rotate coordinates into ellipse frame
        xr = cos_a * (Xn - x0) + sin_a * (Yn - y0);
        yr = -sin_a * (Xn - x0) + cos_a * (Yn - y0);

        mask = (xr/a).^2 + (yr/b).^2 <= 1;
        img(mask) = img(mask) + intensity;
    end

    img = max(img, 0);
end


function recon = fbp_hann(P, angles, det_positions, N)
    % FBP with Hann-windowed ramp filter.
    % Zero padding prevents circular convolution artefacts.
    n_angles   = length(angles);
    P_filtered = zeros(size(P));

    for k = 1:n_angles
        proj        = P(:,k);
        L           = length(proj);
        n_pad       = 2^nextpow2(2*L);
        n_side      = floor((n_pad-L)/2);
        proj_padded = [zeros(n_side,1); proj; zeros(n_pad-L-n_side,1)];
        proj_fft    = fft(proj_padded);
        freq        = ifftshift((-n_pad/2:n_pad/2-1)/n_pad)';
        H           = abs(freq) .* (0.5 + 0.5*cos(2*pi*freq));
        proj_filtered   = real(ifft(proj_fft .* H));
        P_filtered(:,k) = proj_filtered(n_side+1:n_side+L);
    end

    recon = siddon_backprojection(P_filtered, angles, det_positions, N);
    recon = max(recon, 0);
end


function [clean_recon, noisy_recon] = generate_pair(phantom_img, I0, ...
                                       angles, det_positions, N)
    % Forward project the phantom, apply Poisson noise at the given dose,
    % then reconstruct both the clean and noisy sinograms using FBP.
    % The clean reconstruction is the training target, not the true phantom,
    % so the network learns to map noisy FBP to clean FBP.
    P_clean      = siddon_forward(phantom_img, angles, det_positions, N);
    P_scale      = P_clean / max(P_clean(:));
    transmission = exp(-P_scale);
    counts       = poissrnd(I0 * transmission);
    counts       = max(counts, 1);       % avoid log(0)
    P_noisy      = -log(counts / I0);
    P_noisy      = max(P_noisy, 0);
    P_noisy      = P_noisy * max(P_clean(:));
    clean_recon  = fbp_hann(P_clean, angles, det_positions, N);
    noisy_recon  = fbp_hann(P_noisy, angles, det_positions, N);
end

%% Dataset parameters
n_train     = 200;
n_val       = 25;
n_test      = 25;
n_total     = n_train + n_val + n_test;
noise_level = 0.15;      % 15% parameter perturbation for phantom variation
I0_levels   = [1e3, 1e4, 1e5];   % high, medium, low noise

%% Storage
% All images stored as single precision to reduce file size
clean_images   = zeros(N, N, n_total, 'single');
noisy_images   = zeros(N, N, n_total, 'single');
phantom_images = zeros(N, N, n_total, 'single');
I0_used        = zeros(n_total, 1);
split_labels   = zeros(n_total, 1);    % 1=train, 2=val, 3=test

%% Generate all samples
fprintf('Generating %d samples (%d train / %d val / %d test)...\n', ...
    n_total, n_train, n_val, n_test);
fprintf('Dose levels: 1e3, 1e4, 1e5 (randomly sampled per image)\n\n');

for idx = 1:n_total

    % Assign to train, val or test split
    if idx <= n_train
        split_labels(idx) = 1;
        split_name = 'train';
    elseif idx <= n_train + n_val
        split_labels(idx) = 2;
        split_name = 'val';
    else
        split_labels(idx) = 3;
        split_name = 'test';
    end

    % Pick a random dose level for this sample
    I0 = I0_levels(randi(length(I0_levels)));
    I0_used(idx) = I0;

    % Generate a new randomised phantom for this sample
    phantom_img = make_shepp_logan_phantom(N, noise_level, SHEPP_LOGAN_BASE);

    % Generate matched clean and noisy FBP reconstructions
    [clean_recon, noisy_recon] = generate_pair(phantom_img, I0, ...
                                  angles, det_positions, N);

    % Normalise each image independently to [0,1]
    clean_min  = min(clean_recon(:));
    clean_max  = max(clean_recon(:));
    clean_norm = (clean_recon - clean_min) / (clean_max - clean_min + 1e-8);

    noisy_min  = min(noisy_recon(:));
    noisy_max  = max(noisy_recon(:));
    noisy_norm = (noisy_recon - noisy_min) / (noisy_max - noisy_min + 1e-8);

    clean_images(:,:,idx)   = single(clean_norm);
    noisy_images(:,:,idx)   = single(noisy_norm);
    phantom_images(:,:,idx) = single(phantom_img);

    if mod(idx, 10) == 0
        fprintf('Generated %d/%d (%s, I0=%.0e)\n', idx, n_total, split_name, I0);
    end
end

%% Save dataset
fprintf('\nSaving dataset...\n');
save('ct_dataset_siddon.mat', ...
    'clean_images', ...     % [256 x 256 x 250] normalised clean FBP reconstructions
    'noisy_images', ...     % [256 x 256 x 250] normalised noisy FBP reconstructions
    'phantom_images', ...   % [256 x 256 x 250] true phantom attenuation maps
    'I0_used', ...          % [250 x 1] dose level used per sample
    'split_labels', ...     % [250 x 1] 1=train 2=val 3=test
    'n_train', 'n_val', 'n_test', 'N', 'angles', '-v7.3');

fprintf('Saved: ct_dataset_siddon.mat\n');
fprintf('  Train: %d  Val: %d  Test: %d\n', n_train, n_val, n_test);
fprintf('  I0 distribution:\n');
fprintf('    1e3: %d samples\n', sum(I0_used == 1e3));
fprintf('    1e4: %d samples\n', sum(I0_used == 1e4));
fprintf('    1e5: %d samples\n', sum(I0_used == 1e5));

%% Quick visual check of first sample from each split
figure('Position', [50 50 900 900]);
check_idx    = [1, n_train+1, n_train+n_val+1];
check_labels = {'Train sample 1', 'Val sample 1', 'Test sample 1'};

for ci = 1:3
    idx = check_idx(ci);

    subplot(3, 3, (ci-1)*3 + 1);
    imagesc(phantom_images(:,:,idx)); colormap gray; axis image off;
    title(sprintf('%s\nPhantom', check_labels{ci}), 'FontSize', 8);

    subplot(3, 3, (ci-1)*3 + 2);
    imagesc(clean_images(:,:,idx), [0 1]); colormap gray; axis image off;
    title(sprintf('Clean FBP\nI0=%.0e', I0_used(idx)), 'FontSize', 8);

    subplot(3, 3, (ci-1)*3 + 3);
    imagesc(noisy_images(:,:,idx), [0 1]); colormap gray; axis image off;
    title('Noisy FBP', 'FontSize', 8);
end

sgtitle('Dataset check - first sample from each split', 'FontSize', 11);
saveas(gcf, 'dataset_check.png');
fprintf('Saved: dataset_check.png\n');
fprintf('\nDone! Upload ct_dataset_siddon.mat to Colab to train.\n');