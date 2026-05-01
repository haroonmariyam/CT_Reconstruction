rng(42);

%
% Define image size, projection angles, and detector configuration
N = 256;
phantom_img = phantom(N);
angles = 0:1:179;
n_angles = length(angles);
n_detectors = 2*ceil(N/sqrt(2)) + 1;
det_positions = linspace(-N/sqrt(2), N/sqrt(2), n_detectors);
center = (N+1)/2;
num_iterations = 20;
n_subsets = 10;

%% Forward projection (Siddon)
% Generate the clean sinogram from the phantom using Siddon ray tracing
P_clean = siddon_forward(phantom_img, angles, det_positions, N);

%% FBP on clean data (baseline)
% Apply Hann-windowed ramp filter to each projection before back projection
P_filtered = zeros(size(P_clean));
for k = 1:n_angles
    proj              = P_clean(:,k);
    L                 = length(proj);
    n_pad             = 2^nextpow2(2*L);
    n_side            = floor((n_pad-L)/2);
    proj_padded       = [zeros(n_side,1); proj; zeros(n_pad-L-n_side,1)];
    proj_fft          = fft(proj_padded);
    freq              = ifftshift((-n_pad/2:n_pad/2-1)/n_pad)';
    H                 = abs(freq).* (0.5 + 0.5*cos(2*pi*freq));
    proj_fft_filtered = proj_fft .* H;
    proj_filtered     = real(ifft(proj_fft_filtered));
    P_filtered(:,k)   = proj_filtered(n_side+1:n_side+L);
end
recon_fbp = siddon_backprojection(P_filtered, angles, det_positions, N);
recon_fbp = max(recon_fbp, 0);
gt_n      = mat2gray(phantom_img);
fbp_n     = mat2gray(recon_fbp);
rmse_fbp  = sqrt(mean((recon_fbp(:) - phantom_img(:)).^2));
psnr_fbp  = psnr(fbp_n, gt_n);
ssim_fbp  = ssim(fbp_n, gt_n);
fprintf('FBP (clean) -> RMSE=%.5f | PSNR=%.2f | SSIM=%.4f\n', rmse_fbp, psnr_fbp, ssim_fbp);

%% Noise levels - MEDIUM AND HIGH ONLY
% Medium (I0=1e4) and high (I0=1e3) noise levels evaluated
% Lower I0 means fewer photons and therefore higher noise
I0_values    = [1e4, 1e3];
noise_labels = {'Medium Noise', 'High Noise'};
n_noise      = length(I0_values);

%% Generate noisy sinograms
% Poisson noise is applied at the photon count level via Beer-Lambert inversion
P_scale = P_clean / max(P_clean(:));
P_noisy = cell(n_noise, 1);
fprintf('\n=== Sinogram diagnostics ===\n');
for ni = 1:n_noise
    I0           = I0_values(ni);
    transmission = exp(-P_scale);
    counts       = poissrnd(I0 * transmission);
    counts       = max(counts, 1);
    P_temp       = -log(counts / I0);
    P_temp       = max(P_temp, 0);
    P_noisy{ni}  = P_temp * max(P_clean(:));
    fprintf('%s: min=%.4f  max=%.4f  mean=%.4f\n', ...
        noise_labels{ni}, min(P_noisy{ni}(:)), max(P_noisy{ni}(:)), mean(P_noisy{ni}(:)));
end

%% Build ordered subsets
% Angles are assigned round-robin across subsets for uniform angular coverage
subset_indices = cell(n_subsets, 1);
for s = 1:n_subsets
    subset_indices{s} = s:n_subsets:n_angles;
end

%% Parameters - LAMBDA = 1.0 ONLY
relaxation = 1.0;

%% Storage
rmse_fbp_noisy   = zeros(n_noise, 1);
psnr_fbp_noisy   = zeros(n_noise, 1);
ssim_fbp_noisy   = zeros(n_noise, 1);
fbp_noisy_recons_raw = cell(n_noise, 1);

final_rmse_art     = zeros(n_noise, 1);
final_psnr_art     = zeros(n_noise, 1);
final_ssim_art     = zeros(n_noise, 1);
final_rmse_ossart  = zeros(n_noise, 1);
final_psnr_ossart  = zeros(n_noise, 1);
final_ssim_ossart  = zeros(n_noise, 1);

final_recons_art_raw    = cell(n_noise, 1);
final_recons_ossart_raw = cell(n_noise, 1);

for ni = 1:n_noise
    P = P_noisy{ni};
    fprintf('\n=== %s (I0=%.0e, λ=%.1f) ===\n', noise_labels{ni}, I0_values(ni), relaxation);

    %% FBP on noisy sinogram
    % Same filtering pipeline as clean FBP - noise propagates through the ramp filter
    P_filtered_noisy = zeros(size(P));
    for k = 1:n_angles
        proj              = P(:,k);
        L                 = length(proj);
        n_pad             = 2^nextpow2(2*L);
        n_side            = floor((n_pad-L)/2);
        proj_padded       = [zeros(n_side,1); proj; zeros(n_pad-L-n_side,1)];
        proj_fft          = fft(proj_padded);
        freq              = ifftshift((-n_pad/2:n_pad/2-1)/n_pad)';
        H                 = abs(freq).* (0.5 + 0.5*cos(2*pi*freq));
        proj_fft_filtered = proj_fft .* H;
        proj_filtered_n   = real(ifft(proj_fft_filtered));
        P_filtered_noisy(:,k) = proj_filtered_n(n_side+1:n_side+L);
    end
    recon_fbp_noisy      = siddon_backprojection(P_filtered_noisy, angles, det_positions, N);
    recon_fbp_noisy      = max(recon_fbp_noisy, 0);
    fbp_noisy_n          = mat2gray(recon_fbp_noisy);
    fbp_noisy_recons_raw{ni} = recon_fbp_noisy;
    rmse_fbp_noisy(ni)   = sqrt(mean((recon_fbp_noisy(:) - phantom_img(:)).^2));
    psnr_fbp_noisy(ni)   = psnr(fbp_noisy_n, gt_n);
    ssim_fbp_noisy(ni)   = ssim(fbp_noisy_n, gt_n);
    fprintf('  FBP      -> RMSE=%.5f | PSNR=%.2f | SSIM=%.4f\n', ...
        rmse_fbp_noisy(ni), psnr_fbp_noisy(ni), ssim_fbp_noisy(ni));

    %% ART
    % Kaczmarz update applied ray by ray - noise from each measurement
    % propagates sequentially, causing semiconvergence under noisy conditions
    recon_art = zeros(N, N);
    for iter = 1:num_iterations
        for a = 1:n_angles
            theta = angles(a) * pi/180;
            for d = 1:n_detectors
                s_pos = det_positions(d);
                [pixels, lengths] = siddon_ray(s_pos, theta, N, center);
                if isempty(pixels); continue; end
                
                idx     = sub2ind([N,N], pixels(:,1), pixels(:,2));
                ray_sum = sum(recon_art(idx) .* lengths);
                l_sum   = sum(lengths.^2);
                
                if l_sum > 1e-12
                    correction = relaxation * (P(d,a) - ray_sum) / l_sum;
                    recon_art(idx) = recon_art(idx) + correction .* lengths;
                end
            end
        end
        if mod(iter, 10) == 0
            rmse_iter = sqrt(mean((recon_art(:) - phantom_img(:)).^2));
            fprintf('    ART iter %d/%d RMSE: %.5f\n', iter, num_iterations, rmse_iter);
        end
    end
    
    recon_art = max(recon_art, 0);
    art_n = mat2gray(recon_art);
    final_recons_art_raw{ni} = recon_art;
    final_rmse_art(ni) = sqrt(mean((recon_art(:) - phantom_img(:)).^2));
    final_psnr_art(ni) = psnr(art_n, gt_n);
    final_ssim_art(ni) = ssim(art_n, gt_n);
    fprintf('  ART      -> RMSE=%.5f | PSNR=%.2f | SSIM=%.4f\n', ...
        final_rmse_art(ni), final_psnr_art(ni), final_ssim_art(ni));

    %% OS-SART
    % Updates are deferred until all rays in a subset are processed,
    % averaging corrections across rays to partially cancel Poisson noise
    recon_ossart = zeros(N, N);
    for iter = 1:num_iterations
        for s = 1:n_subsets
            angle_idx = subset_indices{s};
            numerator   = zeros(N, N);
            denominator = zeros(N, N);

            for ai = 1:length(angle_idx)
                a     = angle_idx(ai);
                theta = angles(a) * pi/180;
                for d = 1:n_detectors
                    s_pos = det_positions(d);
                    [pixels, lengths] = siddon_ray(s_pos, theta, N, center);
                    if isempty(pixels); continue; end
                    
                    idx       = sub2ind([N,N], pixels(:,1), pixels(:,2));
                    ray_sum   = sum(recon_ossart(idx) .* lengths);
                    ray_total = sum(lengths);
                    
                    if ray_total > 1e-12
                        discrepancy = (P(d,a) - ray_sum) / ray_total;
                        numerator(idx)   = numerator(idx) + lengths .* discrepancy;
                        denominator(idx) = denominator(idx) + lengths;
                    end
                end
            end

            % Apply length-weighted average correction for this subset
            mask = denominator > 1e-12;
            recon_ossart(mask) = recon_ossart(mask) + ...
                relaxation * numerator(mask) ./ denominator(mask);
        end

        if mod(iter, 10) == 0
            rmse_iter = sqrt(mean((recon_ossart(:) - phantom_img(:)).^2));
            fprintf('    OS-SART iter %d/%d RMSE: %.5f\n', iter, num_iterations, rmse_iter);
        end
    end

    recon_ossart = max(recon_ossart, 0);
    ossart_n = mat2gray(recon_ossart);
    final_recons_ossart_raw{ni} = recon_ossart;
    final_rmse_ossart(ni) = sqrt(mean((recon_ossart(:) - phantom_img(:)).^2));
    final_psnr_ossart(ni) = psnr(ossart_n, gt_n);
    final_ssim_ossart(ni) = ssim(ossart_n, gt_n);
    fprintf('  OS-SART  -> RMSE=%.5f | PSNR=%.2f | SSIM=%.4f\n', ...
        final_rmse_ossart(ni), final_psnr_ossart(ni), final_ssim_ossart(ni));
end

%% Print summary table
fprintf('\n%s\n', repmat('=',1,80));
fprintf('SUMMARY: MEDIUM AND HIGH NOISE (λ=1.0)\n');
fprintf('%s\n', repmat('=',1,80));
fprintf('%-20s | %-10s | %-10s | %-10s | %-10s\n', 'Method', 'Noise', 'RMSE', 'PSNR (dB)', 'SSIM');
fprintf('%s\n', repmat('-',1,80));

for ni = 1:n_noise
    fprintf('%-20s | %-10s | %10.4f | %10.2f | %10.4f\n', ...
        'FBP', noise_labels{ni}, rmse_fbp_noisy(ni), psnr_fbp_noisy(ni), ssim_fbp_noisy(ni));
    fprintf('%-20s | %-10s | %10.4f | %10.2f | %10.4f\n', ...
        'ART', noise_labels{ni}, final_rmse_art(ni), final_psnr_art(ni), final_ssim_art(ni));
    fprintf('%-20s | %-10s | %10.4f | %10.2f | %10.4f\n', ...
        'OS-SART', noise_labels{ni}, final_rmse_ossart(ni), final_psnr_ossart(ni), final_ssim_ossart(ni));
    fprintf('%s\n', repmat('-',1,80));
end

%% Figure - comparison across noise levels
figure('Position', [50 50 1800 400]);

subplot(2, 4, 1);
imagesc(fbp_noisy_recons_raw{1}); colormap gray; axis image off;
title(sprintf('Medium Noise - FBP\nRMSE=%.4f', rmse_fbp_noisy(1)), 'FontSize', 9);

subplot(2, 4, 2);
imagesc(final_recons_art_raw{1}); colormap gray; axis image off;
title(sprintf('Medium Noise - ART\nRMSE=%.4f', final_rmse_art(1)), 'FontSize', 9);

subplot(2, 4, 3);
imagesc(final_recons_ossart_raw{1}); colormap gray; axis image off;
title(sprintf('Medium Noise - OS-SART\nRMSE=%.4f', final_rmse_ossart(1)), 'FontSize', 9);

subplot(2, 4, 4);
imagesc(phantom_img); colormap gray; axis image off;
title('Ground Truth', 'FontSize', 9);

subplot(2, 4, 5);
imagesc(fbp_noisy_recons_raw{2}); colormap gray; axis image off;
title(sprintf('High Noise - FBP\nRMSE=%.4f', rmse_fbp_noisy(2)), 'FontSize', 9);

subplot(2, 4, 6);
imagesc(final_recons_art_raw{2}); colormap gray; axis image off;
title(sprintf('High Noise - ART\nRMSE=%.4f', final_rmse_art(2)), 'FontSize', 9);

subplot(2, 4, 7);
imagesc(final_recons_ossart_raw{2}); colormap gray; axis image off;
title(sprintf('High Noise - OS-SART\nRMSE=%.4f', final_rmse_ossart(2)), 'FontSize', 9);

subplot(2, 4, 8);
imagesc(phantom_img); colormap gray; axis image off;
title('Ground Truth', 'FontSize', 9);

sgtitle('ART vs OS-SART Comparison (λ=1.0)', 'FontSize', 12, 'FontWeight', 'bold');