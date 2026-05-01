%% Setup (NO NOISE) ART VS FBP
% Noiseless comparison to isolate the effect of sparse angular sampling
N = 256;
phantom_img = phantom(N);
center = (N+1)/2;
relaxation = 1;
num_iterations = 30;

n_detectors = 2*ceil(N/sqrt(2)) + 1;
det_positions = linspace(-N/sqrt(2), N/sqrt(2), n_detectors);

%% Define sparse angle sets to test
% Three increasingly sparse configurations to observe degradation
angle_sets = {
    0:9:179,  '20 angles';
    0:10:179, '18 angles';
    0:12:179, '15 angles';
};
n_sets = size(angle_sets, 1);

%% Preallocate metrics
rmse_fbp_all = zeros(n_sets,1);
psnr_fbp_all = zeros(n_sets,1);
ssim_fbp_all = zeros(n_sets,1);
rmse_art_all = zeros(n_sets,1);
psnr_art_all = zeros(n_sets,1);
ssim_art_all = zeros(n_sets,1);

%% Preallocate convergence storage
art_convergence      = zeros(num_iterations, n_sets);
ssim_art_convergence = zeros(num_iterations, n_sets);

gt_n = mat2gray(phantom_img);

%% Single figure for all reconstructions
fig_recon = figure('Position', [100 100 900 180*n_sets]);

%% Loop over angle sets
for s = 1:n_sets
    angles   = angle_sets{s,1};
    label    = angle_sets{s,2};
    n_angles = length(angles);
    fprintf('\n=== %s ===\n', label);

    %% Forward projection
    % Generate sinogram using Siddon ray tracing for this angle set
    P = siddon_forward(phantom_img, angles, det_positions, N);

    %% FBP (Hann filter)
    % Filter each projection before back projection to correct 1/rho weighting
    P_filtered = zeros(size(P));
    for k = 1:n_angles
        proj              = P(:,k);
        L                 = length(proj);
        n_pad             = 2^nextpow2(2*L);
        n_side            = floor((n_pad-L)/2);
        proj_padded       = [zeros(n_side,1); proj; zeros(n_pad-L-n_side,1)];
        proj_fft          = fft(proj_padded);
        freq              = ifftshift((-n_pad/2:n_pad/2-1)/n_pad)';
        H                 = abs(freq) .* (0.5 + 0.5*cos(2*pi*freq));
        proj_fft_filtered = proj_fft .* H;
        proj_filtered     = real(ifft(proj_fft_filtered));
        P_filtered(:,k)   = proj_filtered(n_side+1:n_side+L);
    end
    recon_fbp = siddon_backprojection(P_filtered, angles, det_positions, N);
    recon_fbp = max(recon_fbp, 0);

    %% ART
    % Kaczmarz update enforces consistency with each ray in turn.
    % Under sparse angles, ART converges to a smooth solution rather
    % than producing streak artefacts as FBP does.
    recon_art        = zeros(N,N);
    recon_art_iters  = cell(num_iterations, 1);
    ssim_art_iters   = zeros(num_iterations, 1);

    for iter = 1:num_iterations
        for a = 1:n_angles
            theta = angles(a) * pi/180;
            for d = 1:n_detectors
                s_pos = det_positions(d);
                [pixels, lengths] = siddon_ray(s_pos, theta, N, center);
                if isempty(pixels), continue; end
                idx     = sub2ind([N,N], pixels(:,1), pixels(:,2));
                ray_sum = sum(recon_art(idx) .* lengths);
                l_sum   = sum(lengths.^2);
                if l_sum > 1e-12
                    correction     = relaxation * (P(d,a) - ray_sum) / l_sum;
                    recon_art(idx) = max(recon_art(idx) + correction * lengths, 0);
                end
            end
        end
        % Track RMSE and SSIM at each iteration to observe convergence
        RMSE                          = sqrt(mean((recon_art(:) - phantom_img(:)).^2));
        art_convergence(iter, s)      = RMSE;
        recon_art_iters{iter}         = mat2gray(recon_art);
        ssim_art_iters(iter)          = ssim(recon_art_iters{iter}, gt_n);
        ssim_art_convergence(iter, s) = ssim_art_iters(iter);
        fprintf('  ART iter %d/%d RMSE: %.5f | SSIM: %.4f\n', ...
            iter, num_iterations, RMSE, ssim_art_iters(iter));
    end

    %% Normalise
    fbp_n = mat2gray(recon_fbp);
    art_n = mat2gray(recon_art);

    %% Metrics
    rmse_fbp_all(s) = sqrt(mean((recon_fbp(:) - phantom_img(:)).^2));
    rmse_art_all(s) = sqrt(mean((recon_art(:) - phantom_img(:)).^2));
    psnr_fbp_all(s) = psnr(fbp_n, gt_n);
    ssim_fbp_all(s) = ssim(fbp_n, gt_n);
    psnr_art_all(s) = psnr(art_n, gt_n);
    ssim_art_all(s) = ssim(art_n, gt_n);

    fprintf('FBP -> RMSE: %.5f | PSNR: %.2f dB | SSIM: %.4f\n', ...
        rmse_fbp_all(s), psnr_fbp_all(s), ssim_fbp_all(s));
    fprintf('ART -> RMSE: %.5f | PSNR: %.2f dB | SSIM: %.4f\n', ...
        rmse_art_all(s), psnr_art_all(s), ssim_art_all(s));

    %% Add to reconstruction figure
    figure(fig_recon);
    subplot(2, n_sets, s);
    imagesc(fbp_n); colormap gray; axis image off;
    title(label, 'FontSize', 8);
    if s == 1, ylabel('FBP (Hann)', 'FontSize', 9); end

    subplot(2, n_sets, n_sets + s);
    imagesc(art_n); colormap gray; axis image off;
    if s == 1, ylabel(sprintf('ART (%d iters)', num_iterations), 'FontSize', 9); end
end
sgtitle('FBP vs ART - Sparse Angle Comparison', 'FontSize', 13);

%% ART reconstruction at each iteration with RMSE and SSIM
% Visual inspection of how quickly structure is recovered iteration by iteration
n_cols = 5;
n_rows = ceil(num_iterations / n_cols);
figure('Position', [100 100 220*n_cols 280*n_rows]);
for iter = 1:num_iterations
    subplot(n_rows, n_cols, iter);
    imagesc(recon_art_iters{iter}, [0 1]); colormap gray; axis image off;
    title(sprintf('Iter %d\nRMSE=%.4f\nSSIM=%.4f', ...
        iter, art_convergence(iter,s), ssim_art_iters(iter)), 'FontSize', 7);
end
sgtitle(sprintf('ART Reconstruction Progress - %s', label), 'FontSize', 11);

%% Convergence figure - RMSE and SSIM side by side
% Both metrics are tracked to distinguish pixel-wise and structural convergence
figure('Position', [100 100 1200 220*n_sets]);
colors = {'r','b','g','m','k'};
for s = 1:n_sets
    subplot(n_sets, 2, (s-1)*2 + 1);
    plot(1:num_iterations, art_convergence(:,s), '-o', ...
        'Color', colors{s}, 'LineWidth', 2, 'MarkerFaceColor', colors{s});
    ylabel('RMSE');
    title(sprintf('ART RMSE Convergence - %s', angle_sets{s,2}));
    xlabel('Iteration');
    grid on;
    ylim([0 max(art_convergence(:))*1.1]);

    subplot(n_sets, 2, (s-1)*2 + 2);
    plot(1:num_iterations, ssim_art_convergence(:,s), '-o', ...
        'Color', colors{s}, 'LineWidth', 2, 'MarkerFaceColor', colors{s});
    ylabel('SSIM');
    title(sprintf('ART SSIM Convergence - %s', angle_sets{s,2}));
    xlabel('Iteration');
    grid on;
    ylim([0 1]);
end
sgtitle('ART Convergence per Angle Set', 'FontSize', 13);

%% Summary metrics figure
% Shows how RMSE, PSNR, and SSIM degrade as angular sampling decreases
n_angle_counts = cellfun(@length, angle_sets(:,1));
figure('Position', [100 100 1200 400]);

subplot(1,3,1);
plot(n_angle_counts, rmse_fbp_all, 'r-o', 'LineWidth', 2, 'DisplayName', 'FBP'); hold on;
plot(n_angle_counts, rmse_art_all, 'b-o', 'LineWidth', 2, 'DisplayName', 'ART');
xlabel('Number of Angles'); ylabel('RMSE'); title('RMSE vs Angles');
legend; grid on; set(gca, 'XDir', 'reverse');

subplot(1,3,2);
plot(n_angle_counts, psnr_fbp_all, 'r-o', 'LineWidth', 2, 'DisplayName', 'FBP'); hold on;
plot(n_angle_counts, psnr_art_all, 'b-o', 'LineWidth', 2, 'DisplayName', 'ART');
xlabel('Number of Angles'); ylabel('PSNR (dB)'); title('PSNR vs Angles');
legend; grid on; set(gca, 'XDir', 'reverse');

subplot(1,3,3);
plot(n_angle_counts, ssim_fbp_all, 'r-o', 'LineWidth', 2, 'DisplayName', 'FBP'); hold on;
plot(n_angle_counts, ssim_art_all, 'b-o', 'LineWidth', 2, 'DisplayName', 'ART');
xlabel('Number of Angles'); ylabel('SSIM'); title('SSIM vs Angles');
legend; grid on; set(gca, 'XDir', 'reverse');

sgtitle('FBP vs ART Performance Under Sparse Angles');