clear; clc; close all;
cfg.num_subcarriers = 3276;
cfg.num_symbols = 14;
cfg.dmrs_col_index = 4;
cfg.N_FFT = 4096;
cfg.subcarrier_spacing = 30e3;
cfg.sample_rate = cfg.N_FFT * cfg.subcarrier_spacing;
cfg.CP_len_first = 352;
cfg.CP_len_other = 288;
cfg.path_delays_ns = [0, 50, 120, 200, 350];
cfg.avg_path_gains_dB = [0, -2, -5, -8, -12];
cfg.max_doppler_shift = 25;
cfg.snr_range_dB = 0:2:30;
cfg.modulation_schemes = {'QPSK', '16QAM', '64QAM'};
cfg.num_grids_to_simulate = 100;
fprintf('Khởi tạo môi trường mô phỏng...\n');

try
    fileID = fopen('l1_mini_project_ce_eq_dmrs_data.txt', 'r');
    dmrs_string = textscan(fileID, '%s');
    fclose(fileID);
    R_col = str2double(dmrs_string{1});
    R_col = R_col(:);
    disp('Đọc file DMRS thành công.');
catch
    error('Không tìm thấy file DMRS!');
end

num_snr_points = length(cfg.snr_range_dB);
results_ber = cell(length(cfg.modulation_schemes), 1);
results_mse = cell(length(cfg.modulation_schemes), 1);
for i = 1:length(cfg.modulation_schemes)
    results_ber{i} = zeros(3, num_snr_points);
    results_mse{i} = zeros(3, num_snr_points);
end

all_col_indices = 1:cfg.num_symbols;
data_col_indices = setdiff(all_col_indices, cfg.dmrs_col_index);
pilot_indices = find(R_col ~= 0);
if isempty(pilot_indices), error('DMRS không có tín hiệu tham chiếu.'); end

H_corr_matrix = get_channel_correlation(cfg);
R_HH_pilots = H_corr_matrix(pilot_indices, pilot_indices);

fprintf('Bắt đầu mô phỏng...\n');
for mod_idx = 1:length(cfg.modulation_schemes)
    mod_name = cfg.modulation_schemes{mod_idx};
    switch mod_name, case 'QPSK', M = 4; case '16QAM', M = 16; case '64QAM', M = 64; end
    k = log2(M);
    num_data_re = cfg.num_subcarriers * (cfg.num_symbols - 1);
    num_bits = num_data_re * k;
    
    fprintf('--- Đang chạy cho điều chế: %s ---\n', mod_name);
    
    for snr_idx = 1:num_snr_points
        snr_dB = cfg.snr_range_dB(snr_idx);
        snr_linear = 10^(snr_dB / 10);
        
        total_bit_errors = zeros(3, 1);
        total_squared_error = zeros(3, 1);
        
        for grid_idx = 1:cfg.num_grids_to_simulate
            
            %% Khối: Tạo Dữ liệu & Điều chế
            % - Tạo chuỗi bit ngẫu nhiên.
            % - Thực hiện điều chế (QPSK/16QAM/64QAM) để tạo các ký hiệu phức.
            tx_bits = randi([0 1], num_bits, 1);
            tx_qam_symbols = qammod(tx_bits, M, 'InputType', 'bit', 'UnitAveragePower', true);
            original_qam_symbols_vec = tx_qam_symbols;

            %% Khối: Ánh xạ Lưới Tài nguyên
            % - Ánh xạ các ký hiệu dữ liệu và ký hiệu tham chiếu (DMRS) vào lưới tài nguyên.
            resource_grid_tx = zeros(cfg.num_subcarriers, cfg.num_symbols);
            data_matrix = reshape(tx_qam_symbols, cfg.num_subcarriers, cfg.num_symbols - 1);
            resource_grid_tx(:, data_col_indices) = data_matrix;
            resource_grid_tx(:, cfg.dmrs_col_index) = R_col;
            
            %% Khối: Điều chế OFDM
            % - Thực hiện biến đổi IFFT để chuyển tín hiệu sang miền thời gian.
            % - Thêm Tiền tố Tuần hoàn (CP) vào đầu mỗi ký hiệu OFDM.
            ifft_input = zeros(cfg.N_FFT, cfg.num_symbols);
            start_idx_fft = floor((cfg.N_FFT - cfg.num_subcarriers) / 2) + 1;
            end_idx_fft = start_idx_fft + cfg.num_subcarriers - 1;
            ifft_input(start_idx_fft:end_idx_fft, :) = resource_grid_tx;
            
            time_domain_signal_no_cp = ifft(ifft_input, cfg.N_FFT, 1);
            
            tx_signal_len = (cfg.N_FFT + cfg.CP_len_first) + (cfg.N_FFT + cfg.CP_len_other) * (cfg.num_symbols - 1);
            tx_signal = zeros(tx_signal_len, 1);
            
            current_pos = 1;
            len = cfg.N_FFT + cfg.CP_len_first;
            symbol_with_cp = [time_domain_signal_no_cp(end-cfg.CP_len_first+1:end, 1); time_domain_signal_no_cp(:, 1)];
            tx_signal(current_pos : current_pos + len - 1) = symbol_with_cp;
            current_pos = current_pos + len;
            for l = 2:cfg.num_symbols
                len = cfg.N_FFT + cfg.CP_len_other;
                symbol_with_cp = [time_domain_signal_no_cp(end-cfg.CP_len_other+1:end, l); time_domain_signal_no_cp(:, l)];
                tx_signal(current_pos : current_pos + len - 1) = symbol_with_cp;
                current_pos = current_pos + len;
            end

            %% Khối: Kênh Fading & Nhiễu
            % - Cho tín hiệu đi qua kênh truyền Fading theo chuẩn TDLB100-25.
            % - Tiếp tục cho tín hiệu đi qua nhiễu Gauss với các mức SNR khác nhau.
            [fading_channel_taps, ~] = generate_time_varying_channel(cfg, tx_signal_len);
            rx_signal_faded = fftfilt(fading_channel_taps, tx_signal);
            rx_signal = awgn(rx_signal_faded, snr_dB, 'measured');

            %% Khối: Giải điều chế OFDM
            % - Loại bỏ Tiền tố Tuần hoàn (CP).
            % - Thực hiện biến đổi FFT để chuyển tín hiệu về lại miền tần số.
            rx_time_domain_no_cp = zeros(cfg.N_FFT, cfg.num_symbols);
            current_pos = 1;
            len_cp = cfg.CP_len_first;
            rx_time_domain_no_cp(:, 1) = rx_signal(current_pos + len_cp : current_pos + len_cp + cfg.N_FFT - 1);
            current_pos = current_pos + cfg.N_FFT + len_cp;
            for l = 2:cfg.num_symbols
                len_cp = cfg.CP_len_other;
                rx_time_domain_no_cp(:, l) = rx_signal(current_pos + len_cp : current_pos + len_cp + cfg.N_FFT - 1);
                current_pos = current_pos + cfg.N_FFT + len_cp;
            end

            freq_domain_signal = fft(rx_time_domain_no_cp, cfg.N_FFT, 1);
            resource_grid_rx = freq_domain_signal(start_idx_fft:end_idx_fft, :);

            Y_data_matrix = resource_grid_rx(:, data_col_indices);
            Y_dmrs_col = resource_grid_rx(:, cfg.dmrs_col_index);
            
            %% Khối: Ước lượng Kênh
            % - Ước lượng kênh truyền bằng thuật toán LS.
            % - Thay thế thuật toán LS bằng thuật toán ước lượng kênh nâng cao MMSE.
            H_ls_at_pilots = Y_dmrs_col(pilot_indices) ./ R_col(pilot_indices);
            
            I_pilot = eye(length(pilot_indices));
            filter_mmse = R_HH_pilots * inv(R_HH_pilots + (1/snr_linear) * I_pilot); 
            H_mmse_at_pilots = filter_mmse * H_ls_at_pilots;
            
            H_ls_dmrs_col_full = interp1(pilot_indices, H_ls_at_pilots, 1:cfg.num_subcarriers, 'linear', 'extrap').';
            H_mmse_dmrs_col_full = interp1(pilot_indices, H_mmse_at_pilots, 1:cfg.num_subcarriers, 'linear', 'extrap').';
            
            H_est_ls = repmat(H_ls_dmrs_col_full, 1, cfg.num_symbols-1);
            H_est_mmse = repmat(H_mmse_dmrs_col_full, 1, cfg.num_symbols-1);

            %% Khối: Cân bằng Kênh
            % - Cân bằng kênh truyền bằng thuật toán ZF.
            % - Sử dụng bộ cân bằng MMSE.
            X_est_no_eq = Y_data_matrix(:);
            
            X_est_zf = Y_data_matrix ./ H_est_ls;
            X_est_zf = X_est_zf(:);
            
            mmse_eq_filter = conj(H_est_mmse) ./ (abs(H_est_mmse).^2 + 1/snr_linear);
            X_est_mmse = Y_data_matrix .* mmse_eq_filter;
            X_est_mmse = X_est_mmse(:);
            
            %% Khối: Giải điều chế QAM
            % - Giải điều chế QAM hardbit.
            rx_bits_no_eq = qamdemod(X_est_no_eq, M, 'OutputType', 'bit', 'UnitAveragePower', true);
            rx_bits_zf = qamdemod(X_est_zf, M, 'OutputType', 'bit', 'UnitAveragePower', true);
            rx_bits_mmse = qamdemod(X_est_mmse, M, 'OutputType', 'bit', 'UnitAveragePower', true);

            %% Khối: Tính toán BER & MSE
            % - So sánh sự sai lệch dữ liệu giữ ký hiệu QAM đầu phát và đầu thu theo phương pháp MSE.
            % - So sánh và tính toán sai lệch bit (BER) giữa đầu phát và đầu thu.
            total_bit_errors(1) = total_bit_errors(1) + biterr(tx_bits, rx_bits_no_eq);
            total_bit_errors(2) = total_bit_errors(2) + biterr(tx_bits, rx_bits_zf);
            total_bit_errors(3) = total_bit_errors(3) + biterr(tx_bits, rx_bits_mmse);
            
            total_squared_error(1) = total_squared_error(1) + mean(abs(X_est_no_eq - original_qam_symbols_vec).^2);
            total_squared_error(2) = total_squared_error(2) + mean(abs(X_est_zf - original_qam_symbols_vec).^2);
            total_squared_error(3) = total_squared_error(3) + mean(abs(X_est_mmse - original_qam_symbols_vec).^2);
        end
        
        ber_avg = total_bit_errors / (num_bits * cfg.num_grids_to_simulate);
        mse_avg = total_squared_error / cfg.num_grids_to_simulate;
        
        results_ber{mod_idx}(:, snr_idx) = ber_avg;
        results_mse{mod_idx}(:, snr_idx) = mse_avg;
        
        fprintf('SNR: %2.0f dB | BER (LS/ZF): %e | BER (MMSE): %e\n', snr_dB, ber_avg(2), ber_avg(3));
    end
end
fprintf('Mô phỏng hoàn tất!\n');

%% Vẽ biểu đồ
for mod_idx = 1:length(cfg.modulation_schemes)
    mod_name = cfg.modulation_schemes{mod_idx};
    figure('Name', ['Performance for ' mod_name], 'NumberTitle', 'off', 'Position', [100, 100, 1200, 500]);
    subplot(1, 2, 1);
    semilogy(cfg.snr_range_dB, results_ber{mod_idx}(1,:), 'k--x', 'LineWidth', 1, 'DisplayName', 'Không cân bằng'); hold on;
    semilogy(cfg.snr_range_dB, results_ber{mod_idx}(2,:), 'b-o', 'LineWidth', 2, 'DisplayName', 'LS/ZF');
    semilogy(cfg.snr_range_dB, results_ber{mod_idx}(3,:), 'r-s', 'LineWidth', 2, 'DisplayName', 'MMSE/MMSE');
    grid on; ylim([1e-5 1]); title(['BER vs. SNR cho ' mod_name]); xlabel('SNR (dB)'); ylabel('Bit Error Rate (BER)'); legend('show', 'Location', 'southwest');
    subplot(1, 2, 2);
    semilogy(cfg.snr_range_dB, results_mse{mod_idx}(1,:), 'k--x', 'LineWidth', 1, 'DisplayName', 'Không cân bằng'); hold on;
    semilogy(cfg.snr_range_dB, results_mse{mod_idx}(2,:), 'b-o', 'LineWidth', 2, 'DisplayName', 'LS/ZF');
    semilogy(cfg.snr_range_dB, results_mse{mod_idx}(3,:), 'r-s', 'LineWidth', 2, 'DisplayName', 'MMSE/MMSE');
    grid on; title(['MSE vs. SNR cho ' mod_name]); xlabel('SNR (dB)'); ylabel('Mean Squared Error (MSE)'); legend('show', 'Location', 'southwest');
end

%% Các hàm hỗ trợ (do không cài được 5G toolbox)
function [channel_taps_time, H_true] = generate_time_varying_channel(cfg, total_samples)
    path_delays_samples = round(cfg.path_delays_ns * 1e-9 * cfg.sample_rate);
    avg_path_gains_linear = 10.^(cfg.avg_path_gains_dB / 20);
    num_paths = length(path_delays_samples);
    num_oscillators = 20; 
    alpha = 2 * pi * rand(num_paths, num_oscillators); 
    phi = 2 * pi * rand(num_paths, num_oscillators);
    t = (0:total_samples-1)' / cfg.sample_rate;
    fading_process = zeros(total_samples, num_paths);
    for p = 1:num_paths
        doppler_freqs = cfg.max_doppler_shift * cos(alpha(p, :));
        path_fading = sum(exp(1j * (2 * pi * t * doppler_freqs + phi(p, :))), 2);
        fading_process(:, p) = (path_fading / sqrt(num_oscillators)) * avg_path_gains_linear(p);
    end
    max_delay = max(path_delays_samples);
    channel_taps_time = zeros(total_samples, max_delay + 1);
    for p = 1:num_paths
        delay = path_delays_samples(p);
        channel_taps_time(:, delay + 1) = fading_process(:, p);
    end
    H_true = 0;
end

function H_corr_matrix = get_channel_correlation(cfg)
    path_delays_samples = round(cfg.path_delays_ns * 1e-9 * cfg.sample_rate);
    avg_path_gains_linear_power = 10.^(cfg.avg_path_gains_dB / 10);
    pdp_time = zeros(cfg.N_FFT, 1);
    valid_indices = path_delays_samples < cfg.N_FFT;
    pdp_time(path_delays_samples(valid_indices) + 1) = avg_path_gains_linear_power(valid_indices);
    R_hh_freq_full = fft(pdp_time, cfg.N_FFT);
    start_idx_fft = floor((cfg.N_FFT - cfg.num_subcarriers) / 2) + 1;
    end_idx_fft = start_idx_fft + cfg.num_subcarriers - 1;
    R_hh_freq = R_hh_freq_full(start_idx_fft:end_idx_fft);
    H_corr_matrix = diag(abs(R_hh_freq));
end