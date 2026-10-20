#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include "meanrevML/types.hpp"
#include "meanrevML/features.hpp"
#include "meanrevML/labeling.hpp"
#include "meanrevML/cpcv.hpp"
#include "meanrevML/rf.hpp"
#include "meanrevML/pbo.hpp"
#include "meanrevML/io.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <ctime>

using namespace meanrevML;





static std::vector<Bar> make_synthetic(int n, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> N(0.0, 1.0);
    std::vector<Bar> bars(n);
    double price = 100.0, sigma = 1.0, drift = 0.0, theta = 0.20, mu = 100.0;
    for (int i = 0; i < n; ++i) {
        if (i > 0 && i % 500 == 0) {
            bool trending = ((i / 500) % 2 == 0);
            if (trending) { theta = 0.0;  drift = 0.15; sigma = 1.2; }
            else          { theta = 0.30; drift = 0.0;  sigma = 0.8; mu = price; }
        }
        price += drift + theta * (mu - price) + sigma * N(rng);
        bars[i] = { (int64_t)i * 1'000'000LL, price,
                    price + std::abs(N(rng))*sigma,
                    price - std::abs(N(rng))*sigma, price, 1000.0 };
    }
    return bars;
}

static double eval_config_on_split(const ShallowRF::Config& cfg,
                                    const std::vector<Sample>& samples,
                                    const std::vector<double>& px,
                                    const CpcvSplit& sp,
                                    double* out_acc = nullptr,
                                    double* out_brier = nullptr,
                                    std::vector<double>* out_trades = nullptr)
{
    std::vector<std::vector<double>> Xtr, Xte;
    std::vector<int> ytr, yte;
    for (int i : sp.train_indices) { Xtr.push_back(samples[i].features); ytr.push_back(samples[i].label); }
    for (int i : sp.test_indices)  { Xte.push_back(samples[i].features); yte.push_back(samples[i].label); }
    int tr_pos = 0; for (int y : ytr) tr_pos += y;
    double rate = (double)tr_pos / ytr.size();

    ShallowRF rf(cfg);
    rf.fit(Xtr, ytr);
    auto proba = rf.predict_proba(Xte);

    double auc = compute_auc(proba, yte);
    if (out_acc) {
        int c = 0;
        for (size_t i = 0; i < yte.size(); ++i) c += ((proba[i] > rate ? 1 : 0) == yte[i]);
        *out_acc = (double)c / yte.size();
    }
    if (out_brier) {
        double b = 0;
        for (size_t i = 0; i < yte.size(); ++i) b += (proba[i] - yte[i]) * (proba[i] - yte[i]);
        *out_brier = b / yte.size();
    }
    if (out_trades) {
        out_trades->clear();
        for (size_t i = 0; i < yte.size(); ++i) {
            int sample_idx = sp.test_indices[i];
            int start_bar  = samples[sample_idx].bar_index;
            int end_bar    = samples[sample_idx].end_bar_index;
            if (end_bar >= (int)px.size()) end_bar = (int)px.size() - 1;
            if (end_bar <= start_bar)      end_bar = std::min(start_bar + 1, (int)px.size() - 1);
            if (start_bar + 1 >= (int)px.size()) continue;

            int pred_reversion = (proba[i] > rate) ? 1 : 0;
            double z_i = samples[sample_idx].features[0];  // z-score is feature 0

            int signal = pred_reversion ? (z_i > 0 ? -1 : 1)
                                        : (z_i > 0 ?  1 : -1);

            double realized = (px[end_bar] - px[start_bar]) / px[start_bar];
            out_trades->push_back(signal * realized);
        }
    }
    return auc;
    
}

int main(int argc, char** argv) {
    std::vector<Bar> bars;
    if (argc > 1) {
        bars = load_stooq_txt(argv[1]);
        std::cout << "Loaded " << bars.size() << " bars from " << argv[1] << "\n";
        if (bars.empty()) {
            std::cerr << "Failed to load — file missing or wrong format.\n";
            return 1;
        }
    } else {
        bars = make_synthetic(5000);
        std::cout << "Generated " << bars.size() << " synthetic bars\n";
    }
    int n = (int)bars.size();
    std::vector<double> px(n);
    for (int i = 0; i < n; ++i) px[i] = bars[i].close;

    auto z    = compute_zscore(px, 20);
    auto ret  = compute_returns(px);
    auto vol  = compute_rolling_vol(ret, 20);
    auto mom20  = compute_momentum(px, 20);
    auto mom100 = compute_momentum(px, 100);
    auto vr   = compute_variance_ratio(px, ret, 5, 20);
    auto tr2  = compute_trend_r2(px, 50);

    LabelConfig lc{ 5, 1.0, 1.0, 20 };
    auto samples = triple_barrier_labels(bars, z, vol, lc, 1.5);
    std::cout << "Samples: " << samples.size() << std::endl;

    for (auto& s : samples) {
        int i = s.bar_index;
        s.features = { z[i], vol[i], mom20[i], mom100[i], ret[i], vr[i], tr2[i] };
    }

    auto splits = build_cpcv_splits(samples, 10, 2, 5);
    std::cout << "CPCV splits: " << splits.size() << std::endl;

    ShallowRF::Config base_cfg;
    base_cfg.n_trees = 150; base_cfg.max_depth = 6; base_cfg.min_samples_leaf = 15;

    std::vector<double> aucs_detail, accs_detail, briers_detail;
    std::cout << "\n=== Phase 1: detailed eval with baseline config ===\n";
    std::vector<double> sharpes_detail;                       
    for (size_t k = 0; k < splits.size(); ++k) {
        double acc = 0, brier = 0;
        std::vector<double> trades;                           
        double auc = eval_config_on_split(base_cfg, samples, px, splits[k],
                                           &acc, &brier, &trades);
        aucs_detail.push_back(auc);
        accs_detail.push_back(acc);
        briers_detail.push_back(brier);
        double m = trades.empty() ? 0.0 : mean_of(trades);     
        double s = trades.size() > 1 ? std::sqrt(var_of(trades)) : 0.0;
        double sharpe = (s > 0) ? m / s : 0.0;
        sharpes_detail.push_back(sharpe);                      
        std::cout << "Split " << std::setw(2) << k
                  << "  AUC=" << std::fixed << std::setprecision(3) << auc
                  << "  acc=" << acc
                  << "  Brier=" << brier
                  << "  Sharpe=" << std::setprecision(2) << sharpe << "\n";  
    }
    std::cout << "Mean AUC:          " << std::setprecision(3) << mean_of(aucs_detail) << "\n"
              << "Mean raw accuracy: " << mean_of(accs_detail) << "\n"
              << "Mean Brier:        " << mean_of(briers_detail) << "\n"
              << "Mean Sharpe:       " << std::setprecision(2) << mean_of(sharpes_detail) << std::endl;

    std::cout << "\n=== Phase 2: PBO sweep (8 configs x "
              << splits.size() << " splits) ===\n";
    std::vector<ShallowRF::Config> configs;
    for (int depth : {4, 5, 6, 7})
        for (int leaf : {5, 10, 15, 20, 30})
            for (int trees : {50, 100}) {
                ShallowRF::Config c;
                c.n_trees = trees; c.max_depth = depth; c.min_samples_leaf = leaf;
                configs.push_back(c);
            }
    std::cout << "Configs: " << configs.size() << std::endl;

    std::vector<std::vector<double>> M(splits.size(), std::vector<double>(configs.size(), 0.5));
    std::vector<std::vector<double>> S(splits.size(), std::vector<double>(configs.size(), 0.0)); 
    for (size_t c = 0; c < configs.size(); ++c) {
        std::cout << "Config " << std::setw(2) << c
                  << " (depth=" << configs[c].max_depth
                  << ", leaf=" << configs[c].min_samples_leaf
                  << ", trees=" << configs[c].n_trees << ") ... " << std::flush;
        double sum_auc = 0, sum_sharpe = 0;
        for (size_t k = 0; k < splits.size(); ++k) {
            std::vector<double> trades;
            M[k][c] = eval_config_on_split(configs[c], samples, px, splits[k],
                                            nullptr, nullptr, &trades);
            double m = trades.empty() ? 0.0 : mean_of(trades);
            double s = trades.size() > 1 ? std::sqrt(var_of(trades)) : 0.0;
            S[k][c] = (s > 0) ? m / s : 0.0;
            sum_auc += M[k][c];
            sum_sharpe += S[k][c];
        }
        std::cout << "AUC=" << std::fixed << std::setprecision(3) << sum_auc / splits.size()
                  << "  Sharpe=" << std::setprecision(2) << sum_sharpe / splits.size() << "\n";
    }

    double pbo = compute_pbo(M, 6);   // 6 blocks, C(6,3)=20 IS/OOS partitions
    std::cout << "\n=== Phase 3: overfitting diagnostics ===\n"
              << "PBO (probability of backtest overfitting): " << pbo << "\n"
              << "  Interpretation: fraction of IS/OOS partitions where\n"
              << "  the best-in-sample config underperforms the median OOS.\n"
              << "   <0.30 good | 0.30-0.50 marginal | >0.50 overfit-prone\n";

    int best_cfg = 0;
    double best_mean_sharpe = -1e9;
    std::vector<double> config_sharpes(configs.size(), 0.0);
    for (size_t c = 0; c < configs.size(); ++c) {
        std::vector<double> col;
        for (size_t t = 0; t < splits.size(); ++t) col.push_back(S[t][c]);
        double m = mean_of(col), s = std::sqrt(var_of(col));
        config_sharpes[c] = (s > 0) ? m / s : 0.0;
        if (m > best_mean_sharpe) { best_mean_sharpe = m; best_cfg = (int)c; }
    }
    double sr_var = var_of(config_sharpes);

    std::vector<double> sharpes_best;
    for (size_t t = 0; t < splits.size(); ++t) sharpes_best.push_back(S[t][best_cfg]);

    double dsr = compute_dsr(sharpes_best, (int)configs.size(), sr_var);
    std::cout << "\nBest config: #" << best_cfg
              << " (depth=" << configs[best_cfg].max_depth
              << ", leaf=" << configs[best_cfg].min_samples_leaf
              << ", trees=" << configs[best_cfg].n_trees << ")\n"
              << "  mean OOS Sharpe:   " << best_mean_sharpe << "\n"
              << "  config Sharpe:     " << config_sharpes[best_cfg] << "\n"
              << "  sr_variance:       " << sr_var << "\n"
              << "  Deflated Sharpe:   " << dsr << "\n"
              << "  >0.95 publishable | 0.80-0.95 suggestive | <0.80 likely noise\n";

    return 0;
}