#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <utility>   

namespace meanrevML {

inline double mean_of(const std::vector<double>& x) {
    double s = 0; for (double v : x) s += v; return s / x.size();
}
inline double var_of(const std::vector<double>& x) {
    double m = mean_of(x), s = 0;
    for (double v : x) s += (v - m) * (v - m);
    return s / (x.size() - 1);
}
inline double skew_of(const std::vector<double>& x) {
    double m = mean_of(x), s = 0, s3 = 0;
    for (double v : x) { double d = v - m; s += d*d; s3 += d*d*d; }
    s = std::sqrt(s / x.size());
    return s3 / x.size() / (s*s*s);
}
inline double kurt_of(const std::vector<double>& x) {   // non-excess (normal = 3)
    double m = mean_of(x), s = 0, s4 = 0;
    for (double v : x) { double d = v - m; s += d*d; s4 += d*d*d*d; }
    s = std::sqrt(s / x.size());
    return s4 / x.size() / (s*s*s*s);
}
inline double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}
// Beasley-Springer-Moro inverse normal CDF
inline double norm_ppf(double p) {
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};
    if (p <= 0.0) return -1e10;
    if (p >= 1.0) return  1e10;
    double q, r, x;
    if (p < 0.02425) {
        q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p < 0.97575) {
        q = p - 0.5; r = q*q;
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    return x;
}

inline double compute_auc(const std::vector<double>& proba, const std::vector<int>& y) {
    std::vector<std::pair<double,int>> p;
    for (size_t i = 0; i < y.size(); ++i) p.push_back({proba[i], y[i]});
    std::sort(p.begin(), p.end());
    long long n_pos = 0, n_neg = 0, rank_sum = 0;
    for (auto& q : p) (q.second ? n_pos : n_neg)++;
    if (n_pos == 0 || n_neg == 0) return 0.5;
    for (long long i = 0; i < (long long)p.size(); ++i)
        if (p[i].second) rank_sum += (i + 1);
    return (double)(rank_sum - n_pos*(n_pos+1)/2) / (double)(n_pos * n_neg);
}

// PBO via CSCV
// M[t][n] = performance of config n on split t
// T = splits (treated as quasi-independent observations)
// N = configurations (hyperparameter sets)
// Enumerates all C(S, S/2) block combinations exhaustively.
inline double compute_pbo(const std::vector<std::vector<double>>& M, int n_blocks = 6) {
    int T = (int)M.size();
    if (T < 4 || n_blocks < 2 || n_blocks % 2 != 0 || n_blocks > T) return -1.0;
    int N = (int)M[0].size();
    if (N < 2) return -1.0;

    // Assign each split to a block
    std::vector<int> block_of(T);
    for (int t = 0; t < T; ++t) block_of[t] = std::min(t * n_blocks / T, n_blocks - 1);

    int S2 = n_blocks / 2;
    std::vector<int> combo(S2);
    std::iota(combo.begin(), combo.end(), 0);

    int overfit = 0, total = 0;
    while (true) {
        std::vector<bool> is(n_blocks, false);
        for (int c : combo) is[c] = true;

        std::vector<double> is_m(N, 0.0), oos_m(N, 0.0);
        std::vector<int> is_c(N, 0), oos_c(N, 0);
        for (int t = 0; t < T; ++t) {
            bool in_sample = is[block_of[t]];
            for (int n = 0; n < N; ++n) {
                if (in_sample) { is_m[n]  += M[t][n]; is_c[n]++;  }
                else           { oos_m[n] += M[t][n]; oos_c[n]++; }
            }
        }
        bool valid = true;
        for (int n = 0; n < N; ++n) {
            if (is_c[n] == 0 || oos_c[n] == 0) { valid = false; break; }
            is_m[n]  /= is_c[n];
            oos_m[n] /= oos_c[n];
        }
        if (valid) {
            int best = 0;
            for (int n = 1; n < N; ++n) if (is_m[n] > is_m[best]) best = n;
            int rank = 0;
            for (int n = 0; n < N; ++n) if (oos_m[n] < oos_m[best]) rank++;
            double omega = (double)(rank + 1) / (N + 1);
            double logit = std::log(omega / (1.0 - omega));
            if (logit < 0.0) overfit++;
            total++;
        }

        // Advance combination
        int i = S2 - 1;
        while (i >= 0 && combo[i] == n_blocks - S2 + i) i--;
        if (i < 0) break;
        combo[i]++;
        for (int j = i + 1; j < S2; ++j) combo[j] = combo[j-1] + 1;
    }
    return total > 0 ? (double)overfit / total : -1.0;
}

inline double compute_dsr(const std::vector<double>& returns,  // per-split edges (e.g. AUC - 0.5)
                          int n_trials,                        // number of configurations
                          double sr_variance)                  // variance of Sharpe across trials
{
    int T = (int)returns.size();
    if (T < 3 || n_trials < 2) return NAN;
    double m = mean_of(returns);
    double s = std::sqrt(var_of(returns));
    if (s <= 0.0) return NAN;
    double sr = m / s;
    double g3 = skew_of(returns);
    double g4 = kurt_of(returns);

    const double gamma_em = 0.5772156649;
    double sr0 = std::sqrt(sr_variance) *
        ((1.0 - gamma_em) * norm_ppf(1.0 - 1.0 / n_trials) +
         gamma_em * norm_ppf(1.0 - 1.0 / (n_trials * std::exp(1.0))));

    double denom_inner = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr;
    if (denom_inner <= 0.0) return NAN;
    double z = (sr - sr0) * std::sqrt((double)(T - 1)) / std::sqrt(denom_inner);
    return norm_cdf(z);
}

} 