#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "types.hpp"

namespace meanrevML {

inline std::vector<double> compute_returns(const std::vector<double>& p) {
    int n = p.size();
    std::vector<double> r(n, 0.0);
    for (int i = 1; i < n; ++i)
        r[i] = (p[i] - p[i-1]) / std::max(std::abs(p[i-1]), 1e-12);
    return r;
}

inline std::vector<double> compute_zscore(const std::vector<double>& p, int lb) {
    int n = p.size();
    std::vector<double> z(n, 0.0);
    if (n <= lb) return z;
    double s = 0, sq = 0;
    for (int i = 0; i < lb; ++i) { s += p[i]; sq += p[i]*p[i]; }
    for (int i = lb; i < n; ++i) {
        double mean = s / lb;
        double var  = std::max(sq/lb - mean*mean, 1e-12);
        z[i] = (p[i] - mean) / std::sqrt(var);
        s  += p[i] - p[i-lb];
        sq += p[i]*p[i] - p[i-lb]*p[i-lb];
    }
    return z;
}

inline std::vector<double> compute_rolling_vol(const std::vector<double>& r, int lb) {
    int n = r.size();
    std::vector<double> v(n, 0.0);
    if (n <= lb) return v;
    double s = 0, sq = 0;
    for (int i = 1; i <= lb; ++i) { s += r[i]; sq += r[i]*r[i]; }
    for (int i = lb + 1; i < n; ++i) {
        double mean = s / lb;
        v[i] = std::sqrt(std::max(sq/lb - mean*mean, 1e-18));
        s  += r[i] - r[i-lb];
        sq += r[i]*r[i] - r[i-lb]*r[i-lb];
    }
    return v;
}

inline std::vector<double> compute_momentum(const std::vector<double>& p, int lb) {
    int n = p.size();
    std::vector<double> m(n, 0.0);
    for (int i = lb; i < n; ++i)
        m[i] = p[i] / std::max(std::abs(p[i-lb]), 1e-12) - 1.0;
    return m;
}

inline std::vector<double> compute_variance_ratio(
    const std::vector<double>& p, const std::vector<double>& r, int k, int lb)
{
    int n = (int)p.size();
    std::vector<double> vr(n, 1.0);
    for (int i = std::max(lb + k, lb + 1); i < n; ++i) {
        double s1=0, s2=0, s3=0, s4=0;
        for (int j = i-lb+1; j <= i; ++j) {
            double rk = (p[j] - p[j-k]) / std::max(std::abs(p[j-k]), 1e-12);
            s1 += rk; s2 += rk*rk;
            s3 += r[j]; s4 += r[j]*r[j];
        }
        double vk = std::max(s2/lb - (s1/lb)*(s1/lb), 1e-18);
        double v1 = std::max(s4/lb - (s3/lb)*(s3/lb), 1e-18);
        vr[i] = vk / (k * v1);
    }
    return vr;
}
inline std::vector<double> compute_trend_r2(const std::vector<double>& p, int lb) {
    int n = (int)p.size();
    std::vector<double> r2(n, 0.0);
    for (int i = lb - 1; i < n; ++i) {
        double sx=0, sy=0, sxx=0, sxy=0, syy=0;
        for (int j = 0; j < lb; ++j) {
            double x = j, y = p[i - lb + 1 + j];
            sx += x; sy += y; sxx += x*x; sxy += x*y; syy += y*y;
        }
        double denom = lb*sxx - sx*sx;
        if (std::abs(denom) < 1e-12) continue;
        double slope = (lb*sxy - sx*sy) / denom;
        double intercept = (sy - slope*sx) / lb;
        double sst = syy - sy*sy/lb;
        if (sst < 1e-12) continue;
        double ssr = 0;
        for (int j = 0; j < lb; ++j) {
            double pred = intercept + slope*j;
            double actual = p[i - lb + 1 + j];
            ssr += (actual - pred) * (actual - pred);
        }
        r2[i] = 1.0 - ssr/sst;
    }
    return r2;
}
} 