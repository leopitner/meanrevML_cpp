#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>
#include <limits>
#include "types.hpp"

namespace meanrevML {

// Combinatorial Purged Cross-Validation with embargo.
// Splits samples into n_groups contiguous blocks; for every combination of
// n_test_groups blocks, builds a (train, test) split, purging train samples
// whose label window overlaps any test group, and embargoing samples
// immediately after each test group.
inline std::vector<CpcvSplit> build_cpcv_splits(
    const std::vector<Sample>& samples,
    int n_groups, int n_test_groups, int embargo_bars)
{
    std::vector<CpcvSplit> splits;
    int n = (int)samples.size();
    if (n == 0 || n_groups < 2 || n_test_groups < 1 || n_test_groups >= n_groups)
        return splits;

    // Assign each sample to a contiguous group
    std::vector<int> group_of(n);
    for (int i = 0; i < n; ++i)
        group_of[i] = std::min(i * n_groups / n, n_groups - 1);

    // Per-group time bounds (used for purging)
    const int64_t TMAX = std::numeric_limits<int64_t>::max();
    const int64_t TMIN = std::numeric_limits<int64_t>::min();
    std::vector<int64_t> g_start(n_groups, TMAX), g_end(n_groups, TMIN);
    for (int i = 0; i < n; ++i) {
        int g = group_of[i];
        g_start[g] = std::min(g_start[g], samples[i].start_time);
        g_end[g]   = std::max(g_end[g],   samples[i].end_time);
    }

    // Median bar spacing → embargo duration in time units
    std::vector<int64_t> spacings;
    for (int i = 1; i < n; ++i)
        spacings.push_back(samples[i].start_time - samples[i-1].start_time);
    int64_t embargo_dt = 0;
    if (!spacings.empty()) {
        std::nth_element(spacings.begin(),
                         spacings.begin() + spacings.size()/2,
                         spacings.end());
        embargo_dt = spacings[spacings.size()/2] * embargo_bars;
    }

    std::vector<int> combo;
    std::function<void(int)> gen = [&](int start) {
        if ((int)combo.size() == n_test_groups) {
            std::vector<bool> is_test(n_groups, false);
            for (int g : combo) is_test[g] = true;

            CpcvSplit sp;
            for (int i = 0; i < n; ++i)
                if (is_test[group_of[i]]) sp.test_indices.push_back(i);

            for (int i = 0; i < n; ++i) {
                if (is_test[group_of[i]]) continue;
                bool keep = true;
                for (int tg : combo) {
                    // Purge: label window overlaps test group window
                    if (samples[i].start_time <= g_end[tg] &&
                        samples[i].end_time   >= g_start[tg]) { keep = false; break; }
                    // Embargo: immediately after test group
                    if (samples[i].start_time > g_end[tg] &&
                        samples[i].start_time <= g_end[tg] + embargo_dt) { keep = false; break; }
                }
                if (keep) sp.train_indices.push_back(i);
            }

            if (!sp.test_indices.empty() && !sp.train_indices.empty())
                splits.push_back(std::move(sp));
            return;
        }
        for (int i = start; i < n_groups; ++i) {
            combo.push_back(i); gen(i + 1); combo.pop_back();
        }
    };
    gen(0);
    return splits;
}

} 