#pragma once
#include <vector>
#include <cmath>
#include "types.hpp"

namespace meanrevML {

// Triple-barrier labels for mean-reversion signals.
// Enters short when z > +threshold, long when z < -threshold.
// Label 1 = profit barrier hit first (reversion), 0 = stop or adverse time-barrier.
inline std::vector<Sample> triple_barrier_labels(
    const std::vector<Bar>& bars,
    const std::vector<double>& zscores,
    const std::vector<double>& vols,
    const LabelConfig& cfg,
    double z_entry_threshold)
{
    std::vector<Sample> out;
    int n = bars.size();
    if (n <= cfg.vol_lookback + cfg.max_holding_bars) return out;

    for (int i = cfg.vol_lookback; i < n - 1; ++i) {
        double z = zscores[i];
        if (std::abs(z) < z_entry_threshold) continue;
        double sigma = vols[i];
        if (sigma <= 0.0) continue;

        bool is_short = (z > 0);
        double entry = bars[i].close;
        double profit_b = is_short ? entry * (1.0 - cfg.profit_mult * sigma)
                                   : entry * (1.0 + cfg.profit_mult * sigma);
        double stop_b   = is_short ? entry * (1.0 + cfg.stop_mult * sigma)
                                   : entry * (1.0 - cfg.stop_mult * sigma);

        int end_idx = std::min(i + cfg.max_holding_bars, n - 1);
        int label = -1, hit = end_idx;

        for (int j = i + 1; j <= end_idx; ++j) {
            if (is_short) {
                if (bars[j].low  <= profit_b) { label = 1; hit = j; break; }
                if (bars[j].high >= stop_b)   { label = 0; hit = j; break; }
            } else {
                if (bars[j].high >= profit_b) { label = 1; hit = j; break; }
                if (bars[j].low  <= stop_b)   { label = 0; hit = j; break; }
            }
        }
        if (label < 0) {  // time barrier — label by realized direction
            double realized = (bars[end_idx].close - entry) / entry;
            label = (is_short ? realized < 0 : realized > 0) ? 1 : 0;
        }

        Sample s;
        s.bar_index  = i;
        s.start_time = bars[i].timestamp;
        s.end_time   = bars[hit].timestamp;
        s.label      = label;
        out.push_back(std::move(s));
    }
    return out;
}

} 