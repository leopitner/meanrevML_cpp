#pragma once
#include <vector>
#include <cstdint>

namespace meanrevML {

    struct Bar {
        int64_t timestamp;
        double open, high, low, close, volume;
    };

    struct Sample {
        int bar_index;
        int end_bar_index;      
        int64_t start_time;
        int64_t end_time;
        int label;
        double hit_time;        
        double hit_magnitude;   
        std::vector<double> features;
    };

    struct CpcvSplit {
        std::vector<int> train_indices;
        std::vector<int> test_indices;
    };


    struct LabelConfig {
        int max_holding_bars = 20;
        double profit_mult = 2.0;
        double stop_mult = 1.0;
        int vol_lookback = 20;
    };
}