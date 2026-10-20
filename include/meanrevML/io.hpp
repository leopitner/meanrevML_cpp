#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <ctime>
#include <cstdint>
#include "meanrevML/types.hpp"

namespace meanrevML {

inline std::vector<Bar> load_stooq_txt(const std::string& path) {
    std::vector<Bar> bars;
    std::ifstream f(path);
    if (!f.is_open()) return bars;

    std::string line;
    std::getline(f, line);  

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string ticker, per, date_s, time_s, open_s, high_s, low_s, close_s, vol_s, oi_s;

        std::getline(ss, ticker, ',');
        std::getline(ss, per, ',');
        std::getline(ss, date_s, ',');
        std::getline(ss, time_s, ',');
        std::getline(ss, open_s, ',');
        std::getline(ss, high_s, ',');
        std::getline(ss, low_s, ',');
        std::getline(ss, close_s, ',');
        std::getline(ss, vol_s, ',');
        std::getline(ss, oi_s, ',');

        if (date_s.size() != 8 || time_s.size() != 6) continue;

        std::tm tm{};
        tm.tm_year = std::stoi(date_s.substr(0, 4)) - 1900;
        tm.tm_mon  = std::stoi(date_s.substr(4, 2)) - 1;
        tm.tm_mday = std::stoi(date_s.substr(6, 2));
        tm.tm_hour = std::stoi(time_s.substr(0, 2));
        tm.tm_min  = std::stoi(time_s.substr(2, 2));
        tm.tm_sec  = std::stoi(time_s.substr(4, 2));

        #ifdef _WIN32
        int64_t ts = static_cast<int64_t>(_mkgmtime(&tm));
        #else
        int64_t ts = static_cast<int64_t>(timegm(&tm));
        #endif

        Bar b{};
        b.timestamp = ts;
        b.open   = std::stod(open_s);
        b.high   = std::stod(high_s);
        b.low    = std::stod(low_s);
        b.close  = std::stod(close_s);
        b.volume = std::stod(vol_s);
        bars.push_back(b);
    }
    return bars;
}

} 