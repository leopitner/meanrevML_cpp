#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace meanrevML {

class ShallowRF {
public:
    struct Config {
        int n_trees = 100;
        int max_depth = 5;
        int min_samples_leaf = 20;
        int max_features = 0;   // 0 → sqrt(d)
        int seed = 42;
    };
    explicit ShallowRF(Config cfg = {}) : cfg_(cfg) {}

    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
        if (X.empty()) return;
        int n = (int)X.size(), d = (int)X[0].size();
        if (cfg_.max_features <= 0)
            cfg_.max_features = std::max(1, (int)std::sqrt((double)d));

        std::mt19937 rng(cfg_.seed);
        trees_.clear();
        trees_.resize(cfg_.n_trees);

        std::vector<int> all(n); std::iota(all.begin(), all.end(), 0);
        for (int t = 0; t < cfg_.n_trees; ++t) {
            std::vector<int> boot(n);
            for (int i = 0; i < n; ++i) boot[i] = all[rng() % n];
            trees_[t].push_back(Node{});
            build(trees_[t], 0, X, y, boot, 0, d, rng);
        }
    }

    std::vector<double> predict_proba(const std::vector<std::vector<double>>& X) const {
        std::vector<double> out(X.size(), 0.0);
        for (size_t i = 0; i < X.size(); ++i) {
            double s = 0;
            for (const auto& tr : trees_) {
                int nd = 0;
                while (!tr[nd].is_leaf)
                    nd = (X[i][tr[nd].feature] <= tr[nd].threshold) ? tr[nd].left : tr[nd].right;
                s += tr[nd].proba;
            }
            out[i] = s / (double)trees_.size();
        }
        return out;
    }

private:
    struct Node {
        bool is_leaf = true;
        int feature = -1;
        double threshold = 0;
        int left = -1, right = -1;
        double proba = 0;
    };
    Config cfg_;
    std::vector<std::vector<Node>> trees_;

    void build(std::vector<Node>& tree, int id,
               const std::vector<std::vector<double>>& X, const std::vector<int>& y,
               const std::vector<int>& idx, int depth, int d, std::mt19937& rng)
    {
        int pos = 0;
        for (int i : idx) pos += y[i];
        tree[id].proba = (double)pos / idx.size();

        if (depth >= cfg_.max_depth || (int)idx.size() < 2*cfg_.min_samples_leaf ||
            pos == 0 || pos == (int)idx.size()) return;

        std::vector<int> feats(d); std::iota(feats.begin(), feats.end(), 0);
        std::shuffle(feats.begin(), feats.end(), rng);
        int nt = std::min(cfg_.max_features, d);

        int bf = -1; double bt = 0, best = 1e18;
        for (int fi = 0; fi < nt; ++fi) {
            int f = feats[fi];
            std::vector<double> v; v.reserve(idx.size());
            for (int i : idx) v.push_back(X[i][f]);
            std::sort(v.begin(), v.end());

            for (double pct : {0.10, 0.25, 0.50, 0.75, 0.90}) {
                int pos = std::min((int)(pct * v.size()), (int)v.size() - 1);
                double thr = v[pos];

                int lc = 0, lp = 0, rc = 0, rp = 0;
                for (int i : idx) {
                    if (X[i][f] <= thr) { lc++; lp += y[i]; }
                    else                { rc++; rp += y[i]; }
                }
                if (lc < cfg_.min_samples_leaf || rc < cfg_.min_samples_leaf) continue;
                double pl = (double)lp/lc, pr = (double)rp/rc;
                double sc = lc*pl*(1-pl) + rc*pr*(1-pr);
                if (sc < best) { best = sc; bf = f; bt = thr; }
            }
        }
        if (bf < 0) return;

        tree[id].is_leaf = false;
        tree[id].feature = bf;
        tree[id].threshold = bt;

        std::vector<int> li, ri;
        for (int i : idx)
            (X[i][bf] <= bt ? li : ri).push_back(i);

        int lid = (int)tree.size(); tree.push_back(Node{});
        int rid = (int)tree.size(); tree.push_back(Node{});
        tree[id].left = lid; tree[id].right = rid;
        build(tree, lid, X, y, li, depth+1, d, rng);
        build(tree, rid, X, y, ri, depth+1, d, rng);
    }
};

} 