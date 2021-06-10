#pragma once

#include <string>

class ResultBase {
public:
    struct Time {
        double input;
        double hist;
        double eff;
        double construct;
        double merge;

        inline double total() const {
            return input + hist + eff + construct + merge;
        }
    };

protected:
    std::string m_algo;
    size_t m_nodes;
    size_t m_workers_per_node;
    size_t m_threads;
    std::string m_input;
    size_t m_size;
    size_t m_bytes_per_symbol;
    size_t m_alphabet;
    Time   m_time;
    size_t m_memory;
    size_t m_traffic;
    size_t m_traffic_asym;
    size_t m_bits_written;

public:
    std::string sqlplot() const;
    std::string readable() const;
};