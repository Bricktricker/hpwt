#pragma once

#include <chrono>
#include <fstream>
#include <vector>

#include <sys/stat.h>

namespace util {

inline size_t file_size(const std::string& filename) {
    struct stat buf;
    [[maybe_unused]] const auto error = stat(filename.c_str(), &buf);
#ifdef DEBUG
    if(error != 0) {
        throw std::runtime_error("error getting file size");
    }
#endif
    return buf.st_size;
}

inline double time() {
    using namespace std::chrono;
    return double(duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count()) / 1000.0;
}

}

// stream output for bit vectors
inline std::ostream& operator<<(
    std::ostream& os, const std::vector<bool>& bv) {

    for(bool b : bv) {
        os << (b ? '1' : '0');
    }

    return os;
}