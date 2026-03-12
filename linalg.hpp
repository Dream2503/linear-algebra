#pragma once
#include <set>
#include "algebra/algebra.hpp"

namespace linalg {
    inline struct FormatSettings {
        bool verbose = false;
        std::ostream* out = &std::cout;
    } GLOBAL_FORMATTING;

    template <typename>
    class Matrix;
    template <typename T>
    std::ostream& operator<<(std::ostream&, const Matrix<T>&);
} // namespace linalg

#include "src/detail.hpp"
#include "src/matrix.hpp"
