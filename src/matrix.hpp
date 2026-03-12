#pragma once
#include "matrix.hpp"

template <typename T>
class linalg::Matrix {
public:
    using value_type = T;
    uint32_t row, column;
    std::vector<std::vector<T>> matrix;

    Matrix() = default;

    Matrix(const uint32_t row, const uint32_t column, const T& value = T()) :
        row(row), column(column), matrix(std::vector(row, std::vector(column, value))) {}

    Matrix(std::initializer_list<T> list, const uint32_t row, const uint32_t column) : row(row), column(column), matrix(row, std::vector<T>(column)) {
        auto itr = list.begin();

        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                assert(itr != list.end());
                matrix[i][j] = *itr;
                ++itr;
            }
        }
    }

    Matrix(std::initializer_list<std::initializer_list<T>> list) :
        row(list.size()), column(list.begin()->size()), matrix(row, std::vector<T>(column)) {
        uint32_t i = 0;

        for (const std::initializer_list<T>& r : list) {
            assert(r.size() == column);
            uint32_t j = 0;

            for (const T& value : r) {
                matrix[i][j++] = value;
            }
            i++;
        }
    }

    Matrix(const std::vector<T>& vec, const uint32_t row, const uint32_t column) : row(row), column(column), matrix(row, std::vector<T>(column)) {
        const uint32_t size = vec.size();

        for (uint32_t i = 0, k = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                assert(k != size);
                matrix[i][j] = vec[k++];
            }
        }
    }

    Matrix(const std::vector<std::vector<T>>& matrix) : row(matrix.size()), column(matrix[0].size()), matrix(matrix) {}

    Matrix(const Matrix&) = default;

    template <typename U>
    Matrix(const Matrix<U>& mat) : Matrix(mat.row, mat.column) {
        *this = mat;
    }

    Matrix& operator=(const Matrix&) = default;

    template <typename U>
    Matrix& operator=(const Matrix<U>& mat) {
        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                matrix[i][j] = static_cast<T>(mat[i, j]);
            }
        }
        return *this;
    }

    Matrix& operator=(Matrix&&) = default;

    template <typename U>
    Matrix& operator=(Matrix<U>&& mat) {
        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                matrix[i][j] = static_cast<T&&>(mat[i, j]);
            }
        }
        return *this;
    }

    Matrix augment(const Matrix& mat) const {
        assert(row == mat.row);
        Matrix res(row, column + mat.column);

        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                res[i, j] = matrix[i][j];
            }
            for (uint32_t j = 0; j < mat.column; j++) {
                res[i, column + j] = mat[i, j];
            }
        }
        return res;
    }

    T determinant() const {
        assert(row == column);
        Matrix mat = echelon_form();
        T res = 1;

        for (uint32_t i = 0; i < row; i++) {
            res *= mat[i, i];
        }
        return res;
    }


    Matrix echelon_form() const {
        Matrix res = *this;
        uint32_t pivot = 0;

        for (uint32_t cnt = 0; cnt < column && pivot < row; cnt++) {
            uint32_t pivot_row = pivot;

            while (pivot_row < row && res[pivot_row, cnt] == 0) {
                pivot_row++;
            }
            if (pivot_row == row) {
                continue;
            }
            if (pivot_row != pivot) {
                for (uint32_t j = 0; j < column; j++) {
                    std::swap(res[pivot_row, j], res[pivot, j]);
                }
            }
            for (uint32_t i = pivot + 1; i < row; i++) {
                T factor = res[i, cnt] / res[pivot, cnt];

                for (uint32_t j = cnt; j < column; j++) {
                    res[i, j] -= factor * res[pivot, j];
                }
            }
            pivot++;

            if (GLOBAL_FORMATTING.verbose) {
                *GLOBAL_FORMATTING.out << res;
            }
        }
        return res;
    }

    std::vector<T> gauss_elimination() {
        Matrix res = echelon_form();
        std::vector<T> solution(row);

        for (uint32_t i = 0; i < row; i++) {
            bool flag = true;

            for (uint32_t j = 0; j < column - 1; j++) {
                if (res[i, j] != 0) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                if (res[i, column - 1] == 0) {
                    *GLOBAL_FORMATTING.out << "Infinitely many solutions\n";
                } else {
                    *GLOBAL_FORMATTING.out << "No solution\n";
                }
                return {};
            }
        }
        for (int i = row - 1; i >= 0; i--) {
            T sum = res[i, column - 1];

            for (uint32_t j = i + 1; j < row; j++) {
                sum -= res[i, j] * solution[j];
            }
            solution[i] = sum / res[i, i];
        }
        return solution;
    }

    Matrix inverse() const {
        assert(determinant() != 0);
        Matrix res(row, column), aug_matrix = augment(make_identity(row));

        if (GLOBAL_FORMATTING.verbose) {
            *GLOBAL_FORMATTING.out << aug_matrix;
        }
        aug_matrix = aug_matrix.echelon_form();

        for (int i = row - 1; i >= 0; i--) {
            T pivot = aug_matrix[i, i];
            assert(pivot != 0);

            for (uint32_t j = 0; j < row * 2; j++) {
                aug_matrix[i, j] /= pivot;
            }
            for (uint32_t k = 0; k < i; k++) {
                T factor = aug_matrix[k, i];

                for (uint32_t j = 0; j < row * 2; j++) {
                    aug_matrix[k, j] -= factor * aug_matrix[i, j];
                }
            }
            if (GLOBAL_FORMATTING.verbose) {
                *GLOBAL_FORMATTING.out << aug_matrix;
            }
        }
        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < row; j++) {
                res[i][j] = aug_matrix[i, j + row];
            }
        }
        return res;
    }

    Matrix transpose() const {
        Matrix res(column, row);

        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                res[j, i] = matrix[i][j];
            }
        }
        return res;
    }

    Matrix operator-() const {
        Matrix res(row, column);

        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                res[i, j] = -matrix[i][j];
            }
        }
        return res;
    }

    template <typename U>
    Matrix& operator+=(const U& value) {
        return *this = *this + value;
    }

    template <typename U, typename R = decltype(std::declval<T>() + std::declval<detail::unwrap_matrix_t<U>>())>
    Matrix<R> operator+(const U& value) const {
        Matrix<R> res(row, column);

        if constexpr (detail::is_matrix_v<U>) {
            assert(row == value.row && column == value.column);
        }
        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                if constexpr (detail::is_matrix_v<U>) {
                    res[i, j] = matrix[i][j] + value[i, j];
                } else {
                    res[i, j] = matrix[i][j] + value;
                }
            }
        }
        return res;
    }

    template <typename U>
    Matrix& operator-=(const U& value) {
        return *this = *this - value;
    }

    template <typename U, typename R = decltype(std::declval<T>() - std::declval<detail::unwrap_matrix_t<U>>())>
    Matrix<R> operator-(const U& value) const {
        Matrix<R> res(row, column);

        if constexpr (detail::is_matrix_v<U>) {
            assert(row == value.row && column == value.column);
        }
        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                if constexpr (detail::is_matrix_v<U>) {
                    res[i, j] = matrix[i][j] - value[i, j];
                } else {
                    res[i, j] = matrix[i][j] - value;
                }
            }
        }
        return res;
    }

    template <typename U>
    Matrix& operator*=(const U& value) {
        return *this = *this * value;
    }

    template <typename U,
              typename R = decltype(std::declval<T>() * std::declval<detail::unwrap_matrix_t<U>>() +
                                    std::declval<T>() * std::declval<detail::unwrap_matrix_t<U>>())>
    Matrix<R> operator*(const U& value) const {
        if constexpr (detail::is_matrix_v<U>) {
            assert(column == value.row);
            Matrix<R> res(row, value.column);

            for (uint32_t i = 0; i < row; i++) {
                for (uint32_t j = 0; j < value.column; j++) {
                    for (uint32_t k = 0; k < column; k++) {
                        res[i, j] += matrix[i][k] * value[k, j];
                    }
                }
            }
            return res;
        } else {
            Matrix<R> res(row, column);

            for (uint32_t i = 0; i < row; i++) {
                for (uint32_t j = 0; j < column; j++) {
                    res[i, j] = matrix[i][j] * value;
                }
            }
            return res;
        }
    }

    template <typename U>
        requires(!detail::is_matrix_v<U>)
    Matrix& operator/=(const U& value) {
        return *this = *this / value;
    }

    template <typename U, typename R = decltype(std::declval<T>() / std::declval<detail::unwrap_matrix_t<U>>())>
        requires(!detail::is_matrix_v<U>)
    Matrix<R> operator/(const U& value) const {
        Matrix<R> res(row, column);

        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                res[i, j] = matrix[i][j] / value;
            }
        }
        return res;
    }

    bool operator==(const Matrix& other) const { return matrix == other.matrix; }

    std::vector<T>& operator[](const uint32_t i) { return matrix[i]; }

    const std::vector<T>& operator[](const uint32_t i) const { return matrix[i]; }

    T& operator[](const uint32_t i, const uint32_t j) { return matrix[i][j]; }

    const T& operator[](const uint32_t i, const uint32_t j) const { return matrix[i][j]; }

    static Matrix make_identity(const uint32_t row) {
        Matrix res(row, row);

        for (uint32_t i = 0; i < row; i++) {
            res[i, i] = 1;
        }
        return res;
    }


    static std::tuple<Matrix<algebra::Fraction>, Matrix<algebra::Variable>, Matrix<algebra::Fraction>>
    from_equations(const std::vector<algebra::Equation>& equations) {
        const uint32_t size = equations.size();
        Matrix<algebra::Fraction> B(size, 1);
        std::set<algebra::Variable> variables;

        for (uint32_t i = 0; i < size; i++) {
            for (const algebra::Variable& variable : equations[i].lhs.expression) {
                variables.insert(variable.basis());
            }
            B[i, 0] = static_cast<algebra::Fraction>(equations[i].rhs);
        }
        Matrix<algebra::Fraction> A(size, variables.size());

        for (uint32_t i = 0; i < size; i++) {
            for (const algebra::Variable& variable : equations[i].lhs.expression) {
                A[i, std::distance(variables.begin(), variables.find(variable.basis()))] = variable.coefficient;
            }
        }
        return {A, Matrix<algebra::Variable>(std::vector(variables.begin(), variables.end()), variables.size(), 1), B};
    }
};

namespace std {
    template <typename T>
    string to_string(const linalg::Matrix<T>& matrix) {
        uint32_t padding = 0;
        linalg::Matrix<std::string> format(matrix.row, matrix.column);

        for (uint32_t i = 0; i < matrix.row; i++) {
            for (uint32_t j = 0; j < matrix.column; j++) {
                format[i, j] = std::to_string(matrix[i, j]);
                padding = std::max(padding, static_cast<uint32_t>(format[i, j].size()) + 2);
            }
        }
        const std::string empty_space(matrix.column * padding + matrix.column - 1, ' ');
        const std::string border =
            std::string("+").append(padding / 2, '-').append(empty_space.substr(2 * (padding / 2))).append(padding / 2, '-').append("+");
        std::string res = border;

        for (uint32_t i = 0; i < format.row; i++) {
            res.append("\n|");

            for (uint32_t j = 0; j < format.column; j++) {
                const uint32_t remaining = padding - format[i, j].size();
                res.append(std::string(remaining / 2, ' ')).append(format[i, j]).append(std::string(remaining - remaining / 2, ' '));

                if (j < format.column - 1) {
                    res.push_back(' ');
                }
            }
            res.append("|\n");

            if (i < format.row - 1) {
                res.append("|").append(empty_space).push_back('|');
            }
        }
        return res.append(border).append(" ").append(std::to_string(matrix.row)).append("x").append(std::to_string(matrix.column)).append("\n");
    }
} // namespace std

template <typename T>
std::ostream& linalg::operator<<(std::ostream& out, const Matrix<T>& matrix) {
    return out << std::to_string(matrix);
}

template <typename T, typename U, typename R = decltype(std::declval<U>() + std::declval<T>())>
    requires(!linalg::detail::is_matrix_v<T>)
linalg::Matrix<R> operator+(const T& value, const linalg::Matrix<U>& matrix) {
    return matrix + value;
}

template <typename T, typename U, typename R = decltype(-std::declval<U>() + std::declval<T>())>
    requires(!linalg::detail::is_matrix_v<T>)
linalg::Matrix<R> operator-(const T& value, const linalg::Matrix<U>& matrix) {
    return -matrix + value;
}

template <typename T, typename U, typename R = decltype(std::declval<U>() * std::declval<T>())>
    requires(!linalg::detail::is_matrix_v<T>)
linalg::Matrix<R> operator*(const T& value, const linalg::Matrix<U>& matrix) {
    return matrix * value;
}


template <typename T, typename U, typename R = decltype(std::declval<T>() / std::declval<U>())>
    requires(!linalg::detail::is_matrix_v<T>)
linalg::Matrix<R> operator/(const T& value, const linalg::Matrix<U>& matrix) {
    linalg::Matrix<R> res(matrix.row, matrix.column);

    for (uint32_t i = 0; i < matrix.row; i++) {
        for (uint32_t j = 0; j < matrix.column; j++) {
            res[i, j] = value / matrix[i, j];
        }
    }
    return res;
}
