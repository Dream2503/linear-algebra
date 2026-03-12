#include <fstream>
#include "linalg.hpp"

using namespace algebra;
using namespace linalg;

int main() {
    std::ofstream out("output.txt");
    GLOBAL_FORMATTING = {true, &out};
    Variable x1("x1"), x2("x2"), x3("x3"), x4("x4");

    out << Matrix<Fraction>{{0.5, 2, 7}, {3, -1, 9}}.transpose() << std::endl;
    out << 2 * Matrix<Fraction>{{1, -5}, {3, 7}} << std::endl;
    out << Matrix<Fraction>{{2, 3}, {-1, 7}} + Matrix<Fraction>{{7, -8}, {2, 0}} << std::endl;
    out << Matrix<Fraction>{{1, 2}, {3, 4}, {5, 6}} * Matrix<Fraction>{{2, 5}, {6, 8}} << std::endl;
    out << Matrix<Fraction>{{1, 1}, {2, 2}} * Matrix<Fraction>{{-1, 1}, {1, -1}} << std::endl;

    {
        const auto [A, X, B] = Matrix<Variable>::from_equations({
            -x1 + x2 + 2 * x3 == 2,
            3 * x1 - x2 + x3 == 6,
            -x1 + 3 * x2 + 4 * x3 == 4,
        });
        out << A << X << B << std::endl;
        out << A.augment(B) << std::endl;
        out << A.augment(B).echelon_form() << std::endl;
    }
    {
        const auto [A, X, B] = Matrix<Variable>::from_equations({
            3 * x1 + 2 * x2 + 2 * x3 - 5 * x4 == 8,
            0.6 * x1 + 1.5 * x2 + 1.5 * x3 - 5.4 * x4 == 2.7,
            1.2 * x1 - 0.3 * x2 - 0.3 * x3 + 2.4 * x4 == 2.1,
        });
        out << A << X << B << std::endl;
        out << A.augment(B) << std::endl;
        out << A.augment(B).echelon_form() << std::endl;
        A.augment(B).gauss_elimination();
    }


    out << Matrix<Fraction>({{-1, 1, 2}, {3, -1, 1}, {-1, 3, 4}}).inverse() << std::endl;
    return 0;
}
