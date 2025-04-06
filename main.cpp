#pragma unroll
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>
#include <string>
#include <vector>

double round_num(double value, int digits);
double PI_leibniz(int terms);
double PI_monte_carlo(int total);
double factorial(int number);
double PI_ramanujan(int terms);
double PI_gauss_legendre(int iterations);
double PI_chudnovsky(int terms);

std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<double> distribution(0.0, 1.0);

int main() {
     /*
      *   Ramanujan and Chudnovsky overflow as they require arbitrary precision.
      *   Even one term will overflow because of the explosive size of factorials.
      *   As it is not readily available in C++, I will not fix it.
      *   Honestly makes me wish this was Zig, a f128 would come in handy here...
      */
     const double PI_1 = PI_leibniz(10000000);
     const double PI_2 = PI_monte_carlo(10000000);
     // const double PI_3 = PI_ramanujan(1);
     const double PI_4 = PI_gauss_legendre(20);
     // const double PI_5 = PI_chudnovsky(1);

     std::vector<double> PI_interpretations{PI_1, PI_2, PI_4};

     int counter{1};

     for (double value_of_pi : PI_interpretations) {
          std::string output = std::format("\x1b[1;32m[Interpretation {}]:\x1b[0m {}.", counter, value_of_pi);
          std::cout << output << '\n';
          counter++;
     }

     std::cout << std::flush;
}

double round_num(const double value, const int digits) {
     const double factor = std::pow(10.0, digits);
     return std::round(value * factor) / factor;
}

/*
 *   Each iteration does modulo, assignment, multiplication, addition and accumulation (+=);
 *   These operations are quite simple and relatively optimised, with modulo being arguably the least efficient.
 *   Therefore, complexity is mainly driven by the amount of iterations and scales linearly.
 *   T(n) = n * C, wherein n is the number of terms assigned to the function upon calling.
 *   Consequently, T(n) is O(n), whereas space is O(1).
 */
double PI_leibniz(const int terms) {
     double PI = 0;
     for (int i = 0; i < terms; i++) {
          double sign = 1.0;
          sign = i % 2 == 0 ? 1.0 : -1.0;
          PI += sign / static_cast<double>(2 * i + 1);
     }
     PI *= 4.0;
     return round_num(PI, 6);
}

/*
 * Monte Carlo is probabilistic by nature and therefore has to call functions from the random library.
 * Each iteration does two randomisation calls, multiplication, addition, compare and increment (conditional).
 * This adds up to T(n) = n * Sum(2 * C_rng + 2 * C_mul + C_add + C_cmp + C_inc) + C_mul + C_div;
 * Wherein n is the total number of iterations (passed by the function call).
 * Time complexity scales similarly, in linear fashion and is mainly driven by the n number of iterations.
 * Therefore T(n) = O(n), whereas space is O(1).
 */
double PI_monte_carlo(const int total) {
     int inside = 0;
     for (int i = 0; i < total; i++) {
          const double x = distribution(rng);
          const double y = distribution(rng);
          if (x * x + y * y <= 1.0) {
               inside++;
          }
     }
     const double PI = 4.0 * inside / total;
     return round_num(PI, 6);
}

/*
 *   Factorial is iterative and avoids the overhead associated with recursive function calls.
 *   Each call multiplies all integers from 1 upto n, resulting in n multiplications.
 *   Therefore, time complexity is linear. T(n) = n * C_mul => O(n).
 *   Space complexity, on the other hand, is O(1).
 */
double factorial(const int number) {
     double result = 1;
     for (int i = 1; i <= number; i++) {
          result *= i;
     }
     return result;
}

/*
 *   Ramanujan's formula employs large factorials and powers.
 *   Therefore, due to floating point arithmetics and variable-size constraints,
 *   The accuracy is limited and the function is prone to overflow for n > 1
 *   Assuming factorial to be O(n) as stated prior, and used four times per iteration,
 *   The complexity explodes rapidly beyond linear scale and goes up to O(n^2 * max(i))
 *   Performance and precision will degrade unless arbitrary-precision is employed.
 *   T(n) = Sum(i = 0 to n) of [O(i) + O(4i) + O(1103 + 26390i) + O(i)^4 + pow(396, 4i)]
 *   Space, however, remains O(1).
 */
double PI_ramanujan(const int terms) {
     double sum = 0;
     for (int i = 0; i < terms; i++) {
          const double fact_i = factorial(i);
          const double numerator = factorial(4 * i) * factorial(1103 + 26390 * i);
          double denominator = std::pow(fact_i, 4);
          denominator *= std::pow(396.0, 4 * i);
          sum += numerator / denominator;
     }
     const double PI = 9801.0 / (2 * std::numbers::sqrt2 * sum);
     return round_num(PI, 6);
}

/*
 *   Each iteration performs square root, addition, division, subtraction and multiplication.
 *   All operations are constant time (O(1)) for double-precision.
 *   Total time complexity is linear with respect to the number of iterations: T(n) = n * C.
 *   Therefore, T(n) = O(n), and space complexity is also O(1), since no additional data is stored.
 *   Despite the simplicity of per-step cost, convergence is fast and only a few iterations are required.
 */
double PI_gauss_legendre(const int iterations) {
     double a = 1.0;
     double b = 1.0 / std::numbers::sqrt2;
     double t = 0.25;
     double p = 1.0;
     for (int i = 0; i < iterations; i++) {
          const double a_next = (a + b) / 2;
          b = std::sqrt(a * b);
          t -= p * std::pow(a - a_next, 2);
          a = a_next;
          p *= 2;
     }
     const double PI = std::pow(a + b, 2) / (4 * t);
     return round_num(PI, 6);
}

/*
 *   Fastest of the bunch though it requires high precision arithmetic.
 *   Iterations involve several factorials, large exponentiation, and high constants.
 *   Each iteration does as follows:
 *   - factorial(6i), factorial(3i), factorial(i)^3 → ~O(i)
 *   - pow(640320, 3i) → O(log(3i)) ≈ O(i)
 *   - so each iteration is ~O(i), total T(n) is roughly O(n^2)
 *   Time complexity: O(n^2), Space complexity: O(1)
 */
double PI_chudnovsky(const int terms) {
     constexpr double C = 12 * std::pow(640320.0, 1.5);
     double sum = 0;

     for (int i = 0; i < terms; i++) {
          const double numerator = factorial(6 * i) * (13591409 + 545140134 * i);
          const double denominator = factorial(3 * i) * std::pow(factorial(i), 3) * std::pow(640320.0, 3 * i);
          sum += numerator / denominator;
     }
     const double PI = C / sum;
     return round_num(PI, 6);
}
