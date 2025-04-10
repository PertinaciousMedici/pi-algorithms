#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>
#include <string>
#include <thread>
#include <vector>

void print_out(const std::string& name, int mode, int elapsed);
double round_num(double value, int digits);
double PI_leibniz(int terms);
double PI_monte_carlo(int total);
double factorial(int number);
double PI_ramanujan(int terms);
double PI_gauss_legendre(int iterations);
double PI_chudnovsky(int terms);
double arc_tan(double x, int terms);
double PI_taylor(int terms);

std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<double> distribution(0.0, 1.0);

namespace Utility {
     template<typename Func>
     auto measure_time(const char* func_name, Func&& func) {
          const std::string as_str = func_name;

          const auto start = std::chrono::high_resolution_clock::now();
          print_out(as_str, 0, 0);
          const auto return_value = func();

          const auto end = std::chrono::high_resolution_clock::now();
          const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
          const auto as_num = duration.count();

          print_out(as_str, 1, as_num);
          const std::string output = std::format("\x1b[1;32m[OUTPUT]:\x1b[0m {}.", return_value);
          std::cout << output << '\n';

          return return_value;
     }
};

int main() {
     /*
      *   Ramanujan and Chudnovsky overflow as they require arbitrary precision.
      *   Even one term will overflow because of the explosive size of factorials.
      *   As it is not readily available in C++, I will not fix it.
      *   Honestly makes me wish this was Zig, a f128 would come in handy here...
      */
     const double PI_1 = Utility::measure_time("Leibniz", []() {
          return PI_leibniz(10000000);
     });

     const double PI_2 = Utility::measure_time("Monte Carlo", []() {
          return PI_monte_carlo(100000000);
     });

     const double PI_3 = Utility::measure_time("Legendre", []() {
          return PI_gauss_legendre(30);
     });

     const double PI_4 = Utility::measure_time("Taylor", []() {
          return PI_taylor(100000);
     });

     const std::vector results{PI_1, PI_2, PI_3, PI_4};
     std::cout << "\x1b[1;34m[END]:\x1b[34m Calculated PI a total of " << results.size() << " times.\n\x1b[0m";
     std::cout << std::flush;
}

void print_out(const std::string& name, const int mode, const int elapsed) {
     switch (mode) {
          case 0:
               std::cout << "\x1b[1;32m[STARTED]:\x1b[0m Processing \x1b[1;34m" << name << ".\x1b[0m\n";
               break;
          case 1:
               std::cout << "\x1b[1;32m[ENDED]:\x1b[0m Processed \x1b[1;34m" << name << "'s equation\x1b[0m in \x1b[33m" << elapsed << "ms.\x1b[0m\n";
               break;
          default:
               break;
     }
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
     const size_t thread_count = std::thread::hardware_concurrency();
     const size_t it_per_thread = total / thread_count;
     std::atomic<int> total_in{};

     auto worker = [&](int iterations) {
          int local_in{};

          for (size_t i = 0; i < iterations; i++) {
               const double x = distribution(rng);
               const double y = distribution(rng);
               if (x * x + y * y <= 1.0) {
                    local_in++;
               }
          }
          total_in += local_in;
     };

     std::vector<std::thread> threads;
     for (size_t i = 0; i < thread_count; i++) {
          threads.emplace_back(worker, it_per_thread);
     }

     for (auto& thread : threads) {
          thread.join();
     }

     const double PI = 4.0 * total_in / static_cast<double>(it_per_thread * thread_count);
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

double arc_tan(const double x, const int terms) {
     double sum{};
     double power = x;
     for (int i = 0; i < terms; i++) {
          const int n = 2 * i + 1;
          double term = power / n;
          if (i % 2 != 0) term = -term;
          sum += term;
          power *= x * x;
     }
     return sum;
}

double PI_taylor(const int terms) {
     const double arc_tan_1_5 = arc_tan(1.0 / 5.0, terms);
     const double arc_tan_1_239 = arc_tan(1.0 / 239.0, terms);
     const double PI = 16.0 * arc_tan_1_5 - 4.0 * arc_tan_1_239;
     return round_num(PI, 6);
}
