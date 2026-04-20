#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <numbers>
#include <random>
#include <string>
#include <thread>
#include <vector>

// I could have used BigInt or Arbitary Precision, but libraries on Windows are cancer.
#define DESIRED_PRECISION 12

double PI_leibniz(int terms);
double PI_monte_carlo(int total);
double PI_ramanujan(int terms);
double PI_gauss_legendre(int iterations);
double PI_chudnovsky(int terms);
double PI_taylor(int terms);

static int global_counter = 0;

namespace Helpers {
     void print_out(const std::string& name, const int mode, const int elapsed) {
          switch (mode) {
               case 0:
                    std::cout << "\x1b[1;32m[STARTED]:\x1b[0m Processing \x1b[1;34m" << name << ".\x1b[0m\n";
                    break;
               case 1:
                    std::cout << "\x1b[1;32m[ENDED]:\x1b[0m Processed \x1b[1;34m" << name << "'s method\x1b[0m in \x1b[33m" << elapsed << " microseconds.\x1b[0m\n";
                    break;
               default:
                    break;
          }
     }

     double round_num(const double value, const int digits) {
          const double factor = std::pow(10.0, digits);
          return std::round(value * factor) / factor;
     }
};

namespace Utility {
     template<typename Func>
     auto measure_time(const char* func_name, Func&& func) {
          const std::string as_str = func_name;

          const auto start = std::chrono::high_resolution_clock::now();
          Helpers::print_out(as_str, 0, 0);
          const auto return_value = func();

          const auto end = std::chrono::high_resolution_clock::now();
          const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
          const auto as_num = duration.count();
          global_counter = global_counter + as_num;

          Helpers::print_out(as_str, 1, as_num);
          const std::string output = std::format("\x1b[1;32m[OUTPUT]:\x1b[0m {}.", return_value);
          std::cout << output << '\n';

          return return_value;
     }
};


namespace Maths {
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

     double factorial(const int number) {
          double result = 1;
          for (int i = 1; i <= number; i++) {
               result *= i;
          }
          return result;
     }
};

int main() {
     const double PI_1 = Utility::measure_time("Leibniz", []() {
          return PI_leibniz(10e8);
     });

     const double PI_2 = Utility::measure_time("Monte Carlo", []() {
          return PI_monte_carlo(10e8);
     });

     const double PI_3 = Utility::measure_time("Ramanujan", []() {
          return PI_ramanujan(2);
     });

     const double PI_4 = Utility::measure_time("Legendre", []() {
          return PI_gauss_legendre(4);
     });

     const double PI_5 = Utility::measure_time("Chudnovsky", []() {
          return PI_chudnovsky(1);
     });

     const double PI_6 = Utility::measure_time("Taylor", []() {
          return PI_taylor(24);
     });


     const std::vector results{PI_1, PI_2, PI_3, PI_4, PI_5, PI_6};
     std::cout << "\x1b[1;34m[END]:\x1b[34m Calculated PI a total of " << results.size() << " times. " << global_counter << " microseconds elapsed.\n\x1b[0m";
     std::cout << std::flush;
}

double PI_leibniz(const int terms) {
     double PI = 0;
     double sign = 1.0;
     for (int i = 0; i < terms; i++) {
          PI += sign / (2.0 * i + 1.0);
          sign = -sign;
     }
     PI *= 4.0;
     return Helpers::round_num(PI, DESIRED_PRECISION);
}

double PI_monte_carlo(const int total) {
     const size_t thread_count = std::max<size_t>(1, std::thread::hardware_concurrency());
     const size_t it_per_thread = total / thread_count;
     std::atomic<size_t> total_in{};

     auto worker = [&](int iterations) {
          thread_local std::mt19937 local_rng(std::random_device{}());
          std::uniform_real_distribution<double> local_dist(0.0, 1.0);
          int local_in{};

          for (size_t i = 0; i < iterations; i++) {
               const double x = local_dist(local_rng);
               const double y = local_dist(local_rng);
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
     return Helpers::round_num(PI, DESIRED_PRECISION);
}

double PI_ramanujan(const int terms) {
     double sum = 0;
     for (int i = 0; i < terms; i++) {
          const double fact_i = Maths::factorial(i);
          const double numerator = Maths::factorial(4 * i) * (1103.0 + 26390.0 * i);
          double denominator = std::pow(fact_i, 4);
          denominator *= std::pow(396.0, 4 * i);
          sum += numerator / denominator;
     }
     const double PI = 9801.0 / (2 * std::numbers::sqrt2 * sum);
     return Helpers::round_num(PI, DESIRED_PRECISION);
}

double PI_gauss_legendre(const int iterations) {
     double a = 1.0;
     double b = 1.0 / std::numbers::sqrt2;
     double t = 0.25;
     double p = 1.0;
     for (int i = 0; i < iterations; i++) {
          const double a_next = (a + b) / 2;
          b = std::sqrt(a * b);
          const double diff = a - a_next;
          t -= p * (diff * diff);
          a = a_next;
          p *= 2;
     }
     const double PI = std::pow(a + b, 2) / (4 * t);
     return Helpers::round_num(PI, DESIRED_PRECISION);
}

double PI_chudnovsky(const int terms) {
     const double C = std::pow(640320.0, 1.5) / 12.0;
     double sum = 0;

     for (int i = 0; i < terms; i++) {
          const double numerator = Maths::factorial(6 * i)
          * (13591409.0 + 545140134.0 * i)
          * (i % 2 == 0 ? 1.0 : -1.0);
          const double denominator = Maths::factorial(3 * i)
          * std::pow(Maths::factorial(i), 3)
          * std::pow(640320.0, 3 * i);
          sum += numerator / denominator;
     }
     const double PI = C / sum;
     return Helpers::round_num(PI, DESIRED_PRECISION);
}


double PI_taylor(const int terms) {
     const double arc_tan_1_5 = Maths::arc_tan(1.0 / 5.0, terms);
     const double arc_tan_1_239 = Maths::arc_tan(1.0 / 239.0, terms);
     const double PI = 16.0 * arc_tan_1_5 - 4.0 * arc_tan_1_239;
     return Helpers::round_num(PI, DESIRED_PRECISION);
}
