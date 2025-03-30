#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <algorithm>
#include <numeric>

// Prevent compiler from optimizing out computations
volatile int sink_result;
constexpr size_t MEM_SIZE = 50'000'000; // ~50MB

// Test 1: Integer arithmetic and branching
void test_integer_math() {
    const int iterations = 10'000'000;
    int result = 0;
    
    for(int i = 0; i < iterations; i++) {
        if(i % 3 == 0) {
            result += i * 2;
        } else {
            result -= i / 2;
        }
    }
    
    sink_result = result;
}

// Test 2: Floating-point operations
void test_float_math() {
    const int iterations = 5'000'000;
    float result = 0.0f;
    
    for(int i = 1; i < iterations; i++) {
        result += std::sqrt(static_cast<float>(i)) 
                * std::sin(static_cast<float>(i))
                / std::log(static_cast<float>(i));
    }
    
    sink_result = static_cast<int>(result);
}

// Test 3: Memory-intensive operations
void test_memory_access() {
    char* buffer = new char[MEM_SIZE];
    const int block_size = 1024;
    
    // Memory write pattern
    for(size_t i = 0; i < MEM_SIZE; i += block_size) {
        size_t bytes_to_write = (i + block_size <= MEM_SIZE) ? block_size : MEM_SIZE - i;
        std::memset(buffer + i, (i % 256), bytes_to_write);
    }
    
    // Memory read pattern
    int sum = 0;
    for(size_t i = 0; i < MEM_SIZE; i++) {
        sum += buffer[i];
    }
    
    delete[] buffer;
    sink_result = sum;
}

// Test 4: Mixed workload (typical game loop)
void test_mixed_workload() {
    const int entities = 10'000;
    float positions[entities];
    int health[entities];
    
    // Initialize
    for(int i = 0; i < entities; i++) {
        positions[i] = static_cast<float>(i);
        health[i] = 100;
    }
    
    // Simulate game loop
    for(int frame = 0; frame < 1000; frame++) {
        for(int i = 0; i < entities; i++) {
            // Physics-like calculations
            positions[i] += std::cos(frame * 0.01f) * 0.5f;
            
            // Game logic
            if(frame % 60 == 0) {
                health[i] -= (i % 2 == 0) ? 1 : 0;
            }
            
            // Collision-like check
            if(positions[i] > 100.0f) {
                positions[i] = 0.0f;
            }
        }
    }
    
    sink_result = static_cast<int>(positions[0]);
}

void test_simd() {
    const int iterations = 10'000'000;
    __m256 a = _mm256_set1_ps(1.0f);
    __m256 b = _mm256_set1_ps(0.9999f);
    __m256 result = _mm256_setzero_ps();

    for (int i = 0; i < iterations; i++) {
        result = _mm256_add_ps(result, _mm256_mul_ps(a, b));
        a = _mm256_mul_ps(a, b);
    }

    alignas(32) float temp[8];
    _mm256_store_ps(temp, result);
    sink_result = static_cast<int>(temp[0]);
}

// Test 6: Function Call Overhead
void __attribute__((noinline)) simple_function(int& counter) {
    ++counter;
}

class Base {
public:
    virtual void inc(int& counter) = 0;
};

class Derived : public Base {
public:
    virtual void inc(int& counter) override {
        ++counter;
    }
};

void test_function_calls() {
    const int iterations = 10'000'000;
    int counter = 0;

    // Non-virtual calls
    for (int i = 0; i < iterations; i++) {
        simple_function(counter);
    }

    // Virtual calls
    Derived d;
    Base* b = &d;
    for (int i = 0; i < iterations; i++) {
        b->inc(counter);
    }

    sink_result = counter;
}

// Test 7: Memory Subsystem Stress (Pointer Chasing)
void test_memory_stress() {
    const size_t num_nodes = 10'000'000;
    struct Node { Node* next; };
    Node* nodes = new Node[num_nodes];

    // Create random permutation
    std::vector<size_t> indices(num_nodes);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{42});

    for (size_t i = 0; i < num_nodes; ++i) {
        nodes[i].next = &nodes[indices[i]];
    }

    // Traverse the list
    Node* current = &nodes[0];
    int sum = 0;
    const int traversals = 1000;

    for (int i = 0; i < traversals; ++i) {
        current = current->next;
        sum += reinterpret_cast<uintptr_t>(current) % 256;
    }

    delete[] nodes;
    sink_result = sum;
}

// Test 8: Threading/Synchronization
void test_threading() {
    const int num_threads = 4;
    const int increments_per_thread = 250'000;
    int shared_counter = 0;
    std::mutex mtx;

    auto worker = [&]() {
        for (int i = 0; i < increments_per_thread; ++i) {
            std::lock_guard<std::mutex> lock(mtx);
            shared_counter++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    sink_result = shared_counter;
}

// Test 9: Atomic Operations
void test_atomic() {
    const int num_threads = 4;
    const int increments_per_thread = 250'000;
    std::atomic<int> atomic_counter(0);

    auto worker = [&]() {
        for (int i = 0; i < increments_per_thread; ++i) {
            atomic_counter++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    sink_result = atomic_counter.load();
}

// Test 10: Branch Prediction Stress
void test_branch_prediction() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 99);
    const int iterations = 25'000'000;
    int result = 0;

    for (int i = 0; i < iterations; ++i) {
        if (dist(rng) < 50) {  // 50% branch miss prediction
            result += i;
        } else {
            result -= i;
        }
    }

    sink_result = result;
}

template<typename Func>
void run_test(const char* name, Func test) {
    using namespace std::chrono;
    
    // Warm-up run
    test();
    
    // Timed runs
    const int runs = 5;
    long long total_duration = 0;
    
    for(int i = 0; i < runs; i++) {
        auto start = high_resolution_clock::now();
        test();
        auto end = high_resolution_clock::now();
        
        auto duration = duration_cast<milliseconds>(end - start).count();
        total_duration += duration;
        
        std::cout << "Run " << (i+1) << ": " << duration << "ms\n";
    }
    
    std::cout << "[" << name << "] "
              << "Average: " << (total_duration / runs) << "ms\n\n";
}

int main() {
    std::cout << "Starting benchmark...\n\n";
    
    run_test("Integer Math", test_integer_math);
    run_test("Float Math", test_float_math);
    run_test("Memory Access", test_memory_access);
    run_test("Mixed Workload", test_mixed_workload);
    run_test("SIMD Vectorization", test_simd);
    run_test("Function Call Overhead", test_function_calls);
    run_test("Memory Subsystem Stress", test_memory_stress);
    run_test("Threading/Mutex", test_threading);
    run_test("Atomic Operations", test_atomic);
    run_test("Branch Prediction", test_branch_prediction);

    std::cout << "Benchmark completed.\n";
    return 0;
}
