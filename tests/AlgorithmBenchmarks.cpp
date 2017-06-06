//==--- tests/AlgorithmBenchmarks.cpp ---------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file  AlgorithmBenchmarks.cpp
/// \brief This file defines benchmarks for algorithmic functionality.
// 
//==------------------------------------------------------------------------==//

#include <Voxel/Algorithm/Algorithm.hpp>
#include <benchmark/benchmark.h>
#include <algorithm>

// Benchmarks the performance of Voxel's forEach implementation on containers.
static void BmForEach(benchmark::State& state, std::size_t vSize) {
  std::vector<int> v(vSize, 0);

  while (state.KeepRunning()) {
    int result = 0;
    Voxx::forEach(v, [&result] (auto& element) { 
      benchmark::DoNotOptimize(result += element);
    });
  }
}

// Reference benchmark for a loop which the compiler can unroll.
static void BmForEachLoop(benchmark::State& state, std::size_t vSize) {
  std::vector<int> v(vSize, 0);

  while (state.KeepRunning()) {
    int result = 0;
    for (std::size_t i = 0; i < v.size(); ++i)
      benchmark::DoNotOptimize(result += v[i]);
  }
}

// Benchmark for a manually unrolled loop.
static void BmForEachUnroll(benchmark::State& state, std::size_t vSize) {
  std::vector<int> v(vSize, 0);

  while (state.KeepRunning()) {
    int result    = 0;
    std::size_t i = 0;
    if (v.size() >= 4) {
      for (std::size_t i = 0; i < v.size() - 4; i += 4) {
        benchmark::DoNotOptimize(result += v[i]);
        benchmark::DoNotOptimize(result += v[i + 1]);
        benchmark::DoNotOptimize(result += v[i + 2]);
        benchmark::DoNotOptimize(result += v[i + 3]);
      }
    }
    for (; i < v.size(); ++i)
      benchmark::DoNotOptimize(result += v[i]);
  }
}

// Reference benchmark for std::for_each.
static void BmForEachStd(benchmark::State& state, std::size_t vSize) {
  std::vector<int> v(vSize, 0);

  while (state.KeepRunning()) {
    int result = 0;
    std::for_each(v.begin(), v.end(), [&result] (auto& e) {
      benchmark::DoNotOptimize(result += e);
    });
  }
}

//==------------------------------------------------------------------------==//

int main(int argc, char** argv) {
  using namespace benchmark;
  RegisterBenchmark("ForEachStdTiny"     , BmForEachStd   , 3);
  RegisterBenchmark("ForEachVoxelTiny"   , BmForEach      , 3);
  RegisterBenchmark("ForEachLoopTiny"    , BmForEachLoop  , 3);
  RegisterBenchmark("ForEachUnrollTiny"  , BmForEachUnroll, 3);
  RegisterBenchmark("ForEachStdMedium"   , BmForEachStd   , 1003);
  RegisterBenchmark("ForEachVoxelMedium" , BmForEach      , 1003);
  RegisterBenchmark("ForEachLoopMedium"  , BmForEachLoop  , 1003);
  RegisterBenchmark("ForEachUnrollMedium", BmForEachUnroll, 1003);
  RegisterBenchmark("ForEachStdLarge"    , BmForEachStd   , 1000003);
  RegisterBenchmark("ForEachVoxelLarge"  , BmForEach      , 1000003);
  RegisterBenchmark("ForEachLoopLarge"   , BmForEachLoop  , 1000003);
  RegisterBenchmark("ForEachUnrollLarge" , BmForEachUnroll, 1000003);

  Initialize(&argc, argv);
  RunSpecifiedBenchmarks();
}

