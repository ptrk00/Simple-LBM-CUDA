#pragma once

#include <vector>
#include "cell.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <array>

/*
   plane size
 */
inline constexpr unsigned int NX = 800;
inline constexpr unsigned int NY = 600;

extern __device__ __managed__ Cell plane[NY][NX];

__global__ void streaming();
__global__ void collision();

int LBM_Kernel();
void render_plane();

void set_start_state_D2Q4();
void set_start_state_D1Q2();
