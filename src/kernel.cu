#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "automata.h"

static constexpr unsigned int BLOCK_SIZE = 10;

int LBM_Kernel() {

    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid(NX / block.x, NY / block.y, 1);


    collision << <grid, block >> > ();

    cudaDeviceSynchronize();

    streaming << <grid, block >> > ();

    cudaDeviceSynchronize();

    return 0;
}

