#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec.h"
#include "grid.h"
#include "glut/glut.h"

__global__ void collision() {

    static constexpr SCALAR tau = 1.f;
    static constexpr SCALAR dt = 1.f;


    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (IS_BLOCK(plane[y][x]))
        return;

#ifdef __D1Q2__
    static constexpr SCALAR w = 0.5f;
#endif

#ifdef __D2Q4__
    static constexpr SCALAR w = 0.25f;
#endif

    auto& cell = plane[y][x];

    // out and eq values are computed based on selected velocity model

#ifdef __D1Q2__

    // same type as vector values
		SCALAR c = cell.in.l + cell.in.r;

		cell.eq.l = w * c;
		cell.eq.r = w * c;

		cell.out.r = cell.in.l + (dt / tau) * (cell.eq.l - cell.in.l);
		cell.out.l = cell.in.r + (dt / tau) * (cell.eq.r - cell.in.r);
#endif

#ifdef __D2Q4__

    // same type as vector values
		SCALAR c = cell.in.l + cell.in.r + cell.in.d + cell.in.u;

		cell.eq.l = cell.eq.r = cell.eq.d = cell.eq.u = w * c;

		cell.out.r = cell.in.l + (dt / tau) * (cell.eq.l - cell.in.l);
		cell.out.l = cell.in.r + (dt / tau) * (cell.eq.r - cell.in.r);
		cell.out.u = cell.in.d + (dt / tau) * (cell.eq.d - cell.in.d);
		cell.out.d = cell.in.u + (dt / tau) * (cell.eq.u - cell.in.u);
#endif

}



/*
	 Parallel function that computes streaming step.
*/
__global__ void streaming() {

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    auto& cell = plane[y][x];

    if (IS_BLOCK(cell))
        return;

    // in values are computed based on selected velocity model

#ifdef __D1Q2__

    unsigned int left = x == 0 ? x : x - 1;
	unsigned int right = x + 1 >= NX ? x : x + 1;


	cell.in.l = IS_BLOCK(plane[y][left]) ? cell.in.r : plane[y][left].out.r;
	cell.in.r = IS_BLOCK(plane[y][right]) ? cell.in.l :  plane[y][right].out.l;

#endif // __D1Q2__

#ifdef __D2Q4__

    unsigned int left = x == 0 ? x : x - 1;
	unsigned int right = x + 1 >= NX ? x - 1 : x + 1;
	unsigned int up = y == 0 ? y : y - 1;
	unsigned int down = y + 1 >= NY ? y - 1 : y + 1;

	cell.in.l = IS_BLOCK(plane[y][left]) ? cell.in.r : plane[y][left].out.r;
	cell.in.r = IS_BLOCK(plane[y][right]) ? cell.in.l :	plane[y][right].out.l;
	cell.in.u = IS_BLOCK(plane[up][x]) ? cell.in.d : plane[up][x].out.d;
	cell.in.d = IS_BLOCK(plane[down][x]) ? cell.in.u : plane[down][x].out.u;

#endif // __D1Q2__


}

/*
	 render function
*/
void render_plane() {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBegin(GL_POINTS);

    for (unsigned int i = 0; i < NY; ++i) {
        for (unsigned int j = 0; j < NX; ++j) {
            if (IS_BLOCK(plane[i][j])) {
                glColor3f(1.0, 0, 0);
                glVertex3f(static_cast<GLfloat>(j), static_cast<GLfloat>(i), 0.0);
            }
            else {
                SCALAR c = plane[i][j].in.l + plane[i][j].in.r;
                if (c > 0) {
                    glColor3f(c, c, c);
                    glVertex3f(static_cast<GLfloat>(j), static_cast<GLfloat>(i), 0.0);
                }
            }
        }
    }

    glEnd();
    glutSwapBuffers();
}


void set_start_state_D1Q2()
{
    static constexpr int blockXstart = NX / 2 - 5;
    static constexpr int blockXend = NX / 2 + 5;

    static constexpr int holeYstart = NY / 2 - 50;
    static constexpr int holeYend = NY / 2 + 50;

    for (unsigned int i = 0; i < NY; ++i) {
        for (unsigned int j = 0; j < NX; ++j) {
            auto& cell = plane[i][j];
            if (blockXstart <= j && j <= blockXend && !(holeYstart <= i && i <= holeYend)) {
                MAKE_BLOCK(cell);
            }
            else if (j < NX / 2u) {
                cell.in.l = 1;
                cell.in.r = 0;
            }
            else {
                cell.in.l = 0;
                cell.in.r = 0;
            }
        }
    }
}

void set_start_state_D2Q4()
{

#ifdef __D2Q4__
    static constexpr unsigned int blockXstart = NX / 2 - 5;
	static constexpr unsigned int blockXend = NX / 2 + 5;

	static constexpr unsigned int holeYstart = NY / 2 - 50;
	static constexpr unsigned int holeYend = NY / 2 + 50;

	for (unsigned int i = 0; i < NY; ++i) {
		for (unsigned int j = 0; j < NX; ++j) {
			auto& cell = plane[i][j];
			if (blockXstart <= j && j <= blockXend && !(holeYstart <= i && i <= holeYend)) {
				MAKE_BLOCK(cell);
			}

			else if (j < NX / 2) {
				cell.in.l = 1;
				cell.in.r = 1;
			}
			else {
				cell.in.u = 0;
				cell.in.d = 0;
				cell.in.l = 0;
				cell.in.r = 0;
			}
		}
	}

#endif

}
