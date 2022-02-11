#include "render.h"
#include <glut/glut.h>
#include "grid.h"

#define __MULTI_ITER_PER_RENDER__

void initialize() {

    glClearColor(0.0, 0.0, 0.0, 1.0);

    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();

    glOrtho(-10, 800 + 10, -10, 600 + 10, -200.0, 200.0);

    glMatrixMode(GL_MODELVIEW);

}

/*
	function called by glut on rendnering.
*/
void renderScene() {
#ifdef __MULTI_ITER_PER_RENDER__
    for (int i = 0; i < 100; ++i)
#endif // __MULT_ITER_PER_RENDER__
            LBM_Kernel();
    render_plane();
    glutPostRedisplay();
}