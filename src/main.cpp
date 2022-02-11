#include "grid.h"
#include "glut/glut.h"
#include "render.h"

int main(int argc, char** argv)

{
	// init GLUT and create Window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(800, 600);
	glutCreateWindow("LGA");


	// register callback
	glutDisplayFunc(renderScene);

	// configure opengl
	initialize();

#ifdef __D1Q2__
	set_start_state_D1Q2();
#endif

#ifdef __D2Q4__
	set_start_state_D2Q4();
#endif

	// enter event loop
	glutMainLoop();


	return 0;
}