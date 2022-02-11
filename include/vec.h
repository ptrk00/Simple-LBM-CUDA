#pragma once

/*
	the Vec definition is based on macro defined in header file
*/
#include "velocity_def.h"

using SCALAR = float;


#ifdef __D1Q2__

struct Vec {

	Vec(SCALAR l, SCALAR r) : l(l), r(r) {}
	Vec() : l(0.), r(0.) {}

	// left, right, up, down
	SCALAR l, r;
};

#endif

#ifdef __D2Q4__

struct Vec {

	Vec(SCALAR l, SCALAR r, SCALAR u, SCALAR d) : l(l), r(r), u(u), d(d) {}
	Vec() : l(0.), r(0.), u(0.), d(0.) {}

	// left, right, up, down
	SCALAR l, r, u, d;
};

#endif