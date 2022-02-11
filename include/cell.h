#pragma once

#include "vec.h"

#define BLOCK -1

#define MAKE_BLOCK(v) v.in.r = BLOCK;

#define IS_BLOCK(v) v.in.r == BLOCK


/*
	cell is defined as a pair of two 2 or 4-args (based on velocity model) vectors. The 'in'
	vector defines input state, 'out' the output state of each cell.
*/
struct Cell {

    Cell() = default;
    explicit Cell(Vec in) : in(in) {}

    Vec in;
    Vec out;
    Vec eq;
};

