#include <cstdio>
#include <cstdlib>

#define _USE_MATH_DEFINES
#include <cmath>

#include <mkl.h>
#include "MEF.h"

void getF(const int N, const double *x, double *f){
	for (int i = 0; i < N; i++){
		f[i] = 3.0*sin(2.0*x[i]);
	}
}

double exactSolution(double x, double a, double b, double ya, double yb){
	double A = (cos(b) * (ya + sin(2 * a)) - cos(a)*(yb + sin(2 * b))) / sin(a - b);
	double B = (-sin(b) *(ya + sin(2 * a)) + sin(a)*(yb + sin(2 * b))) / sin(a - b);
	return A*sin(x) + B*cos(x) - sin(2.0*x);
}


int main(int argc, char eargv0){
	MEF *mef = new MEF();
	double a = 0.;
	double b = M_PI_2;
	double ya = 0.0;
	double yb = 1.0;
	int N = 20;
	double h = (b - a) / (double)(N-1);

	double *x = (double *)mkl_malloc(N*sizeof(double), 64);
	double *y = (double *)mkl_malloc(N*sizeof(double), 64);
	double *f = (double *)mkl_malloc(N*sizeof(double), 64);
	for (int i = 0; i < N; i++) x[i] = a + (double)i*h;
	getF(N, x, f);


	double B = 0.0;
	double C = 1.0;

	int ret = mef->solveDirichlet(N, h, B, C, f, ya, yb, y);
	// alculo de los valores exactos 
	double *yexact = (double*)mkl_malloc(N*sizeof(double), 64);
	printf("a:%g, b:%g,ya:%g,yb:%g", a, b, ya, yb);
	std::getchar();
	for (int i = 0; i < N; i++){
		yexact[i] = exactSolution(x[i], a, b, ya, yb);
	}
	
	double error = 0.0;
	int n = 0;
	for (int i = 0; i < N; i++){
		// se excluyen los valores muy prOximos a cero 
		if (fabs(yexact[i]) > 1.0e-4){
			error += fabs((yexact[i] - y[i]) / yexact[i]);
			n++;
		}
	}
	error /= n;
	printf("\nError relativo promedio(%%): %g\n", 100.0*error);
	mkl_free(x);
	mkl_free(y);
	mkl_free(yexact);
	delete mef;
	std::getchar();
	return 0;
}