/*
Método de los Elementos Finitos, condiciones de Dirichlet
ULPGC, EII, MNC
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <mkl.h>

#include "MEF.h"

MEF::MEF(){}

MEF::~MEF(){}

int MEF::solveDirichlet(	const int N, const double h,
				const double B, const double C, const double *f, 
				const double ya, const double yb,
				double *y){

	double *RHS = (double *)mkl_malloc(N*sizeof(double), 64);
	double *LHS = (double *)mkl_malloc(3 * N*sizeof(double), 64);
	int *ipiv = (int *)mkl_malloc(N*sizeof(int),32);

	ensambleDirichlet(N, h, B, C, f, ya,yb, LHS, RHS);
	
	int info = LAPACKE_dgtsv(LAPACK_ROW_MAJOR, N, 1, &(LHS[2*N]), &(LHS[N]), &(LHS[1]), RHS, 1);

	for (int i = 0; i < N; i++) y[i] = RHS[i];

	mkl_free(LHS);
	mkl_free(RHS);
	mkl_free(ipiv);

	return info;
}

// Ensambla para el problema de Dirichlet
void MEF::ensambleDirichlet(	const int N, const double h, 
							const double B, const double C, const double *f,
							const double ya, const double yb,
							double *LHS, double *RHS){

	double *U = (double*)mkl_malloc(3*N*sizeof(double),64);
	double *V = (double*)mkl_malloc(3*N*sizeof(double),64);
	double *W = (double*)mkl_malloc(3*N*sizeof(double),64);

	Matrix(N, h, U, V, W); 

	// LHS = U+B*V+C*W

	for (int i = 0; i < 3*N; i++) LHS[i] = U[i] + B*V[i] + C*W[i];
	LHS[N] = LHS[2*N-1]   = 1.0;
	LHS[1] = LHS[3*N - 2] = 0.0;
	
	for (int i = 1; i < N - 1; i++) RHS[i] = h*(f[i-1]+4.0*f[i]+f[i+1])/6.0;
	RHS[0]     = ya;
	RHS[N - 1] = yb;

	/*
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < N; j++){
			printf("%g ", LHS[i*N + j]);
		}
		printf("\n");
	}
	printf("\n");
	*/

	mkl_free(U);
	mkl_free(V);
	mkl_free(W);
}

// tres matrices en banda
void MEF::Matrix(const int N, const double h, double *U, double *V, double *W){

	for (int i = 0; i < N; i++){
		U[i] = 1.0 / h;
		U[N + i] = -2.0 / h;
		U[2 * N + i] = 1.0 / h;
	}
	U[0] = 0.0;
	U[N] = U[2*N-1] = -1.0 / h;
	U[3*N - 1] = 0.0;

	for (int i = 0; i < N; i++){
		V[i] = 1.0 / 2.0;
		V[N + i] = 0.0;
		V[2 * N + i] = - 1.0 / 2.0;
	}
	V[0] = 0.0;
	V[3*N - 1] = 0.0;

	for (int i = 0; i < N; i++){
		W[i] = h / 6.0;
		W[N + i] = 4.0*h/6.0;
		W[2 * N + i] = h / 6.0;
	}
	W[0] = 0.0;
	W[N] = W[2 * N - 1] = h / 3.0;
	W[3 * N - 1] = 0.0;
}