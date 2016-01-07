/*


*/

class MEF{

public:
	MEF();
	~MEF();
	int solveDirichlet(	const int N, const double h, 
				const double B, const double C, const double *f, 
				const double ya, const double yb,
				double *y);

private:
	void ensambleDirichlet(	const int N, const double h, 
							const double B, const double C, const double *f,
							const double ya, const double yb,
							double *LHS, double *RHS);

	void Matrix(const int N, const double h, double *U, double *V, double *W);
};