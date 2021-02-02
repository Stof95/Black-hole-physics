#include "kadath_bispheric.hpp"
#include "mpi.h"

using namespace Kadath ;
int main (int argc, char** argv) {

	int rc = MPI_Init(&argc, &argv) ;
	int rank = 0 ;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;

	if (argc <2) {
		cout <<"File missing..." << endl ;
		abort() ;
	}

//	int resolution ;
	double r1, r2, rshell1, rshell2, rext ;
	double t1, t2, omega, distance, rotaxis, om1, om2 ;

//	double mu = 0.14142 ;
	double mu = 0.182574186 ;
	int n = 1 ;
        int ell = 1 ;
        double freq = mu - 2*mu*mu*mu/(n*n) ;

	char* name_fich = argv[1] ;
	FILE* fich = fopen(name_fich, "r") ;

	Space_bispheric space (fich) ;
//	fread_be (&resolution, sizeof(int), 1, fich) ;
	fread_be (&distance, sizeof(double), 1, fich) ;
	fread_be (&r1, sizeof(double), 1, fich) ;
	fread_be (&r2, sizeof(double), 1, fich) ;
	fread_be (&rshell1, sizeof(double), 1, fich) ;
	fread_be (&rshell2, sizeof(double), 1, fich) ;
	fread_be (&rext, sizeof(double), 1, fich) ;
	fread_be (&t1, sizeof(double), 1, fich) ;
	fread_be (&t2, sizeof(double), 1, fich) ;
	fread_be (&omega, sizeof(double), 1, fich) ;
	fread_be (&rotaxis, sizeof(double), 1, fich) ;
	fread_be (&om1, sizeof(double), 1, fich) ;
	fread_be (&om2, sizeof(double), 1, fich) ;
	Scalar conf (space, fich) ;
	Scalar lapse (space, fich) ;
	Vector shift (space, fich) ;
	fclose(fich) ;


	if (rank==0) {
                cout << "Restart and increase t1, t2 with 0.1" << endl ;
        	cout << "distance: " << distance << endl ;
        	cout << "r1, r2: " << r1 << ", " << r2 << endl ;
		cout << "Mbh: " << 2*r1 << endl ;
        	cout << "rshell1: " << rshell1 << endl ;
       		cout << "rshell2: " << rshell2 << endl ;
        	cout << "omega: " << omega << endl ;
        	cout << "t1, t2: " << t1 << ", " << t2 << endl ;
        	cout << "rext: " << rext << endl ;
        	cout << "rotaxis: " << rotaxis << endl ;
        	cout << "om1, om2: " << om1 << ", " << om2 << endl ;
                cout << "mu: " << mu << endl ;
		cout << "alpha: " << 2*mu << endl ;
		cout << endl ;
        }

	Scalar field1 (space) ;
	field1 = 1 ;
	field1.std_base() ;

//	// initial value field1
//	Scalar field1 (space) ;
//	field1.annule_hard() ; // set zero everywhere
//	field1.set_domain(0) = 0 ; //  even at the origin if you want to use std_base
//	Index pp (space.get_domain(1)->get_nbr_points()) ; // index over nbr of points per domain
//      Val_domain rr (space.get_domain(1)->get_radius()) ; // all radii
//	Val_domain initVal (space.get_domain(1)->get_radius()) ; // initial field values specific domain
//	do {
//		double x1 = 2 * mu * mu * rr(pp) / n ;
//		double field_val = exp(-x1/2) * pow(x1,ell) * assoc_laguerre(n+1,2*ell+1, x1) ;
//		double field_val = 1.4 * exp(-2 * .08 * rr(pp)) ;
//		double field_val = exp(-.04 * rr(pp)) ;
//		initVal.set(pp) = field_val ;
//	} while (pp.inc()) ;
//
//	field1.set_domain(1) = initVal ;
//	field1.std_base() ;

	Scalar field2 (space) ;
	field2 = 1 ;
	field2.std_base() ;


	Base_tensor basis (shift.get_basis()) ;
	Metric_flat fmet (space, basis) ;
	System_of_eqs syst (space, 0) ;

	// Unknown
	syst.add_var ("ph1", field1) ;
	syst.add_var ("ph2", field2) ;

	// constants
	syst.add_cst ("mu", mu) ;
	syst.add_cst ("Psi", conf) ;
	syst.add_cst ("N", lapse) ;
	syst.add_cst ("bet", shift) ;
	syst.add_cst ("ome", omega) ;
	syst.add_cst ("rot", rotaxis) ;
	syst.add_cst ("frq", freq) ;

	// Metric :
	fmet.set_system (syst, "f") ;
	
	// Defenitions of momenta
//	syst.add_def ("Pi1 = (bet^i * D_i ph1 - frq * ph2) / (2 * N) ") ;
//	syst.add_def ("Pi2 = - (bet^i * D_i ph2 - frq * ph1) / ( 2 * N) ") ;
	syst.add_def ("LP1 = (2 * frq * bet^i * D_i ph2)/(2*N*N) + (frq * frq * ph1)/(2*N*N) - (bet^i * bet^j * D_i D_j ph1)/(2*N*N) - (bet^i * D_i bet^j * D_j ph1)/(2*N*N) + (bet^i * D_i ph1 * bet^j * D_j N)/(2*N*N*N) - (frq * ph2 * bet^i * D_i N)/(2*N*N*N)" ) ;
	syst.add_def ("LP2 = (2 * frq * bet^i * D_i ph1)/(2*N*N) + (frq * frq * ph2)/(2*N*N) - (bet^i * bet^j * D_i D_j ph2)/(2*N*N) - (bet^i * D_i bet^j * D_j ph2)/(2*N*N) + (bet^i * D_i ph2 * bet^j * D_j N)/(2*N*N*N) - (frq * ph1 * bet^i * D_i N)/(2*N*N*N)" ) ;


	// Inner BC sphere minus
	syst.add_eq_bc (0, INNER_BC, "ph1=1.") ;
	syst.add_eq_bc (0, INNER_BC, "ph2=1.") ;


	// Outer BC
	syst.add_eq_bc (0, OUTER_BC, "ph1=0.") ;
	syst.add_eq_bc (0, OUTER_BC, "ph2=0.") ;


	// Equations
	//syst.add_eq_inside (0, "bet^i * D_i Pi1 - N * D_i D^i ph1 / 2. - 2 * D_i N * D^i ph1 + N * mu * mu * ph1 / 2. = 0") ;
	//syst.add_eq_inside (0, "bet^i * D_i Pi2 - N * D_i D^i ph2 / 2. - 2 * D_i N * D^i ph2 + N * mu * mu * ph2 / 2. = 0") ;
	//syst.add_eq_inside (0, "bet^i * D_i Pi1 - N * D_i D^i ph1 / (2. * Psi^4) - N * D_i Psi * D^i ph1 / Psi^5 - 2 * D_i N * D^i ph1 + N * mu * mu * ph1 / 2. = 0") ;
	//syst.add_eq_inside (0, "bet^i * D_i Pi2 - N * D_i D^i ph2 / (2. * Psi^4) - N * D_i Psi * D^i ph2 / Psi^5 - 2 * D_i N * D^i ph2 + N * mu * mu * ph2 / 2. = 0") ;

	// Right equations
	//syst.add_eq_inside (0, "D_i N * D^i ph1 / (N * Psi^4) + D_i D^i ph1 / Psi^4 + 2 * D_i Psi * D^i ph1 / Psi^5 + (2 * frq * bet^i * D_i ph2 + frq * frq * ph1 - bet^i * bet^j * D_i D_j ph1 - bet^i * D_i bet^j * D_j ph1 + (bet^i * D_i ph1 - frq * ph2) * bet^j * D_j N / N) / N^2 - mu * mu * ph1 = 0") ;
        //syst.add_eq_inside (0, "D_i N * D^i ph2 / (N * Psi^4) + D_i D^i ph2 / Psi^4 + 2 * D_i Psi * D^i ph2 / Psi^5 + (- 2 * frq * bet^i * D_i ph1 + frq * frq * ph2 - bet^i * bet^j * D_i D_j ph2 - bet^i * D_i bet^j * D_j ph2 + (bet^i * D_i ph2 + frq * ph1) * bet^j * D_j N / N) / N^2 - mu * mu * ph2 = 0") ;

	// Trying again
	syst.add_eq_inside (0, "D_i N * D^i ph1 / (N * Psi^4) + D_i D^i ph1 / Psi^4 + 2 * D_i ph1 * D^i Psi / Psi^5 + 2 * LP1 - mu * mu * ph1 = 0") ;
	syst.add_eq_inside (0, "D_i N * D^i ph2 / (N * Psi^4) + D_i D^i ph2 / Psi^4 + 2 * D_i ph2 * D^i Psi / Psi^5 + 2 * LP2 - mu * mu * ph2 = 0") ;

//	syst.add_eq_inside (0, "D_i N * D^i ph1 / (N * Psi^4) + D_i D^i ph1 / Psi^4 + 2 * D_i ph1 * D^i Psi / Psi^5 + 2 - mu * mu * ph1 = 0") ;
//	syst.add_eq_inside (0, "D_i N * D^i ph2 / (N * Psi^4) + D_i D^i ph2 / Psi^4 + 2 * D_i ph2 * D^i Psi / Psi^5 + 2 - mu * mu * ph2 = 0") ;


	bool endloop = false ;
	int ite = 1 ;
	double conv ;

	while (!endloop) {
		endloop = syst.do_newton(1e-8, conv) ;
		if (rank==0) {
		    cout << "Newton iteration " << ite << " " << conv << endl ;
		}
		ite++ ;
	

		// Output
		if (rank==0) {
			char name[100] ;
			sprintf (name, "field_schw_25_%.0f_%.0f_%.0f_%.0f_%.0f_%.0f_%.1f_%.1f_%.2f_%.2f_%.3f.dat", distance, r1, r2, rshell1, rshell2, rext, t1, t2, om1, om2, mu) ;
			FILE* fich = fopen(name, "w") ;
			space.save(fich) ;
//			fwrite_be (&resolution, sizeof(int), 1, fich) ;
        	        fwrite_be (&distance, sizeof(double), 1, fich) ;
       		        fwrite_be (&r1, sizeof(double), 1, fich) ;
			fwrite_be (&r2, sizeof(double), 1, fich) ;
        	        fwrite_be (&rshell1, sizeof(double), 1, fich) ;
			fwrite_be (&rshell2, sizeof(double), 1, fich) ;
			fwrite_be (&rext, sizeof(double), 1, fich) ;
			fwrite_be (&t1, sizeof(double), 1, fich) ;
			fwrite_be (&t2, sizeof(double), 1, fich) ;
			fwrite_be (&omega, sizeof(double), 1, fich) ;
			fwrite_be (&rotaxis, sizeof(double), 1, fich) ;
			fwrite_be (&om1, sizeof(double), 1, fich) ;
			fwrite_be (&om2, sizeof(double), 1, fich) ;
			fwrite_be (&mu, sizeof(double), 1, fich) ;
			conf.save(fich) ;
			lapse.save(fich) ;
			shift.save(fich) ;
			field1.save(fich) ;
			field2.save(fich) ;
			fclose(fich) ;
		}
	}

	MPI_Finalize() ;

	return EXIT_SUCCESS ;
}
