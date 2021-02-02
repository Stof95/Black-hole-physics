/* Author: Christoffel Doorman
 * Date: 14-04-2020
 *
 * Compute complex scalar field in Schwarzschild spacetime
 *
 * Inputs:
 * .dat file containing metric fields in conformal flat spacetime
 *
 */

#include "kadath_spheric.hpp"
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

	int r2_res, r3_res ;
	double r0, r1, r2, r3 ;
	double t, ome ;

//	double mu = 0.1 ;
//	double mu = 0.11 ;
//	double mu = 0.12 ;
//	double mu = 0.13 ;
//	double mu = 0.14 ;
//	double mu = 0.15 ;
//	double mu = 0.16 ;
//	double mu = 0.17 ;
//	double mu = 0.18 ;
//	double mu = 0.19 ;
	double mu = 0.20 ;

//	double freq = 0.0994719732994934 ; // mu = 0.10
//
//	double freq = 0.1092953566202027 ; // mu = 0.11

//	double freq = 0.1190736173193454 ; // mu = 0.12
//	double freq = 0.1288049783744057 ; // mu = 0.13
//
//	double freq = 0.138482137126494  ; // mu = 0.14
//
//	double freq = 0.1480961731994996 ; // mu = 0.15

//	double freq = 0.157636041452291 ;  // mu = 0.16
//	double freq = 0.1670882997930869 ; // mu = 0.17
//	double freq = 0.1764379644998751 ; // mu = 0.18
//	double freq = 0.185670425087694 ;  // mu = 0.19
	double freq = 0.1944546186467369 ; // mu = 0.20

	char* name_fich = argv[1] ;
	FILE* fich = fopen(name_fich, "r") ;

	Space_spheric space (fich) ;
    fread_be (&r2_res, sizeof(int), 1, fich) ;
    fread_be (&r3_res, sizeof(int), 1, fich) ;
	fread_be (&r0, sizeof(double), 1, fich) ;
	fread_be (&r1, sizeof(double), 1, fich) ;
	fread_be (&r2, sizeof(double), 1, fich) ;
	fread_be (&r3, sizeof(double), 1, fich) ;
	fread_be (&t, sizeof(double), 1, fich) ;
	fread_be (&ome, sizeof(double), 1, fich) ;

	Scalar conf (space, fich) ;
	Scalar lapse (space, fich) ;
	Vector shift (space, fich) ;
	fclose(fich) ;

    if (rank==0) {
	cout << "cloud 2p" << endl ;
    	cout << "reading success" << endl ;
    	cout << "r2_res= " << r2_res << endl ;
        cout << "r3_res= " << r3_res << endl ;
    	cout << "r0 = " << r0 << endl ;
    	cout << "r1 = " << r1 << endl ;
    	cout << "r2 = " << r2 << endl ;
    	cout << "t  = " << t << endl ;
    	cout << "ome= " << ome << endl ;
	cout << "mu = " << mu << endl ;
	cout << "frq= " << freq << endl ;
    }

    // retrieve number of computational domains
	int ndom = space.get_nbr_domains() ;

    // Computation adm mass :
    Val_domain integ_adm (conf(ndom-1).der_r()) ;
    double adm = -space.get_domain(ndom-1)->integ(integ_adm, OUTER_BC)/2/M_PI ;
    if (rank==0) {cout << "adm mass: " << adm << endl ; }

    double a = adm * mu ;
    if (rank==0) {cout << "alpha: " << a << endl ; }

    if (rank==0) {cout << "freq: " << freq << endl ; }

	int r = r0 ;

    // initialize complex scalar fields
	Scalar field1 (space) ;
	field1.set_domain(0) = 0 ;

	Scalar field2 (space) ;
	field2.set_domain(0) = 0 ;

	for (int d=1; d<=(ndom-1) ; d++) {
		Val_domain rr = space.get_domain(d)->get_radius() ;
		rr.std_base_r_spher() ;
		Val_domain rh = rr + 1/(16*rr) + 0.5 ; // convert to harmonic coordinates
		field1.set_domain(d) = (-2*sqrt(2)*mu*mu*rh* exp(-mu*mu*rh)).mult_cos_phi().mult_sin_theta() ;
		field2.set_domain(d) = (-2*sqrt(2)*mu*mu*rh* exp(-mu*mu*rh)).mult_sin_phi().mult_sin_theta() ;
	}
	field1.set_domain(ndom-1) = 0 ;
	field1.set_domain(ndom-1) = 0 ;
	field1.std_base() ;
	field2.std_base() ;

	Scalar p1 (space) ;
	Scalar p2 (space) ;
	p1 = 0 ;
	p2 = 0 ;
	double A = 1.0 ;
	double chi = M_PI/4 ;
	for (int d=1 ; d<=(ndom-1) ; d++) {
		Val_domain rr = space.get_domain(d)->get_radius() ;
		rr.std_base_r_spher() ;
		Val_domain rh = rr + 1/(16*rr) + 0.5 ; // convert to harmonic coordinates

		p1.set_domain(d) = A*rh.mult_cos_phi().mult_sin_theta() ;
		p2.set_domain(d) = A*rh.mult_sin_phi().mult_sin_theta() ;

//		p1.set_domain(d) = -(2*sqrt(2)*mu*mu*rh* exp(-mu*mu*rh)).mult_cos_phi().mult_sin_theta() ;
//		p2.set_domain(d) = -(2*sqrt(2)*mu*mu*rh* exp(-mu*mu*rh)).mult_sin_phi().mult_sin_theta() ;

	}
	p1.std_base() ;
	p2.std_base() ;

	Base_tensor basis (shift.get_basis()) ;
	Metric_flat fmet (space, basis) ;
	System_of_eqs syst (space, 2, 3) ;

	// fields to solve
	syst.add_var ("ph1", field1) ;
	syst.add_var ("ph2", field2) ;

	// scalar field inner boundary conditions
	syst.add_cst ("p1", p1) ;
	syst.add_cst ("p2", p2) ;

	// constants
	syst.add_cst ("mu", mu) ;
	syst.add_cst ("Psi", conf) ;
	syst.add_cst ("N", lapse) ;
	syst.add_cst ("bet", shift) ;
	syst.add_cst ("frq", freq) ;

	// mtric
	fmet.set_system (syst, "f") ;

	// Defenitions of momenta
//	syst.add_def ("LP1 = (2 * frq * bet^i * D_i ph2)/(2*N*N) + (frq * frq * ph1)/(2*N*N) - (bet^i * bet^j * D_i D_j ph1)/(2*N*N) - (bet^i * D_i bet^j * D_j ph1)/(2*N*N) + (bet^i * D_i ph1 * bet^j * D_j N)/(2*N*N*N) - (frq * ph2 * bet^i * D_i N)/(2*N*N*N)" ) ;
//	syst.add_def ("LP2 = (2 * frq * bet^i * D_i ph1)/(2*N*N) + (frq * frq * ph2)/(2*N*N) - (bet^i * bet^j * D_i D_j ph2)/(2*N*N) - (bet^i * D_i bet^j * D_j ph2)/(2*N*N) + (bet^i * D_i ph2 * bet^j * D_j N)/(2*N*N*N) - (frq * ph1 * bet^i * D_i N)/(2*N*N*N)" ) ;

	// for n_0 = 0.0
	syst.add_def ("LP1 = (N * frq * bet^i * D_i ph2) + (N * frq * frq * ph1)/2 - (N * bet^i * bet^j * D_i D_j ph1)/2 - (N * bet^i * D_i bet^j * D_j ph1)/2 + (bet^i * D_i ph1 * bet^j * D_j N)/2 - (frq * ph2 * bet^i * D_i N)/2" ) ;
	syst.add_def ("LP2 = (N * frq * bet^i * D_i ph1) + (N * frq * frq * ph2)/2 - (N * bet^i * bet^j * D_i D_j ph2)/2 - (N * bet^i * D_i bet^j * D_j ph2)/2 + (bet^i * D_i ph2 * bet^j * D_j N)/2 - (frq * ph1 * bet^i * D_i N)/2" ) ;


	// Inner BC sphere minus
//	syst.add_eq_bc (1, INNER_BC, "ph1=1.") ;
//	syst.add_eq_bc (1, INNER_BC, "ph2=1.") ;
	space.add_inner_bc (syst, "ph1=p1") ;
	space.add_inner_bc (syst, "ph2=p2") ;


	// Outer BC
//	syst.add_eq_bc (1, OUTER_BC, "ph1=0.") ;
//	syst.add_eq_bc (1, OUTER_BC, "ph2=0.") ;
	space.add_outer_bc (syst, "ph1=0.") ;
	space.add_outer_bc (syst, "ph2=0.") ;

	// Equations with shift
//	space.add_eq (syst, "D_i N * D^i ph1 / (N * Psi^4) + D_i D^i ph1 / Psi^4 + 2 * D_i ph1 * D^i Psi / Psi^5 + 2 * LP1 - mu * mu * ph1 = 0", "ph1", "dn(ph1)") ;
//	space.add_eq (syst, "D_i N * D^i ph2 / (N * Psi^4) + D_i D^i ph2 / Psi^4 + 2 * D_i ph2 * D^i Psi / Psi^5 + 2 * LP2 - mu * mu * ph2 = 0", "ph2", "dn(ph2)") ;

	// for n_0 = 0.0
	space.add_eq (syst, "N^2 * D_i N * D^i ph1 / Psi^4 + N^3 * D_i D^i ph1 / Psi^4 + 2 * N^3 * D_i ph1 * D^i Psi / Psi^5 + 2 * LP1 - N^3 * mu * mu * ph1 = 0", "ph1", "dn(ph1)") ;
	space.add_eq (syst, "N^2 * D_i N * D^i ph2 / Psi^4 + N^3 * D_i D^i ph2 / Psi^4 + 2 * N^3 * D_i ph2 * D^i Psi / Psi^5 + 2 * LP2 - N^3 * mu * mu * ph2 = 0", "ph2", "dn(ph2)") ;


	// Newton Raphson iterations
	bool endloop = false ;
	int ite = 1 ;
	double conv ;

	while (!endloop) {
		endloop = syst.do_newton(1e-8, conv) ;
		if (rank==0) {
		    cout << "Newton iteration " << ite << " " << conv << endl ;
		}
		ite++ ;
	}

	// Output
	if (rank==0) {
		char name[100] ;
		sprintf (name, "schwField_2p_%d_%d_%.0f_%.1f_%.0f_%.0f_%.1f_%.2f.dat", r2_res, r3_res, r0, r1, r2, r3, t, mu) ;
		FILE* fich = fopen(name, "w") ;
		space.save(fich) ;
        fwrite_be (&r2_res, sizeof(int), 1, fich) ;
        fwrite_be (&r3_res, sizeof(int), 1, fich) ;
		fwrite_be (&r0, sizeof(double), 1, fich) ;
		fwrite_be (&r1, sizeof(double), 1, fich) ;
		fwrite_be (&r2, sizeof(double), 1, fich) ;
		fwrite_be (&r3, sizeof(double), 1, fich) ;
		fwrite_be (&t, sizeof(double), 1, fich) ;
		fwrite_be (&ome, sizeof(double), 1, fich) ;
		fwrite_be (&freq, sizeof(double), 1, fich) ;
		fwrite_be (&mu, sizeof(double), 1, fich) ;
		conf.save(fich) ;
		lapse.save(fich) ;
		shift.save(fich) ;
		field1.save(fich) ;
		field2.save(fich) ;
		fclose(fich) ;
	}

	MPI_Finalize() ;

	return EXIT_SUCCESS ;
}
