#include "kadath_bispheric.hpp"
#include "mpi.h"

using namespace Kadath ;
int main(int argc, char** argv) {
	
	int rc = MPI_Init(&argc, &argv) ;
	int rank = 0 ;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;

    // parameters of the space
    double distance, r1, r2, rshell1, rshell2, rext, t1, t2, omega, rotaxis, om1, om2 ;

    FILE* fich = fopen(argv[1], "r") ;
    Space_bispheric space (fich) ;

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
        	cout << "rshell1: " << rshell1 << endl ;
        	cout << "rshell2: " << rshell2 << endl ;
        	cout << "omega: " << omega << endl ;
        	cout << "t1, t2: " << t1 << ", " << t2 << endl ;
        	cout << "rext: " << rext << endl ;
        	cout << "rotaxis: " << rotaxis << endl ;
        	cout << "om1, om2: " << om1 << ", " << om2 << endl ;
		cout << endl ;
	}

	// Espace :
	int polynomial_type = CHEB_TYPE ;

	// Number of domains (=3)
	int ndom = space.get_nbr_domains() ;

	// Position of bh origins
	double x1 = space.get_domain(0)->get_center()(1) ;
	double x2 = space.get_domain(1)->get_center()(1) ;

	// Check if radius did go through well
	Index pp1 (space.get_domain(0)->get_nbr_points()) ;
	double rr1 = space.get_domain(0)->get_radius()(pp1) ;
    	Index pp2 (space.get_domain(1)->get_nbr_points()) ;
    	double rr2 = space.get_domain(1)->get_radius()(pp2) ;
	if (rank == 0 ) {
		if (r1 != rr1) { cout << "something wrong with r1 " << endl ; }
		if (r2 != rr2) { cout << "something wrong with r2 " << endl ; }
		else { cout << "r1 and r2 oke " << endl ; }
	}

    // Increase t1 and t2
    t1 += 0.1 ;
    t2 += 0.1 ;

    // Tensorial basis used by the various tensors is cartesian
    Base_tensor basis (space, CARTESIAN_BASIS) ;


	// Define cartesian vector in all domains except infinity (X,Y,Z)
	Vector cart(space, CON, basis) ;
	for (int i=1 ; i<=3 ; i++) {
		cart.set(i).set_domain(ndom-1) = 0 ;
		for (int d=0 ; d<ndom-1; d++)
			cart.set(i).set_domain(d) = space.get_domain(d)->get_cart(i) ;
		}
	cart.std_base() ;


	// Angular momentum bh1
	Vector spin1 (space, CON, basis) ;
	spin1.set(1) = cart(2) ;
	spin1.set(2) = - (cart(1)-x1) ;
	spin1.set(3) = 0. ;
	spin1.std_base() ;

	// Angular momentum bh2
	Vector spin2 (space, CON, basis) ;
	spin2.set(1) = cart(2) ;
	spin2.set(2) = - (cart(1)-x2) ;
	spin2.set(3) = 0. ;
	spin2.std_base() ;

	// Rotation of the center of BH1
	Vector J1_center (space, CON, basis) ;
	J1_center.set(1) = 0. ;
	J1_center.set(2) = - x1 ;
	J1_center.set(3) = 0. ;
	J1_center.std_base() ;

	// Rotation of center of BH2
	Vector J2_center (space, CON, basis) ;
	J2_center.set(1) = 0. ;
	J2_center.set(2) = - x2 ;
	J2_center.set(3) = 0. ;
	J2_center.std_base() ;

	// Account for the rotation axis position
	Vector corr_rotaxis (space, CON, basis) ;
	corr_rotaxis.set(1) = 0. ;
	corr_rotaxis.set(2) = 1 ;
	corr_rotaxis.set(3) = 0. ;
	corr_rotaxis.std_base() ;

	// Normal to sphere BH1 :
	Vector n1 (space, CON, basis) ;
	n1.set(1) = (cart(1)-x1)/r1 ;
	n1.set(2) = cart(2)/r1 ;
	n1.set(3) = cart(3)/r1  ;
	n1.std_base() ;

	// Normal to sphere BH2
	Vector n2 (space, CON, basis) ;
	n2.set(1) = (cart(1)-x2)/r2 ;
	n2.set(2) = cart(2)/r2 ;
	n2.set(3) = cart(3)/r2 ;
	n2.std_base() ;

	// Y-vector to correct for linear momentum in case of non-equal bh masses
	Vector ey (space, COV, basis) ;
	ey.set(1) = 0 ;
	ey.set(2) = 1 ;
	ey.set(3) = 0 ;
	ey.std_base() ;

	// Radial vector with respect to the bh's in inf domain
	Vector er (space, COV, basis) ;
	er = 0 ;
	for (int i=1 ; i<=3 ; i++)
	      er.set(i).set_domain(ndom-1) = space.get_domain(ndom-1)->get_cart_surr(i) ;
	er.std_base() ;

	// Radius used for inner BC bh's
	Scalar rad (space) ;
	rad = 0 ;
	for (int i=0 ; i<=1 ; i++)
	    rad.set_domain(i) = space.get_domain(i)->get_radius() ;
	rad.std_base() ;


	{
        // Define metric and add all parameters to system of equations
        Metric_flat fmet (space, basis) ;
        System_of_eqs syst (space, 0, ndom-1) ;

        // Unknowns
        syst.add_var ("Psi", conf) ;
        syst.add_var ("N", lapse) ;
        syst.add_var ("bet", shift) ;
        syst.add_var ("ome", omega) ;
        syst.add_var ("rot", rotaxis) ;

        // User defined constants
        syst.add_cst ("rad", rad) ;

        syst.add_cst ("s1", spin1) ;
        syst.add_cst ("s2", spin2) ;
        syst.add_cst ("om1", om1) ;
        syst.add_cst ("om2", om2) ;

        syst.add_cst ("J1", J1_center) ;
        syst.add_cst ("J2", J2_center) ;

        syst.add_cst ("n1", n1) ;
        syst.add_cst ("n2", n2) ;

        syst.add_cst ("t1", t1) ;
        syst.add_cst ("t2", t2) ;
        syst.add_cst ("cor", corr_rotaxis) ;
        syst.add_cst ("ey", ey) ;
        syst.add_cst ("er", er) ;

        // Add metric
        fmet.set_system (syst, "f") ;

        // Definition of the extrinsic curvature :
        syst.add_def ("A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) /2. / N") ;

        // Add integral equations :
        space.add_eq_int_inf (syst, "integ(dn(N) + 2 * dn(Psi))  = 0") ;
        space.add_eq_int_inf (syst, "integ(A^ij * er_i * ey_j) = 0") ;

        // Add inner BC sphere 1
        syst.add_eq_bc (0, INNER_BC, "N=t1") ;
        syst.add_eq_bc (0, INNER_BC, "bet^i = t1 / Psi^2 * n1^i + ome * J1^i + om1 * s1^i + ome * rot * cor^i") ;
        syst.add_eq_bc (0, INNER_BC, "dn(Psi) + 0.5 * Psi / rad + Psi^3*A_ij * n1^i *n1^j/4. = 0") ;

        // Inner BC sphere 2
        syst.add_eq_bc (1, INNER_BC, "N=t2") ;
        syst.add_eq_bc (1, INNER_BC, "bet^i = t2 / Psi^2 * n2^i + ome * J2^i + om2 * s2^i + ome * rot * cor^i") ;
        syst.add_eq_bc (1, INNER_BC, "dn(Psi) + 0.5 * Psi / rad + Psi^3*A_ij * n2^i * n2^j /4.= 0") ;

        // Constraint equations
        //space.add_eq_no_nucleus (syst, "D_i D^i N + 14 * D_i Psi * D^i N / Psi + 42 * N * D_i Psi * D^i Psi / Psi^2 - 7 * N * A_ij *A^ij / 8. / Psi^8 + 7 * N * D_i D^i Psi = 0", "N", "dn(N)") ;
        //space.add_eq_no_nucleus (syst, "D_i D^i Psi + A_ij * A^ij / 8. / Psi^7= 0", "Psi", "dn(Psi)") ;
        //space.add_eq_no_nucleus (syst, "D_j D^j bet^i + D^i D_j bet^j - 2*A^ij*D_j N - 2 * D_j D_k bet^k * f^ij / 3. = 0", "bet^i", "dn(bet^i)") ;

	//////////// or :
        space.add_eq_no_nucleus (syst, "D_i D^i N + 2 * D_i Psi * D^i N / Psi - N * Psi^4 * A_ij *A^ij= 0", "N", "dn(N)") ;
        space.add_eq_no_nucleus (syst, "D_i D^i Psi + Psi^5 *A_ij * A^ij / 8= 0", "Psi", "dn(Psi)") ;
        space.add_eq_no_nucleus (syst, "D_j D^j bet^i + D^i D_j bet^j /3. - 2*A^ij*(D_j N -6*N*D_j Psi / Psi) =0", "bet^i", "dn(bet^i)") ;
        ///////////////

        // Outer BC
        syst.add_eq_bc (7, OUTER_BC, "N=1") ;
        syst.add_eq_bc (7, OUTER_BC, "Psi=1") ;
        syst.add_eq_bc (7, OUTER_BC, "bet^i=0") ;

        // Start final iteration to solve for everything
        bool endloop = false ;
        int ite = 1 ;
        double conv ;
        if (rank==0)
            cout << "Start of iterations... " << endl ;
        while (!endloop) {
            endloop = syst.do_newton(3e-8, conv) ;
            if (rank==0) {
                cout << "Newton iteration " << ite << " " << conv << endl ;
                cout << "Omega   = " << omega << " ; axis = " << rotaxis << endl ;
            }
            ite++ ;
			// Output
	if (rank==0) {
		char name[100] ;
		sprintf (name, "binsyst_adj_25_%.0f_%.0f_%.0f_%.0f_%.0f_%.0f_%.1f_%.1f_%.2f_%.2f.dat", distance, r1, r2, rshell1, rshell2, rext, t1, t2, om1, om2) ;
		FILE* fich = fopen(name, "w") ;
		space.save(fich) ;
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
		conf.save(fich) ;
		lapse.save(fich) ;
		shift.save(fich) ;
		fclose(fich) ;
	}
	}
	}


	// Output
	if (rank==0) {
		char name[100] ;
		sprintf (name, "binsyst_adj_25_%.0f_%.0f_%.0f_%.0f_%.0f_%.0f_%.1f_%.1f_%.2f_%.2f.dat", distance, r1, r2, rshell1, rshell2, rext, t1, t2, om1, om2) ;
		FILE* fich = fopen(name, "w") ;
		space.save(fich) ;
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
		conf.save(fich) ;
		lapse.save(fich) ;
		shift.save(fich) ;
		fclose(fich) ;
	}

	MPI_Finalize() ;

	return EXIT_SUCCESS ;
}


