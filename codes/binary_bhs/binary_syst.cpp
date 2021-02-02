#include "kadath_bispheric.hpp"
#include "mpi.h"

using namespace Kadath ;
int main(int argc, char** argv) {
	
	int rc = MPI_Init(&argc, &argv) ;
	int rank = 0 ;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;


	// resolution in bispherical domains
	int biResol = 17 ;

	// Parameters of the bispherical :
	double r = 1. ;
	double m = 2*r ;
	double distance = 170 ;
 	double r1 = r ;
 	double r2 = r ;
	double rshell1 = 15 ;
	double rshell2 = 15 ;
	double omega = 0.0045 ; // freq = sqrt(8(m1 + m2)/distance^3)
	double t1 = 0.1 ; // choice of lapse at horizons
	double t2 = 0.1 ;
	double rext = 340 ;
	double rotaxis = -0.01 ;
	double om1 = 0. ;
	double om2 = 0. ;

	// Resolution is now passed domain by domain.
	// The various resolutions are stored in an array of Dim_array
	Dim_array**  resol = new Dim_array*[8] ;
	for (int i=0 ; i<8 ; i++)
		resol[i] = new Dim_array(3) ;
	// Resolution in the two shells 
	resol[0]->set(0) = 33 ; resol[0]->set(1) = 33 ; resol[0]->set(2) = 27 ;
	resol[1]->set(0) = 17 ; resol[1]->set(1) = 17 ; resol[1]->set(2) = 16 ;
	// Resolutions in the bispheric domain (keep the same for all of them otherwise it will cause problems)
	for (int i=2 ; i<7 ; i++) {
		resol[i]->set(0) = biResol ; resol[i]->set(1) = biResol ; resol[i]->set(2) = biResol ;
	}
	// Last compactified domain
	resol[7]->set(0) = 13 ; resol[7]->set(1) = 13 ; resol[7]->set(2) = 12 ;

	if (rank==0) {
		cout << "resolution: " << resol << endl ;
		cout << "m: " << m << endl ;
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
	
	Space_bispheric space(polynomial_type, distance, r1, rshell1, r2, rshell2, rext, resol) ;

	// Number of domains (=3)
	int ndom = space.get_nbr_domains() ;

	// Position of bh origins
	double x1 = space.get_domain(0)->get_center()(1) ;
	double x2 = space.get_domain(1)->get_center()(1) ;
	
	// Tensorial basis used by the various tensors is cartesian
	Base_tensor basis (space, CARTESIAN_BASIS) ;
	
	// Define conformal factor Psi = 1
	Scalar conf (space) ;
	conf = 1. ;
	conf.std_base() ;

	// Define lapse function alpha = 1
	Scalar lapse (space) ;
	lapse = 1. ;
	lapse.std_base() ;

	// Define shift vector beta^i = (0, 0, 0)
	Vector shift (space, CON, basis) ;
	for (int i=1 ; i<=3 ; i++)
		shift.set(i).annule_hard() ;
	shift.std_base() ;


	{
        // Define flat metric f = diag(1,1,1)
        Metric_flat fmet (space, basis) ;

        // Set surface of bh's to constant variable (currently not used)
        Scalar one (space) ;
        one = 1 ;
        one.std_base() ;

        // Add initial data to sytem of equations
        System_of_eqs init(space, 0, ndom-1) ;

        // Add unknown parameters
        init.add_var ("Psi", conf) ;
        init.add_var ("N", lapse) ;

        // Add defined constants
        init.add_cst ("r1", r1) ;
        init.add_cst ("r2", r2) ;
        init.add_cst ("t1", t1) ;
        init.add_cst ("t2", t2) ;

        // Add flat metric
        fmet.set_system (init, "f") ;

        // Add inner boundary conditions on bh's
        init.add_eq_bc (0, INNER_BC, "N=t1") ;
        init.add_eq_bc (0, INNER_BC, "dn(Psi) + 0.5 * Psi / r1 = 0") ;
        init.add_eq_bc (1, INNER_BC, "N=t2") ;
        init.add_eq_bc (1, INNER_BC, "dn(Psi) + 0.5 * Psi / r2 = 0") ;

        // Add equations
        space.add_eq_no_nucleus (init, "D_i D^i N + 2 * D_i Psi * D^i N / Psi = 0", "N", "dn(N)") ;
        space.add_eq_no_nucleus (init, "D_i D^i Psi = 0", "Psi", "dn(Psi)") ;

        // Outer BC
        init.add_eq_bc (7, OUTER_BC, "N=1") ;
        init.add_eq_bc (7, OUTER_BC, "Psi=1") ;

        // Start iteration to solve lapse N and conformal factor Psi
        double conv ;
        bool endloop = false ;
        int ite = 1 ;
        if (rank==0)
            cout << "Phase 1 : N and Psi only" << endl ;
        while (!endloop) {
            endloop = init.do_newton(1e-6, conv) ;
            if (rank==0)
              cout << "Newton iteration " << ite << " " << conv << endl ;
            ite++ ;
            }
	}

  
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
        // Define again the metric and system of equations
        Metric_flat fmet (space, basis) ;
        System_of_eqs syst_fixed (space, 0, ndom-1) ;

        // Unknowns
        syst_fixed.add_var ("Psi", conf) ;
        syst_fixed.add_var ("N", lapse) ;
        syst_fixed.add_var ("bet", shift) ;

        // User defined constants
        syst_fixed.add_cst ("rad", rad) ;

        syst_fixed.add_cst ("s1", spin1) ;
        syst_fixed.add_cst ("s2", spin2) ;
        syst_fixed.add_cst ("om1", om1) ;
        syst_fixed.add_cst ("om2", om2) ;

        syst_fixed.add_cst ("J1", J1_center) ;
        syst_fixed.add_cst ("J2", J2_center) ;

        syst_fixed.add_cst ("n1", n1) ;
        syst_fixed.add_cst ("n2", n2) ;

        syst_fixed.add_cst ("t1", t1) ;
        syst_fixed.add_cst ("t2", t2) ;
        syst_fixed.add_cst ("ome", omega) ;
        syst_fixed.add_cst ("rot", rotaxis) ;
        syst_fixed.add_cst ("cor", corr_rotaxis) ;

        // Metric
        fmet.set_system (syst_fixed, "f") ;

        // Definition of the extrinsic curvature :
        syst_fixed.add_def ("A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) /2. / N") ;

        // Inner BC sphere minus
        syst_fixed.add_eq_bc (0, INNER_BC, "N=t1") ;
        syst_fixed.add_eq_bc (0, INNER_BC, "bet^i = t1 / Psi^2 * n1^i + ome * J1^i + om1 * s1^i + ome * rot * cor^i") ;
        syst_fixed.add_eq_bc (0, INNER_BC, "dn(Psi) + 0.5 * Psi / rad + Psi^3*A_ij * n1^i *n1^j/4. = 0") ;

        // Inner BC sphere plus
        syst_fixed.add_eq_bc (1, INNER_BC, "N=t2") ;
        syst_fixed.add_eq_bc (1, INNER_BC, "bet^i = t2 / Psi^2 * n2^i + ome * J2^i + om2 * s2^i + ome * rot * cor^i") ;
        syst_fixed.add_eq_bc (1, INNER_BC, "dn(Psi) + 0.5 * Psi / rad + Psi^3*A_ij * n2^i * n2^j /4.= 0") ;

        // Constraint equations
        // space.add_eq_no_nucleus (syst_fixed, "D_i D^i N + 14 * D_i Psi * D^i N / Psi + 42 * N * D_i Psi * D^i Psi / Psi^2 - 7 * N * A_ij *A^ij / 8. / Psi^8 = 0", "N", "dn(N)") ;
        // space.add_eq_no_nucleus (syst_fixed, "D_i D^i Psi + A_ij * A^ij / 8. / Psi^7 = 0", "Psi", "dn(Psi)") ;
        // space.add_eq_no_nucleus (syst_fixed, "D_j D^j bet^i + D^i D_j bet^j - 2*A^ij*D_j N - 2 * D_j D_k bet^k * f^ij / 3. = 0", "bet^i", "dn(bet^i)") ;

        //////////// or :
        space.add_eq_no_nucleus (syst_fixed, "D_i D^i N + 2 * D_i Psi * D^i N / Psi - N * Psi^4 * A_ij *A^ij= 0", "N", "dn(N)") ;
        space.add_eq_no_nucleus (syst_fixed, "D_i D^i Psi + Psi^5 *A_ij * A^ij / 8= 0", "Psi", "dn(Psi)") ;
        space.add_eq_no_nucleus (syst_fixed, "D_j D^j bet^i + D^i D_j bet^j /3. - 2*A^ij*(D_j N -6*N*D_j Psi / Psi) =0", "bet^i", "dn(bet^i)") ;
        ///////////////


        // Outer BC
        syst_fixed.add_eq_bc (7, OUTER_BC, "N=1") ;
        syst_fixed.add_eq_bc (7, OUTER_BC, "Psi=1") ;
        syst_fixed.add_eq_bc (7, OUTER_BC, "bet^i=0") ;

        // Start iteration to solve conf Psi, lapse N and shift beta with fixed omega
        bool endloop = false ;
        int ite = 1 ;
        double conv ;
        if (rank==0)
            cout << "Phase 2 : omega is fixed to " << omega << " -- Rotation axis also fixed" << endl ;
        while (!endloop) {
            endloop = syst_fixed.do_newton(1e-6, conv) ;
            if (rank==0) {
              cout << "Newton iteration " << ite << " " << conv << endl ;
		cout << "Omega: " << omega << endl ;
            }
            ite++ ;
        }
	}


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
            cout << "Phase 3 " << endl ;
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
		sprintf (name, "binsyst_25_%d_%.0f_%.0f_%.0f_%.0f_%.0f_%.0f_%.1f_%.1f_%.2f_%.2f.dat", biResol, distance, r1, r2, rshell1, rshell2, rext, t1, t2, om1, om2) ;
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

	MPI_Finalize() ;

	return EXIT_SUCCESS ;
}


