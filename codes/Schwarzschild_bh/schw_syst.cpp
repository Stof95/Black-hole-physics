/* Author: Christoffel Doorman
 * Date: 14-04-2020
 *
 * Compute metric fields of Schwarzschild spacetime
 *
 */

#include "kadath_spheric.hpp"
#include "mpi.h"

using namespace Kadath ;
int main(int argc, char** argv) {

    int rc = MPI_Init(&argc, &argv) ;
    int rank = 0 ;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;

    // 3D :
    int dim = 3 ;

    // Center of the coordinates
    Point center (dim) ;
    for (int i=1 ; i<=dim ; i++)
        center.set(i) = 0 ;

    // Number of domains and boundaries :
    int ndom = 5 ;
    Array<double> bounds (ndom-1) ;

    // Radius of the BH !
    double r = 1 ;
    double t = 0.0 ;
    double ome = 0.0 ;
    bounds.set(0) = r ; bounds.set(1) = (r+1.2) ; bounds.set(2) = 250 ; bounds.set(3) = 600 ;

    int tet_res = 17 ;
    int phi_res = 16 ;

    // Number of points
    Dim_array** resols = new Dim_array* [ndom] ;
    for (int i=0 ; i<ndom ; i++)
        resols[i] = new Dim_array(3) ;

    // Domain 0
    resols[0]->set(0) = 9 ; resols[0]->set(1) = tet_res ; resols[0]->set(2) = phi_res ;
    // Domain 1
    resols[1]->set(0) = 9 ; resols[1]->set(1) = tet_res ; resols[1]->set(2) = phi_res ;
    // Domain 2
    resols[2]->set(0) = 65 ; resols[2]->set(1) = tet_res ; resols[2]->set(2) = phi_res ;
    // Domain 3
    resols[3]->set(0) = 45 ; resols[3]->set(1) = tet_res ; resols[3]->set(2) = phi_res ;
    // Domain 4
    resols[4]->set(0) = 13 ; resols[4]->set(1) = tet_res ; resols[4]->set(2) = phi_res ;

    // assign variable names to domain radii
    double r0 = bounds(0) ;
    double r1 = bounds(1) ;
    double r2 = bounds(2) ;
    double r3 = bounds(3) ;

    // assing variable names to radius resolutions
    int r2_res = resols[2]->get(0) ;
    int r3_res = resols[3]->get(0) ;

    if (rank==0) {
	    cout << "r_res: " << resols[1]->get(0) << ", "
	                    << resols[2]->get(0) << ", "
                        << resols[3]->get(0) << ", "
                        << resols[4]->get(0) << ", "<< endl ;
	    cout << "theta_res: " << tet_res << endl ;
	    cout << "phi_res: " << phi_res << endl ;
	    cout << "r0:  " << r0 << endl ;
	    cout << "r1:  " << r1 << endl ;
	    cout << "r2:  " << r2 << endl ;
	    cout << "r3:  " << r3 << endl ;
	    cout << "n0:  " << t << endl ;
	    cout << "ome: " << ome << endl;
    }

    // Chebyshev or Legendre :
    int type_coloc = CHEB_TYPE ;

    // Sherical space :
    Space_spheric space(type_coloc, center, resols, bounds) ;

    // Tensorial basis used by the various tensors is cartesian
    Base_tensor basis (space, SPHERICAL_BASIS) ;

    // Initial guess for the conformal factor :
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


    /////// Phase 1: conformal factor and lapse function /////////
    {
        // Solve the equation in space outside the nucleus
        System_of_eqs init(space, 1, ndom - 1);

        // Define again the metric and system of equations
        Metric_flat fmet(space, basis);

        // Add metric
        fmet.set_system(init, "f");

        // Only one unknown
        init.add_var("Psi", conf);
        init.add_var("N", lapse);

        // One user defined constant
        init.add_cst("r", r);
        init.add_cst("t", t);

        // Inner BC
        space.add_inner_bc(init, "N=t");
        space.add_inner_bc(init, "dn(Psi) + 0.5 * Psi / r = 0");

        // Equations
        space.add_eq(init, "D_i D^i N + 2 * D_i Psi * D^i N / Psi = 0", "N", "dn(N)");
        space.add_eq(init, "D_i D^i Psi = 0", "Psi", "dn(Psi)");

        // Outer BC
        space.add_outer_bc(init, "N=1");
        space.add_outer_bc(init, "Psi=1");

        // Newton-Raphson
        double conv;
        bool endloop = false;
        int ite = 1;
        if (rank==0)
            cout << "Phase 1: solving Phi and N only " << endl ;
        while (!endloop) {
            endloop = init.do_newton(1e-6, conv);
            if (rank == 0)
                cout << "Newton iteration " << ite << " " << conv << endl;
            ite++;
        }
    }


    //////// Phase 2: conformal factor, lapse function and shift vector ///////
    // Vector parallel to the sphere (needed only for inner BC)
    Vector mm (space, CON, basis) ;
    for (int i=1 ; i<=3 ; i++)
	mm.set(i) = 0. ;
    Val_domain xx (space.get_domain(1)->get_cart(1)) ;
    Val_domain yy (space.get_domain(1)->get_cart(2)) ;
    mm.set(3).set_domain(1) = sqrt(xx*xx + yy*yy) ;
    mm.std_base() ;

    // Normal to sphere BH1 :
    Vector n (space, COV, basis) ;
    n.set(1) = 1. ;
    n.set(2) = 0. ;
    n.set(3) = 0. ;
    n.std_base() ;

    {
        // Solve the equation in space outside the nucleus
        System_of_eqs syst(space, 1, ndom - 1);

        // Define again the metric and system of equations
        Metric_flat fmet(space, basis) ;

        // Add metric
        fmet.set_system(syst, "f") ;

        // Only one unknown
        syst.add_var("Psi", conf) ;
        syst.add_var("N", lapse) ;
        syst.add_var("bet", shift) ;

        // One user defined constant
        syst.add_cst("r", r) ;
        syst.add_cst("t", t) ;
        syst.add_cst("n", n) ;
        syst.add_cst("ome", ome) ;
	    syst.add_cst("m", mm) ;


        // Definition of the extrinsic curvature :
        syst.add_def("A^ij = (D^i bet^j + D^j bet^i - (2. / 3.)* D_k bet^k * f^ij) /(2.* N)");
	    syst.add_def ("nn^i = n^i / sqrt(n_i * n^i)") ;

        // Inner BC
        space.add_inner_bc(syst, "N=t");
        space.add_inner_bc(syst, "bet^i = t / Psi^2 * nn^i + ome * m^i * r");
        space.add_inner_bc(syst, "4 * nn^i * D_i Psi / Psi + D_i nn^i + Psi^2 * A_ij * nn^i * nn^j = 0");

        // Equations
        space.add_eq(syst, "D_i D^i N + 2 * D_i Psi * D^i N / Psi - N * Psi^4 * A_ij *A^ij= 0", "N", "dn(N)");
        space.add_eq(syst, "D_i D^i Psi + Psi^5 *A_ij * A^ij / 8= 0", "Psi", "dn(Psi)");
        space.add_eq(syst, "D_j D^j bet^i + D^i D_j bet^j /3. - 2*A^ij*(D_j N -6*N*D_j Psi / Psi) =0", "bet^i", "dn(bet^i)");

        // Outer BC
        space.add_outer_bc(syst, "N=1");
        space.add_outer_bc(syst, "Psi=1");
        space.add_outer_bc(syst, "bet^i=0");


        // Newton-Raphson
        double conv;
        bool endloop = false;
        int ite = 1;
        if (rank == 0)
            cout << "Phase 2, solve for Psi, N, bet" << endl ;
        while (!endloop) {
            endloop = syst.do_newton(1e-8, conv);
            if (rank == 0)
                cout << "Newton iteration " << ite << " " << conv << endl;
            ite++;
        }
    }


    if (rank==0) {
        char dat_name[100] ;
        sprintf (dat_name, "SchwSyst_%d_%d_%.1f_%.0f_%.1f_%.0f_%.0f.dat", r2_res, r3_res, t, bounds(0), bounds(1), bounds(2), bounds(3)) ;
        FILE* fich = fopen(dat_name, "w") ;
        space.save(fich) ;
        fwrite_be (&r2_res, sizeof(int), 1, fich) ;
        fwrite_be (&r3_res, sizeof(int), 1, fich) ;
        fwrite_be (&r0, sizeof(double), 1, fich) ;
        fwrite_be (&r1, sizeof(double), 1, fich) ;
        fwrite_be (&r2, sizeof(double), 1, fich) ;
	    fwrite_be (&r3, sizeof(double), 1, fich) ;
        fwrite_be (&t, sizeof(double), 1, fich) ;
        fwrite_be (&ome, sizeof(double), 1, fich) ;
        conf.save(fich) ;
        lapse.save(fich) ;
        shift.save(fich) ;
        fclose(fich) ;
    }

    MPI_Finalize() ;

    return EXIT_SUCCESS ;
}


