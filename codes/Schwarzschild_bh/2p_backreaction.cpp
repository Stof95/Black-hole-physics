/* Author: Christoffel Doorman
 * Date: 14-04-2020
 *
 * Compute backreaction of complex scalar field in Schwarzschild spacetime
 *
 * Inputs:
 * .dat file containing metric fields in conformal flat spacetime
 * .dat file containing scalar field without backreaction
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

    int r2_res, r3_res ; // radial resolution in domain 2 and 3
    double r0, r1, r2, r3 ; // radii of domains
    double t, ome ; // boundary value lapse function and bh rotation (=0 for Schwarzschild)

    // scalar field mass and frequency
    double mu, freq ;

    // read background metric and scalar field data
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
    fread_be (&freq, sizeof(double), 1, fich) ;
    freqd_be (&mu, sizeof(double), 1, fich) ;

    Scalar conf (space, fich) ;
    Scalar lapse (space, fich) ;
    Vector shift (space, fich) ;
    Scalar field1 (space, fich) ;
    Scalar field2 (space, fich) ;
    fclose(fich) ;

    if (rank==0) {
        cout << "Backreaction for cloud 2p" << endl ;
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

    // retrieve number of domains
    int ndom = space.get_nbr_domains() ;

    // Computation adm mass before backreaction :
    Val_domain integ_adm_init (conf(ndom-1).der_r()) ;
    double adm_init = -space.get_domain(ndom-1)->integ(integ_adm_init, OUTER_BC)/2/M_PI ;

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

    // copy fields for inner boundary values
    Scalar p1 (space) = field1 ;
    Scalar p2 (space) = field2 ;


    Base_tensor basis (shift.get_basis()) ;
    Metric_flat fmet (space, basis) ;

    // initialize system carrying all equations
    System_of_eqs syst (space, 1, ndom-1) ;

    // initialize conformally flat metric
    fmet.set_system (syst, "f") ;

    // fields to solve
    syst.add_var("Psi", conf) ;
    syst.add_var("N", lapse) ;
    syst.add_var("bet", shift) ;
    syst.add_var ("ph1", field1) ;
    syst.add_var ("ph2", field2) ;

    // user defined constants
    //    syst.add_cst ("bet", shift) ;
    syst.add_cst("r", r0) ;
    syst.add_cst("t", t) ;
    syst.add_cst("n", n) ;
    syst.add_cst("ome", ome) ;
    syst.add_cst("m", mm) ;
    syst.add_cst ("mu", mu) ;
    syst.add_cst ("frq", freq) ;
    syst.add_cst ("pi", M_PI)

    // inner boundary values scalar fields
    syst.add_cst ("p1", p1) ;
    syst.add_cst ("p2", p2) ;

    // Definition of the extrinsic curvature
    syst.add_def("A^ij = (D^i bet^j + D^j bet^i - (2. / 3.)* D_k bet^k * f^ij) /(2.* N)");
    syst.add_def ("nn^i = n^i / sqrt(n_i * n^i)") ;

    // definitions of momenta
    syst.add_def ("LP1 = (N * frq * bet^i * D_i ph2) + (N * frq * frq * ph1)/2 - (N * bet^i * bet^j * D_i D_j ph1)/2 - (N * bet^i * D_i bet^j * D_j ph1)/2 + (bet^i * D_i ph1 * bet^j * D_j N)/2 - (frq * ph2 * bet^i * D_i N)/2" ) ;
    syst.add_def ("LP2 = (N * frq * bet^i * D_i ph1) + (N * frq * frq * ph2)/2 - (N * bet^i * bet^j * D_i D_j ph2)/2 - (N * bet^i * D_i bet^j * D_j ph2)/2 + (bet^i * D_i ph2 * bet^j * D_j N)/2 - (frq * ph1 * bet^i * D_i N)/2" ) ;

    // definitions of the matter equations
    syst.add_def ("rho = (frq^2/N^2 + mu^2) * (ph1^2 + ph2^2) / 2 + (D_i ph1 * D^i ph1 + D_i ph2 * D^i ph2) / 2") ;
    syst.add_def ("mom^i = - frq * (ph2 * D^i ph1 - ph1 * D^i ph2) / N ") ;
    syst.add_sef ("RpS = (2 * frq^2 / N^2 - mu^2) * (ph1^2 + ph2^2) ") ;

    // inner BC metric
    space.add_inner_bc(syst, "N=t");
    space.add_inner_bc(syst, "bet^i = t / Psi^2 * nn^i + ome * m^i * r");
    space.add_inner_bc(syst, "4 * nn^i * D_i Psi / Psi + D_i nn^i + Psi^2 * A_ij * nn^i * nn^j = 0");

    // outer BC metric
    space.add_outer_bc(syst, "N=1");
    space.add_outer_bc(syst, "Psi=1");
    space.add_outer_bc(syst, "bet^i=0");

    // inner BC scalar field
    syst.add_eq_bc (1, INNER_BC, "ph1=p1") ;
    syst.add_eq_bc (1, INNER_BC, "ph2=p2") ;

    // outer BC scalar field
    syst.add_eq_bc (2, OUTER_BC, "ph1=0.") ;
    syst.add_eq_bc (2, OUTER_BC, "ph2=0.") ;

    // Einstein Equations
    space.add_eq(syst, "D_i D^i N + 2 * D_i Psi * D^i N / Psi - N * Psi^4 * A_ij *A^ij - 4 * pi * N * Psi^4 * RpS = 0", "N", "dn(N)") ;
    space.add_eq(syst, "D_i D^i Psi + Psi^5 *A_ij * A^ij / 8 + 2*pi * rho * Psi^5 = 0", "Psi", "dn(Psi)") ;
    space.add_eq(syst, "D_j A^ij + 10 * A^ij * D_j Psi / Psi - 8 * pi * mom^i / Psi^4", "bet^i", "dn(bet^i)") ;

    // KG equations
    space.add_eq (syst, "N^2 * D_i N * D^i ph1 / Psi^4 + N^3 * D_i D^i ph1 / Psi^4 + 2 * N^3 * D_i ph1 * D^i Psi / Psi^5 + 2 * LP1 - N^3 * mu * mu * ph1 = 0", "ph1", "dn(ph1)") ;
    space.add_eq (syst, "N^2 * D_i N * D^i ph2 / Psi^4 + N^3 * D_i D^i ph2 / Psi^4 + 2 * N^3 * D_i ph2 * D^i Psi / Psi^5 + 2 * LP2 - N^3 * mu * mu * ph2 = 0", "ph2", "dn(ph2)") ;


    // Newton-Raphson
    double conv;
    bool endloop = false;
    int ite = 1;
    if (rank == 0)
        cout << "Solve all fields simultaneously" << endl ;
    while (!endloop) {
        endloop = syst.do_newton(1e-8, conv);
        if (rank == 0) {
            cout << "Newton iteration " << ite << " " << conv << endl;
            ite++;

            // Output interim results
            char name[100];
            sprintf(name, "BR_2p_%d_%d_%.0f_%.1f_%.0f_%.0f_%.1f_%.2f.dat", r2_res, r3_res, r0, r1, r2, r3, t, mu);
            FILE *fich = fopen(name, "w");
            space.save(fich);
            fwrite_be(&r2_res, sizeof(int), 1, fich);
            fwrite_be(&r3_res, sizeof(int), 1, fich);
            fwrite_be(&r0, sizeof(double), 1, fich);
            fwrite_be(&r1, sizeof(double), 1, fich);
            fwrite_be(&r2, sizeof(double), 1, fich);
            fwrite_be(&r3, sizeof(double), 1, fich);
            fwrite_be(&t, sizeof(double), 1, fich);
            fwrite_be(&ome, sizeof(double), 1, fich);
            fwrite_be(&freq, sizeof(double), 1, fich);
            fwrite_be(&mu, sizeof(double), 1, fich);
            conf.save(fich);
            lapse.save(fich);
            shift.save(fich);
            field1.save(fich);
            field2.save(fich);
            fclose(fich);
        }
    }

    // Computation adm mass after backreaction :
    Val_domain integ_adm (conf(ndom-1).der_r()) ;
    double adm = -space.get_domain(ndom-1)->integ(integ_adm, OUTER_BC)/2/M_PI ;
    if (rank==0) {
        cout << "ADM mass before backreaction: " << adm_init << endl ;
        cout << "ADM mass after backreaction:  " << adm << endl ;
    }

    // Output final results
    if (rank==0) {
        char name[100];
        sprintf(name, "BR_2p_%d_%d_%.0f_%.1f_%.0f_%.0f_%.1f_%.2f.dat", r2_res, r3_res, r0, r1, r2, r3, t, mu);
        FILE *fich = fopen(name, "w");
        space.save(fich);
        fwrite_be(&r2_res, sizeof(int), 1, fich);
        fwrite_be(&r3_res, sizeof(int), 1, fich);
        fwrite_be(&r0, sizeof(double), 1, fich);
        fwrite_be(&r1, sizeof(double), 1, fich);
        fwrite_be(&r2, sizeof(double), 1, fich);
        fwrite_be(&r3, sizeof(double), 1, fich);
        fwrite_be(&t, sizeof(double), 1, fich);
        fwrite_be(&ome, sizeof(double), 1, fich);
        fwrite_be(&freq, sizeof(double), 1, fich);
        fwrite_be(&mu, sizeof(double), 1, fich);
        conf.save(fich);
        lapse.save(fich);
        shift.save(fich);
        field1.save(fich);
        field2.save(fich);
        fclose(fich) ;
    }