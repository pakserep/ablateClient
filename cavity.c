//
// Created by pedram pakseresht on 2/18/21.
//

static char help[] = "Simple Petsc program";

#include <petsc.h>
#include "incompressibleFlow.h"
#include "mesh.h"
#include "particleTracer.h"
#include "particles.h"
#include "particleInitializer.h"
#include "petscviewer.h"


static PetscErrorCode uniform_u(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *u, void *ctx) {
    //PetscInt d;
    //for (d = 0; d < Dim; ++d)
    //    u[d] = 1.0;
    //return 0;
    if (X[1]==1.0){
        u[0] = 1.0;
    }
    else {
        u[0] = 0.0;
    }
        u[1] =0.0;
    // u[0] = X[1] - X[1]*X[1] ;
    // u[1] = 0.0;
    return 0;
}

static PetscErrorCode uniform_u_t(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 0.0;
    u[1] = 0.0;
    return 0;
}

static PetscErrorCode uniform_p(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = 0.0;
    return 0;
}

static PetscErrorCode uniform_T(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 0.0;
    return 0;
}
static PetscErrorCode uniform_T_t(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 0.0;
    return 0;
}




static PetscErrorCode SetInitialConditions(TS ts, Vec u) {

    PetscErrorCode (*initFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx) = {uniform_u,uniform_p,uniform_T};
    // u here is the solution vector including velocity, temperature and pressure fields.
    DM dm;
    PetscReal t;
    PetscErrorCode ierr;
    PetscFunctionBegin;

    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = TSGetTime(ts, &t);

    CHKERRQ(ierr);

    ierr = DMProjectFunction(dm, 0.0, initFuncs, NULL, INSERT_ALL_VALUES, u);
    CHKERRQ(ierr);
    // get the flow to apply the completeFlowInitialization method
    ierr = IncompressibleFlow_CompleteFlowInitialization(dm, u);
    CHKERRQ(ierr);


    PetscFunctionReturn(0);
}


static PetscErrorCode MonitorFlowAndParticleError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) {
    // PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    // void *ctxs[3];
    DM dm;
    PetscDS ds;
    Vec v;
    PetscReal ferrors[3];
    PetscInt f;
    PetscErrorCode ierr;
    PetscInt num;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDS(dm, &ds);
    CHKERRQ(ierr);

    // get the particle data from the context
    ParticleData particlesData = (ParticleData)ctx;
    PetscInt particleCount;
    ierr = DMSwarmGetSize(particlesData->dm, &particleCount);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // compute the average particle location
    const PetscReal *coords;
    PetscInt dims;
    PetscReal avg[3] = {0.0, 0.0, 0.0};
    ierr = DMSwarmGetField(particlesData->dm, DMSwarmPICField_coor, &dims, NULL, (void **)&coords);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    for (PetscInt p = 0; p < particleCount; p++) {
        for (PetscInt n = 0; n < dims; n++) {
            avg[n] += coords[p * dims + n] / particleCount;  // PetscReal
        }
    }
    ierr = DMSwarmRestoreField(particlesData->dm, DMSwarmPICField_coor, &dims, NULL, (void **)&coords);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g, %2.3g] ParticleCount: %d\n",
                       (int)step,
                       (double)crtime,
                       (double)ferrors[0],
                       (double)ferrors[1],
                       (double)ferrors[2],
                       particleCount,
                       (double)avg[0],
                       (double)avg[1]);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Avg Particle Location: [%2.3g, %2.3g, %2.3g]\n", (double)avg[0], (double)avg[1], (double)avg[2]);

    CHKERRABORT(PETSC_COMM_WORLD, ierr);


     ierr = VecViewFromOptions(u, NULL, "-vec_view");
     CHKERRABORT(PETSC_COMM_WORLD, ierr);

    //  ierr = VecView(u, NULL);CHKERRABORT(PETSC_COMM_WORLD, ierr);
    // ierr = PetscPrintf(PETSC_COMM_WORLD, "current time=%g\n", crtime);CHKERRQ(ierr);

    ierr = DMSetOutputSequenceNumber(particlesData->dm, step, crtime);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

     /*
     if (step == 0 ){
        ierr = VecViewFromOptions(u, NULL, "-vec_init");
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
     } else {
        ierr = VecViewFromOptions(u, NULL, "-vec_view");
         CHKERRABORT(PETSC_COMM_WORLD, ierr);
     }
     */

    if (step == 0) {
        ierr = ParticleViewFromOptions(particlesData, NULL, "-particle_init");
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
    } else {
        ierr = ParticleViewFromOptions(particlesData, NULL, "-particle_view");
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
    }

    PetscFunctionReturn(0);
}





int main( int argc, char *argv[] )
{

    DM dm;     // domain definition
    TS ts;     // time-stepper
    PetscErrorCode ierr;

    PetscBag parameterBag; // constant flow parameters
    Vec flowField;         // flow solution vector
    FlowData flowData;


    PetscReal t;
    PetscInt Dim=2;
    PetscReal dt=0.1;       // dt for time stepper
    PetscInt max_steps=10; // maximum time steps

    PetscReal Re=1.0;       // Reynolds number
    PetscReal St=1.0;       // Strouhal number
    PetscReal Pe=1.0;       // Peclet number
    PetscReal Mu=1.0;      // viscosity
    PetscReal K_input=1.0;  // thermal conductivity
    PetscReal Cp = 1.0;    // heat capacity
    // PetscInt Np=10;


    // initialize Petsc ...
    ierr = PetscInitialize(&argc, &argv, NULL, "");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "*** Set up a simple flow *** \n");CHKERRQ(ierr);

    // setup the ts
    ierr = TSCreate(PETSC_COMM_WORLD, &ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = CreateMesh(PETSC_COMM_WORLD, &dm, PETSC_TRUE, Dim);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSSetDM(ts, dm);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);


    //output the mesh
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Setup the flow data
    ierr = FlowCreate(&flowData);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // setup problem
    ierr = IncompressibleFlow_SetupDiscretization(flowData, dm);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);



    IncompressibleFlowParameters flowParameters;


    // changing non-dimensional parameters manually here ...
    flowParameters.strouhal = St;
    flowParameters.reynolds = Re;
    flowParameters.peclet = Pe;
    flowParameters.mu = Mu;
    flowParameters.k = K_input;
    flowParameters.cp = Cp;

    // print out the parameters
    ierr = PetscPrintf(PETSC_COMM_WORLD, "*** non-dimensional parameters ***\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "St=%g\n", flowParameters.strouhal);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Re=%g\n", flowParameters.reynolds);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Pe=%g\n", flowParameters.peclet);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Mu=%g\n", flowParameters.mu);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "K=%g\n", flowParameters.k);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Cp=%g\n", flowParameters.cp);CHKERRQ(ierr);


    // Start the problem setup
    PetscScalar constants[TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS];
    ierr = IncompressibleFlow_PackParameters(&flowParameters, constants);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = IncompressibleFlow_StartProblemSetup(flowData, TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, constants);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);


    // Override problem with source terms, boundary, and set the exact solution
    PetscDS prob;
    ierr = DMGetDS(dm, &prob);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Setup Boundary Conditions
    // Note: DM_BC_ESSENTIAL is a Dirichlet BC.
    PetscInt id;
    id = 3;
    ierr = PetscDSAddBoundary(
        prob, DM_BC_ESSENTIAL, "top wall velocity", "marker", VEL, 0, NULL, (void (*)(void))uniform_u, (void (*)(void))uniform_u_t, 1, &id, NULL);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    id = 1;
    ierr = PetscDSAddBoundary(
        prob, DM_BC_ESSENTIAL, "bottom wall velocity", "marker", VEL, 0, NULL, (void (*)(void))uniform_u, (void (*)(void))uniform_u_t, 1, &id, NULL);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    id = 2;
    ierr = PetscDSAddBoundary(
        prob, DM_BC_ESSENTIAL, "right wall velocity", "marker", VEL, 0, NULL, (void (*)(void))uniform_u, (void (*)(void))uniform_u_t, 1, &id, NULL);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    id = 4;
    ierr = PetscDSAddBoundary(
        prob, DM_BC_ESSENTIAL, "left wall velocity", "marker", VEL, 0, NULL, (void (*)(void))uniform_u, (void (*)(void))uniform_u_t, 1, &id, NULL);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    id = 3;
    ierr = PetscDSAddBoundary(
        prob, DM_BC_ESSENTIAL, "top wall temp", "marker", TEMP, 0, NULL, (void (*)(void))uniform_T, (void (*)(void))uniform_T_t, 1, &id, NULL);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    id = 1;
    ierr = PetscDSAddBoundary(
        prob, DM_BC_ESSENTIAL, "bottom wall temp", "marker", TEMP, 0, NULL, (void (*)(void))uniform_T, (void (*)(void))uniform_T_t, 1, &id, parameterBag);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    id = 2;
    ierr = PetscDSAddBoundary(
        prob, DM_BC_ESSENTIAL, "right wall temp", "marker", TEMP, 0, NULL, (void (*)(void))uniform_T, (void (*)(void))uniform_T_t, 1, &id, parameterBag);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    id = 4;
    ierr = PetscDSAddBoundary(
        prob, DM_BC_ESSENTIAL, "left wall temp", "marker", TEMP, 0, NULL, (void (*)(void))uniform_T, (void (*)(void))uniform_T_t, 1, &id, parameterBag);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);


    ierr = IncompressibleFlow_CompleteProblemSetup(flowData, ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Name the flow field
    ierr = PetscObjectSetName(((PetscObject)flowData->flowField), "Numerical Solution");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);


    ierr = SetInitialConditions(ts, flowData->flowField);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);


    ierr = TSGetTime(ts, &t);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    //  *** added by Pedram for dt and number of time steps ***
    ierr = TSSetTimeStep(ts,dt);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    ierr = TSSetMaxSteps(ts,max_steps);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    // ***********************************

    ierr = DMSetOutputSequenceNumber(dm, 0, t);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // checks for convergence ...
    ierr = DMTSCheckFromOptions(ts, flowData->flowField);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);


    ParticleData particles;

    // Setup the particle domain
    ierr = ParticleTracerCreate(&particles, Dim);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // link the flow to the particles
    ParticleInitializeFlow(particles, flowData);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // name the particle domain
    ierr = PetscObjectSetOptionsPrefix((PetscObject)(particles->dm), "particles_");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscObjectSetName((PetscObject)particles->dm, "Particles");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // initialize the particles
    ParticleInitialize(dm, particles->dm);

    // setup the flow monitor to also check particles
    ierr = TSMonitorSet(ts, MonitorFlowAndParticleError, particles, NULL);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSSetFromOptions(ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);


    // Setup particle position integrator
    TS particleTs;
    ierr = TSCreate(PETSC_COMM_WORLD, &particleTs);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)particleTs, "particle_");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);


    ierr = ParticleTracerSetupIntegrator(particles, particleTs, ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);


    // setup the initial conditions for error computing
    /*
    ierr = TSSetComputeExactError(particleTs, computeParticleError);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSSetComputeInitialCondition(particleTs, setParticleExactSolution);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    */

    // copy over the initial location
    /*
    PetscReal *coord;
    PetscReal *initialLocation;
    PetscInt numberParticles;
    ierr = DMSwarmGetLocalSize(particles->dm, &numberParticles);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMSwarmGetField(particles->dm, DMSwarmPICField_coor, NULL, NULL, (void **)&coord);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMSwarmGetField(particles->dm, "InitialLocation", NULL, NULL, (void **)&initialLocation);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    for (int i = 0; i < numberParticles * 2; ++i) {
        initialLocation[i] = coord[i];
    }
    ierr = DMSwarmRestoreField(particles->dm, DMSwarmPICField_coor, NULL, NULL, (void **)&coord);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMSwarmRestoreField(particles->dm, "InitialLocation", NULL, NULL, (void **)&initialLocation);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    */

    // Solve the one way coupled system
    ierr = TSSolve(ts, flowData->flowField);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Compare the actual vs expected values
    /*
    ierr = DMTSCheckFromOptions(ts, flowData->flowField);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    */

    //PetscBool hdf5;
    // ierr = ParticleView(particles, );CHKERRQ(ierr);

    // PETSCVIEWERHDF5

    // Cleanup
    ierr = DMDestroy(&dm);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSDestroy(&ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSDestroy(&particleTs);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = FlowDestroy(&flowData);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = ParticleTracerDestroy(&particles);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscFinalize();
    exit(ierr);

    // _lower 0.25,0.25 _upper 0.75,0.75
    // Npb 10
    // _ts_dt 0.1
    // _layout_type box
    // _ts_convergence_estimate
    // _num_refine 1
    // -particle_ts_monitor_cancel
    // -ts_monitor_cancel


    // -particle_layout_type box
    // -particle_lower 0.1,0.1
    // -particle_upper 0.3,0.3
    // -Npb 5
    // -particle_ts_max_steps 10
    // -particle_ts_dt 0.05


    /*
    -ksp_type
    fgmres
    -ksp_gmres_restart
    10
    -ksp_rtol
    1.0e-9
    -ksp_error_if_not_converged
    -pc_type
    fieldsplit
    -pc_fieldsplit_0_fields
    0,2
      -pc_fieldsplit_1_fields
    1
    -pc_fieldsplit_type
    schur
    -pc_fieldsplit_schur_factorization_type
    full
    -fieldsplit_0_pc_type
    lu
    -fieldsplit_pressure_ksp_rtol
    1e-10
    -fieldsplit_pressure_pc_type
    jacobi
     */


    /*
     * -Npb 1 -particle_lower 0.05,0.1 -particle_upper 0.3,0.9 -particle_layout_type box -vec_init hdf5:/Users/pedram/scratch/sol.h5 -vec_view hdf5:/Users/pedram/scratch/sol.h5::append -dm_view hdf5:/Users/pedram/scratch/sol.h5 -dm_refine 2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 -ksp_type fgmres -ksp_rtol 1.0e-9 -ksp_atol 1.0e-12 -ksp_error_if_not_converged -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_0_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi -dm_plex_separate_marker -snes_monitor -ksp_monitor -snes_converged_reason -ksp_converged_reason -particle_view hdf5:/Users/pedram/scratch/solP.h5::append -particle_init hdf5:/Users/pedram/scratch/solP.h5
     */





}