//
// Created by pedram pakseresht on 2/18/21.
//

static char help[] = "settling in quiescent fluid";

#include <petsc.h>
#include "incompressibleFlow.h"
#include "mesh.h"
#include "particleInertial.h"
#include "particles.h"
#include "particleInitializer.h"
#include "petscviewer.h"


static PetscErrorCode uniform_u(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscInt d;
    for (d = 0; d < Dim; ++d)
        u[d] = 0.0;
    return 0;

}

static PetscErrorCode uniform_u_t(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscInt d;
    for (d = 0; d < Dim; ++d)
        u[d] = 0.0;
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

    ierr = DMSetOutputSequenceNumber(particlesData->dm, step, crtime);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    if (step == 0) {
        ierr = ParticleViewFromOptions(particlesData, "-particle_init");
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
    } else {
        ierr = ParticleViewFromOptions(particlesData, "-particle_view");
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
    }

    PetscFunctionReturn(0);
}

static PetscErrorCode ParticleInertialInitialize(ParticleData particles) {

    PetscFunctionBeginUser;
    PetscErrorCode  ierr;
    Vec vel,diam,dens;
    DM particleDm = particles->dm;

    // input parameters required for particles
    PetscScalar partVel = 0.0;
    PetscScalar partDiam = 0.22;
    PetscScalar partDens = 90.0;
    PetscScalar fluidDens = 1.0;
    PetscScalar fluidVisc = 1.0;
    PetscScalar gravity[3] = {0.0,1.0,0.0};

    ierr = DMSwarmCreateGlobalVectorFromField(particleDm,ParticleVelocity, &vel);CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(particleDm,ParticleDiameter, &diam);CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(particleDm,ParticleDensity, &dens);CHKERRQ(ierr);
    // set particle velocity, diameter and density
    ierr = VecSet(vel, partVel);CHKERRQ(ierr);
    ierr = VecSet(diam, partDiam);CHKERRQ(ierr);
    ierr = VecSet(dens, partDens);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(particleDm,ParticleVelocity, &vel);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(particleDm,ParticleDiameter, &diam);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(particleDm,ParticleDensity, &dens);CHKERRQ(ierr);

    InertialParticleParameters *data;
    PetscNew(&data);
    particles->data =data;

    // set fluid parameters
    data->fluidDensity = fluidDens;
    data->fluidViscosity = fluidVisc;
    data->gravityField[0] = gravity[0];
    data->gravityField[1] = gravity[1];
    data->gravityField[2] = gravity[2];
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
    PetscInt Dim = 2;
    PetscReal dt = 0.05;       // dt for time stepper
    PetscInt max_steps = 10; // maximum time steps

    PetscReal Re = 0.1;       // Reynolds number
    PetscReal St = 1.0;       // Strouhal number
    PetscReal Pe = 1.0;       // Peclet number
    PetscReal Mu = 1.0;      // viscosity
    PetscReal K_input = 1.0;  // thermal conductivity
    PetscReal Cp = 1.0;    // heat capacity
    // PetscInt Np=10;


    // initialize Petsc ...
    ierr = PetscInitialize(&argc, &argv, NULL, "");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Settling particle in a quiescent fluid \n");CHKERRQ(ierr);

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
    /*
    ierr = PetscPrintf(PETSC_COMM_WORLD, "*** non-dimensional parameters ***\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "St=%g\n", flowParameters.strouhal);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Re=%g\n", flowParameters.reynolds);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Pe=%g\n", flowParameters.peclet);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Mu=%g\n", flowParameters.mu);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "K=%g\n", flowParameters.k);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Cp=%g\n", flowParameters.cp);CHKERRQ(ierr);
    */

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
    ierr = ParticleInertialCreate(&particles, Dim);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // link the flow to the particles
    ierr = ParticleInitializeFlow(particles, flowData);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // name the particle domain
    ierr = PetscObjectSetOptionsPrefix((PetscObject)(particles->dm), "particles_");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscObjectSetName((PetscObject)particles->dm, "Particles");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // initialize the particles position
    ierr = ParticleInitialize(dm, particles->dm);CHKERRABORT(PETSC_COMM_WORLD, ierr);
    // initialize inertial particles velocity, diameter and density
    ierr = ParticleInertialInitialize(particles);CHKERRABORT(PETSC_COMM_WORLD, ierr);

    //ierr = ParticleInertialInitialize(particles->dm);CHKERRABORT(PETSC_COMM_WORLD, ierr);

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

    ierr = ParticleInertialSetupIntegrator(particles, particleTs, flowData);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    // ierr = VecView(particleVelocity,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


    // Solve the one way coupled system
    ierr = TSSolve(ts, flowData->flowField);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);


    // Cleanup
    ierr = DMDestroy(&dm);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSDestroy(&ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSDestroy(&particleTs);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = FlowDestroy(&flowData);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = ParticleInertialDestroy(&particles);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscFinalize();
    exit(ierr);

}