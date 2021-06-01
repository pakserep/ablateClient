//
// Created by pedram pakseresht on 2/18/21.
//

#include <petsc.h>



int main( int argc, char *argv[] )
{

    Vec a,b,c;
    DM packer;
    PetscErrorCode ierr;
    PetscInt i[4] = {0,1,2,3};
    PetscReal v[4] = {1.5,2.5,3.5,4.5};
    PetscReal w[4] = {5.5,6.5,7.5,8.5};

    // initialize Petsc ...
    ierr = PetscInitialize(&argc, &argv, NULL, "");CHKERRQ(ierr);

    ierr= VecCreate(PETSC_COMM_WORLD, &a); CHKERRQ(ierr);
    ierr= VecCreate(PETSC_COMM_WORLD, &b); CHKERRQ(ierr);

    ierr= VecSetSizes(a,PETSC_DECIDE,4); CHKERRQ(ierr);
    ierr = VecSetFromOptions(a); CHKERRQ(ierr);
    ierr= VecSetValues(a,4,i,v,INSERT_VALUES);CHKERRQ(ierr);
    ierr= VecAssemblyBegin(a);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(a);CHKERRQ(ierr);
    ierr = VecView(a,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr= VecSetSizes(b,PETSC_DECIDE,4); CHKERRQ(ierr);
    ierr = VecSetFromOptions(b); CHKERRQ(ierr);
    ierr= VecSetValues(b,4,i,w,INSERT_VALUES);CHKERRQ(ierr);
    ierr= VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
    ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);




    ierr = PetscFinalize();exit(ierr);




}