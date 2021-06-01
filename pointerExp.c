//
// Created by pedram pakseresht on 2/18/21.
//

// #include <petsc.h>

#include <printf.h>
#include <stdlib.h>

void setNumber(double *zxy);


int main( int argc, char *argv[] )
{
  double *x = NULL;
    x = malloc( sizeof(double ));
    *x =2.0;

 printf("hello world %g\n", *x);
 setNumber(x);
 printf("hello world %g\n", *x);
 printf("size of %lu", sizeof(double *));

 free(x);
 x=NULL;


}

void setNumber(double *z){
    printf("hello world %g\n", *z);
    *z =1;
    printf("hello world %g\n", *z);
}


