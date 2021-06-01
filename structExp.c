//
// Created by pedram pakseresht on 2/18/21.
//

// #include <petsc.h>

#include <printf.h>
#include <stdlib.h>

void setNumber(double *zxy);

typedef struct{

    double x;
    double z;
    int g[3];

}Pstruct;


void printStruct(Pstruct str) {

    printf("print struct %f %f %d %d %d \n", str.x, str.z, str.g[0], str.g[1], str.g[2]);
//    printf("print x %lf \n", str.z );
//    printf("print x %lf \n", str.g[1] );

};


void setValues(){

}

int main( int argc, char *argv[] )
{

    Pstruct a;
    a.x = 7.0;
    a.z = 8.0;
    a.g[0] = 1;
    a.g[1] = 2;
    a.g[2] = 3;

    Pstruct b;
    b.x = 5.0;
    b.z = 4.0;
    b.g[0] = 3;
    b.g[1] = 1;
    b.g[2] = 2;


    printStruct(a);
    printStruct(b);




//    printf("x of b is %lf \n",b.x);
//printf("size of struc %lu \n", sizeof(Pstruct));
//printf("size of int %lu", sizeof(int [4]));

}




