#include <stdio.h>
#include <arm_sve.h>

#ifndef __ARM_FEATURE_SVE
#warning "Make sure to compile for SVE!"
#endif

int main()
{
    printf("SVE vector length is: %ld bits\n", 8 * svcntb());
}