#include <math.h>

const double sqrt3 = 1.7320508; // sqrt(3)

double integrand_dir(int n, double *x, void *user_data)
{
        double r = *(double *)user_data;
        double a = *((double *)user_data + 1);
        double xx=x[0]*x[0], yy=x[1]*x[1], zz=x[2]*x[2];

        return pow(fmax(0., 1.-0.5*(xx+yy+zz)), a) * cos(r/sqrt3*(x[0]+x[1]+x[2]));
}

double integrand_pos(int n, double *x, void *user_data)
{
        double r = *(double *)user_data;
        double a = *((double *)user_data + 1);
        double gamma = *((double *)user_data + 2);
        double xx=x[0]*x[0], yy=x[1]*x[1], zz=x[2]*x[2];

        return exp(-pow(a*sqrt(xx+yy+zz), gamma)) * cos(r/sqrt3*(x[0]+x[1]+x[2]));
}
