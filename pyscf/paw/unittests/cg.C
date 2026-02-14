#include <cmath>
#include <stdio.h>
#include <iostream>

using namespace std;

int get_cast(double x) {
    int i;
    //i = (x / (int) x >= 1) ? (int) x : (int) x + 1 ;
    i = (int) x;
    //pout << "x " << x << " i " << i << endl;
    return i;
 }
 
 
double fbinom(double dn, double dr)
{
  double res;
  int n = get_cast(dn);
  int r = get_cast(dr);


  if(n==r || r==0) 
  {
    res = 1.0;
  }
  else if (r==1)
    res = n;
  else
    res = 1.0*n/(n-r)*fbinom((double)n-1,(double)r);
//    pout << n << " " << r<< " -> " << res << endl;
  return res;
}

double facto(double n) {
	double fac;
   int nint;
   nint = get_cast(n);
	fac=1.0;
	int i;
	if (n==0 || n==1)
		return fac;
	for (i=2; i<=nint; i++) 
		fac *= i;
	return fac;
}

int mone(double n) {
	int value;
   int nint;
   nint = get_cast(n);
   //pout << "nint %2 " << nint %2 << endl;
	if (nint % 2 == 0)
		value = 1;
	else
		value = -1;
	return (value);
}	


double clebsch(int nj1, int nm1, int nj2, int nm2, int nj3, int nm3) {


     double j1, j2, j3;
     double m1, m2, m3;
  
     //Converting to half its value
     j1=nj1/2.;
     j2=nj2/2.;
     j3=nj3/2.;
     m1=nm1/2.;
     m2=nm2/2.;
     m3=nm3/2.;
  
     double cleb=0.0;
     if ( j1 < 0 || j2 < 0 || j3 < 0 || abs(m1) > j1 || abs(m2) > j2 ||
        abs(m3) > j3 || j1 + j2 < j3 || abs(j1-j2) > j3 || m1 + m2 != m3) {
  
        cleb=0.0;
     }
     else
     {
        double factor = 0.0;
        double sum = 0.0;
        int t;
  
        double num1 = pow(2*j3+1,2);
        double num2 = fbinom(j1+j2+j3+1, j1+j2-j3);
        double num3 = fbinom(2*j3, j3+m3);
        double den1 = (2*j1+1);
        double den2 = (2*j2+1);
        double den3 = fbinom(j1+j2+j3+1, j1-j2+j3);
        double den4 = fbinom(j1+j2+j3+1, j2-j1+j3);
        double den5 = fbinom(2*j1, j1+m1);
        double den6 = fbinom(2*j2, j2+m2);
  
        double num = num1*num2*num3;
        double den = den1*den2*den3*den4*den5*den6;
        factor = sqrt(num/den);
  
        double mint = max(max(0., j1-m1-(j3-m3)), j2 + m2 - (j3 + m3)); 
        double maxt = min(min(j1-m1, j2+m2),j1+j2-j3);
        
        //pout << "mint " << mint << endl;
        //pout << "maxt " << maxt << endl;
        double bin1;
        double bin2;
        double bin3;
        for (t=mint; t<=maxt; t++) {
           bin1=fbinom(j1+j2-j3, t);
           bin2=fbinom(j3-m3,     j1-m1-t);
           bin3=fbinom(j3+m3,     j2+m2-t);
           sum = sum + mone(t)*bin1*bin2*bin3;
  
           //pout << "t " << t << endl;
           //pout << "sum " << sum << endl;
           //pout << "bin1 " << bin1 << endl;
           //pout << "bin2 " << bin2 << endl;
           //pout << "bin3 " << bin3 << endl;
        }
  
        cleb = factor*sum;
        //pout << "factor: " << factor << endl;
        //pout << "sum: " << sum << endl;
        //pout << "Clebsch: " << cleb << endl;
     }
        return cleb;
  }
  
int main(int input, char** output) {
    int Lmax = 10;
    printf("import jax.numpy as jnp\n");
    printf("import jax.scipy.special as jsp\n");
    printf("from jax import lax\n");
    printf("import ctypes\n");
    printf("import math\n");
    printf("import numpy as np\n");
    printf("Lmax = %d\n", Lmax);
    printf("CG = np.zeros((Lmax**2, Lmax**2, (2*Lmax)**2)) \n");

    for (int j1 = 0; j1 < Lmax; j1++)
    for (int m1 = -j1; m1 < j1+1; m1++)

    for (int j2 = 0; j2 < Lmax; j2++)
    for (int m2 = -j2; m2 < j2+1; m2++)

    for (int j3 = 0; j3 < 2*Lmax; j3++)
    for (int m3 = -j3; m3 < j3+1; m3++) {

        double cg =  clebsch(2*j1, 2*m1, 2*j2, 2*m2, 2*j3, 2*m3);
        if (abs(cg) > 1.e-5)
            printf("CG[%d, %d, %d]= %14.8f\n", (j1*j1 + (m1+j1)), (j2*j2 + (m2+j2)), (j3*j3 + (m3+j3)), cg);
    }
    return 0;
}
 