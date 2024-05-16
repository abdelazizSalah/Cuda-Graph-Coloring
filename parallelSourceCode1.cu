#include <thrust/count.h>

__global__ void color_jpl_kernel(int n, int c, const int *Ao, 
                                 const int *Ac, const double *Av, 
                                 const int *randoms, int *colors)
{   
  for (int i = threadIdx.x+blockIdx.x*blockDim.x; 
       i < n; 
       i += blockDim.x*gridDim.x) 
  {   
    bool f=true; // true iff you have max random

    // ignore nodes colored earlier
    if ((colors[i] != -1)) continue; 

    int ir = randoms[i];

    // look at neighbors to check their random number
    for (int k = Ao[i]; k < Ao[i+1]; k++) {        
      // ignore nodes colored earlier (and yourself)
      int j = Ac[k];
      int jc = colors[j];
      if (((jc != -1) && (jc != c)) || (i == j)) continue; 
      int jr = randoms[j];
      if (ir <= jr) f=false;         
    }

    // assign color if you have the maximum random number
    if (f) colors[i] = c;
  }
}

#define CUDA_MAX_BLOCKS 8

void color_jpl(int n, 
               const int *Ao, const int *Ac, const double *Av, 
               int *colors) 
{
    int *randoms; // allocate and init random array 
    thrust::fill(colors, colors+n, -1); // init colors to -1

    for(int c=0; c < n; c++) {
        int nt = 256;
        int nb = min((n + nt - 1)/nt,CUDA_MAX_BLOCKS);
        color_jpl_kernel<<<nb,nt>>>(n, c, 
                                    Ao, Ac, Av, 
                                    randoms, 
                                    colors);
        int left = (int)thrust::count(colors,colors+n,-1);
        if (left == 0) break;
    }
}