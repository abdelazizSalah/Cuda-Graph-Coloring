#include <time.h>
#include <stdio.h>
int loop() {
    for (int i = 0; i < 1000000000; i++) { 

    }
}

int main () { 
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    loop();
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f\n", cpu_time_used);
    return 0;
}