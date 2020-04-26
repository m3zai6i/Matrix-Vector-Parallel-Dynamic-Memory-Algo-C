/* 
	Matrix-Vector Multiplication
	Parallel Algorithm in C
*/
// E-mail: m3zai6i@gmail.com
// Compile: mpicc mat_vec_par_dyn.c -o output
// Run: mpirun -np <rows> code <rows> <columns>
// Run: mpirun -np 10 output 10 10
// Rows and number of processes must be same

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
	double time1, time2,duration,global;
	
	int size,rank;
	MPI_Init(&argc,&argv);
	time1 = MPI_Wtime(); //Start Time
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	//Taking arguments from the user
	int r = atoi(argv[1]); 
	int c = atoi(argv[2]);

	//int mat[r][c],vec[c];
	//int mat_slice[c];
	int res = 0;
  
  int *mat = (int *)malloc(r * c * sizeof(int)); //Martrix using single pointer
  int *vec = (int *) malloc(c * sizeof(int)); //vector
	int *mat_slice = (int *) malloc(c * sizeof(int)); //Matrix Slice
	
	if (rank == 0)
	{
		int i,j,count=0;
		//Matrix Initialization in sequence
		printf("Our Matrix is = \n");
		for (i = 0; i < r; i++)
		{
			for (j = 0; j < c; j++)
			{
				//mat[i][j] = ++count;
				*(mat + i*c + j) = ++count;
				//printf("%4d\t", mat[i][j]);
				printf("%4d\t", *(mat + i*c + j));
			}
			printf("\n");
		}

		count = 0;
		//Vector Initialization in sequence
		printf("Our Vector is = \n");
		for (i = 0; i < c; i++)
		{
			vec[i] = ++count;
			printf("%4d\n",vec[i]);
		}
	}

	//Broadcasting VECTOR to all processes from rank 0
	MPI_Bcast(vec, c, MPI_INT, 0, MPI_COMM_WORLD);

	//Scattering MATRIX to other processes   
	MPI_Scatter(mat, c, MPI_INT, mat_slice, c, MPI_INT, 0, MPI_COMM_WORLD);
	
	//Every Process is multiplying its matrix_slice with broadcasted vector
	for (int i = 0; i < c; i++)
	{
		res = res + (mat_slice[i]*vec[i]);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	//Gathering all data of result_matrix into rank/process 1
	int rbuf[size];
	MPI_Gather(&res, 1, MPI_INT, &rbuf, 1, MPI_INT, 1, MPI_COMM_WORLD);

	if (rank==1)
	{		
	 	printf("Our Result Vector is = \n");
		for (int i = 0; i < r; i++)
	    {
		    printf("%4d\n",rbuf[i]);
	 	}
	}	
	
	time2 = MPI_Wtime();  //End Time
	duration = time2 - time1;
   	MPI_Reduce(&duration,&global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

   	if(rank == 0) 
   	{
       printf("Global runtime is %f\n",global);
   	}

	MPI_Finalize();
	return 0;
}
