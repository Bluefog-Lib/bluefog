// A pure C++ mpi win ops to implement simple gossip algorithm.
// Compile and run with
//   mpicxx -o mpi_gossip mpi_gossip.cc -std=c++11 && bfrun -np 2 mpi_gossip
#include <mpi.h>
#include <stdio.h>

#include <cassert>
#include <thread>

void AvgTwo(void* invec, void* inoutvec, int* len,
                    MPI_Datatype* datatype) {
  if (*datatype == MPI_DOUBLE) {
    double* invec_d = (double*)invec;
    double* inoutvec_d = (double*)inoutvec;
    for (int i = 0; i < *len; i++) {
      inoutvec_d[i] += invec_d[i];
      inoutvec_d[i] /= 2.0;
    }
  } else if (*datatype == MPI_FLOAT) {
    float* invec_d = (float*)invec;
    float* inoutvec_d = (float*)inoutvec;
    for (int i = 0; i < *len; i++) {
      inoutvec_d[i] += invec_d[i];
      inoutvec_d[i] /= 2.0;
    }
  }
}

int main(int argc, char** argv) {
  int rank, nproc;
  int mpi_threads_provided;
  MPI_Init_thread(&argc, &argv, 1, &mpi_threads_provided);
  // MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  assert(nproc % 2 == 0);

  const int win_size = 10;
  double array[win_size];
  double recv_array[win_size];
  for (int i = 0; i < win_size; i++) {
    array[i] = rank;
    recv_array[i] = 0.0;
  }

  MPI_Op MPI_AVG_TWO;
  MPI_Op_create((MPI_User_function*)AvgTwo, true, &MPI_AVG_TWO);

  MPI_Win win;
  MPI_Win_create(array, win_size * sizeof(double), sizeof(double),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &win);
  for (int i = 0; i < 100; i++) {
    if (rank % 2 == 1) continue;  // passive worker
    int target_rank = (rank + 1 + 2*i) % nproc;
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_rank, MPI_MODE_NOCHECK, win);
    MPI_Fetch_and_op(array, recv_array, MPI_DOUBLE, target_rank, 0, MPI_AVG_TWO,
                     win);
    auto temp_type = MPI_DOUBLE;
    AvgTwo(recv_array, array, (int*)&win_size, &temp_type);
    MPI_Win_unlock(target_rank, win);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  printf("Rank %d: %f, %f\t", rank, array[0], recv_array[0]);

  printf("Done\n");
  MPI_Win_free(&win);

  MPI_Finalize();
  return 0;
}