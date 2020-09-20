// A pure C++ mpi win ops used to test the environment.
// Compile and run with 
//   mpicxx -o mpi_win_ops mpi_win_ops.cc -std=c++11 && bfrun -np 4 mpi_win_ops
#include <mpi.h>
#include <stdio.h>

#include <cassert>
#include <thread>

const int max_sent = 10022;

void MPI_Win_ops(int rank, int nproc, int* array, int win_size, MPI_Win win) {
  if (rank == 0) {
    for (int target_rank = 1; target_rank < nproc; target_rank++) {
      MPI_Win_lock(MPI_LOCK_SHARED, target_rank, MPI_MODE_NOCHECK, win);
      MPI_Put(array, win_size, MPI_INT, target_rank, 0, win_size, MPI_INT, win);
      MPI_Win_unlock(target_rank, win);
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

  const int win_size = 23*23*203;
  double array[win_size];
  for (int i = 0; i < win_size; i++) {
    if (rank == 0) {
      array[i] = 1234.0;
    } else {
      array[i] = rank;
    }
  }
  MPI_Win win;
  MPI_Win_create(array, win_size * sizeof(double), sizeof(double), MPI_INFO_NULL,
                 MPI_COMM_WORLD, &win);
  if (rank == 0) {
    for (int target_rank = 1; target_rank < nproc; target_rank++) {
      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_rank, MPI_MODE_NOCHECK, win);
      int bias = 0;
      int sent_size = std::min(max_sent, win_size - bias);
      while (sent_size != 0) {
        MPI_Put(array + bias, sent_size, MPI_DOUBLE, target_rank, bias, sent_size,
                MPI_DOUBLE, win);
        bias += sent_size;
        sent_size = std::min(max_sent, win_size - bias);
      }
      MPI_Win_unlock(target_rank, win);
    }
  }

  // std::thread communication_thread(MPI_Win_ops, rank, nproc, array, win_size,
  //                                  win);
  // communication_thread.join();

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != 0) {
    assert(array[0] == 1234.0);
    assert(array[win_size - 1] == 1234.0);
  }

  printf("Done\n");
  MPI_Win_free(&win);

  MPI_Finalize();
  return 0;
}