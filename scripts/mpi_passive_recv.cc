// A pure C++ mpi to implement passive receive.
// Compile and run with
//   mpicxx -o mpi_passive_recv mpi_passive_recv.cc -std=c++11 && bfrun -np 4 mpi_passive_recv

#include <mpi.h>
#include <stdio.h>

#include <atomic>
#include <cassert>
#include <memory>
#include <queue>
#include <thread>
#include <vector>

const int kSendLength = 20;
const int kTag = 1234567;

#define MPICHECK(cmd)                                                  \
  do {                                                                 \
    int e = cmd;                                                       \
    if (e != MPI_SUCCESS) {                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

void active_send(const int self_rank) {
  std::vector<int> sendbuf;
  for (int i = 0; i < kSendLength; i++) {
    sendbuf.push_back(self_rank * self_rank);
  }
  srand(time(NULL) * self_rank);
  int random_milliseconds = rand() % 600 + 100;
  int send_times = rand() % 5 + 1;
  for (int i = 0; i < send_times; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(random_milliseconds));
    printf("Rank [%d]: Random Sleep %d\n", self_rank, random_milliseconds);
    MPI_Send(sendbuf.data(), kSendLength, MPI_INT, /*dest=*/0, /*tag=*/kTag,
             MPI_COMM_WORLD);
  }
}

void add_irecv_request(
    std::vector<std::pair<std::shared_ptr<MPI_Request>, int>>& request_vec,
    int* recvbuf, int rank) {
  auto request = std::make_shared<MPI_Request>();
  request_vec.push_back(std::make_pair(request, rank));
  MPI_Irecv(recvbuf, kSendLength, MPI_INT, /*src=*/rank, /*tag=*/kTag,
            MPI_COMM_WORLD, request.get());
}

void passive_recv(const int size) {
  MPI_Status mpi_status;
  // MPI_Request* requests = new MPI_Request[size-1];
  std::vector<std::pair<std::shared_ptr<MPI_Request>, int>> request_vec;
  std::vector<int*> recv_buffs;

  for (int i = 1; i < size; i++) {
    int* recvbuf = new int[kSendLength];
    recv_buffs.push_back(recvbuf);
    add_irecv_request(request_vec, recvbuf, i);
  }

  while (request_vec.size() > 0) {
    // printf("Waiting....");
    for (auto it = request_vec.begin(); it != request_vec.end();) {
      MPI_Status status;
      int flag = 0;
      MPI_Test(it->first.get(), &flag, &status);
      if (flag == 1) {
        int k = it->second - 1;
        printf("\nrecv from %d: %d\n", k + 1, recv_buffs[k][0]);
        // printf("Status, %d, %d, %d\n", status.MPI_ERROR, status.MPI_SOURCE,
        // status.MPI_TAG);
        it = request_vec.erase(it);
        add_irecv_request(request_vec, recv_buffs[k], k + 1);
      } else {
        ++it;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void any_send(const int self_rank) {
  std::vector<int> sendbuf;
  for (int i = 0; i < kSendLength; i++) {
    sendbuf.push_back(self_rank * self_rank);
  }
  srand(time(NULL) * self_rank);
  int random_milliseconds = rand() % 600 + 100;
  int send_times = rand() % 5 + 1;
  for (int i = 0; i < send_times; i++) {
    int tag = rand() % 1000000;
    int pre_info[2] = {kSendLength, tag};
    MPI_Send(pre_info, 2, MPI_INT, 0, kTag, MPI_COMM_WORLD);

    std::this_thread::sleep_for(std::chrono::milliseconds(random_milliseconds));
    // printf("Rank [%d]: Random Sleep %d\n", self_rank, random_milliseconds);
    MPI_Send(sendbuf.data(), kSendLength, MPI_INT, /*dest=*/0, /*tag=*/tag,
             MPI_COMM_WORLD);
  }
  int pre_info[3] = {0, 0};  // 0 length means the end.
  MPI_Send(pre_info, 2, MPI_INT, 0, kTag, MPI_COMM_WORLD);
}

struct Request {
  int source;
  int length;
  int tag;
};

void any_recv(const int world_size, std::queue<Request>& queue,
              std::atomic_bool& shut_down) {
  int num_stop_needed = world_size - 1;
  while (num_stop_needed != 0) {
    int* buf = new int[2];  // [Length, tag]
    MPI_Status status;
    // receive message from any source
    MPI_Recv(buf, 2, MPI_INT, MPI_ANY_SOURCE, kTag, MPI_COMM_WORLD, &status);

    if (buf[0]) {
      Request req;
      req.source = status.MPI_SOURCE;
      req.tag = buf[1];
      req.length = buf[0];
      // printf("Recv request from %d, length %d, tag: %d\n", req.source,
      //        req.length, req.tag);
      queue.push(req);  // NOT Thread safe!
    } else {
      num_stop_needed--;
      printf("Recv stop request from %d. Now num_stop_needed is %d.\n", status.MPI_SOURCE, num_stop_needed);
    }
  }
  printf("Done of receiving!");
  shut_down = true;
}

void DoRecv(std::queue<Request>& queue, std::atomic_bool& shut_down) {
  while (!shut_down.load()) {
    if (!queue.empty()) {  // NOT Thread safe!
      Request req = queue.front();
      queue.pop();
      int* recvbuf = new int[req.length];
      printf("Start recv request from %d, length %d, tag: %d\n", req.source,
             req.length, req.tag);
      MPICHECK(MPI_Recv(recvbuf, req.length, MPI_INT, req.source, req.tag,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      printf("\nrecv from %d: %d\n", req.source, recvbuf[0]);
      // delete[] recvbuf;
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }
}

void sendrecv_gossip(const int self_rank, const int target_rank) {
  std::vector<int> sendbuf;
  for (int i = 0; i < kSendLength; i++) {
    sendbuf.push_back(self_rank * self_rank);
  }
  std::vector<int> recvbuf(kSendLength);
  MPI_Sendrecv(sendbuf.data(), kSendLength, MPI_INT, target_rank, /*tag=*/0,
               recvbuf.data(), kSendLength, MPI_INT, target_rank, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  printf("Rank [%d]: recv from %d\n", self_rank, recvbuf[1]);
}

int main(int argc, char** argv) {
  int rank, nproc;
  int mpi_threads_provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_threads_provided);
  assert(mpi_threads_provided > 2);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  assert(nproc >= 2);
  // --------- Pair Gossip Example ----------
  // assert( (nproc % 2) == 0);
  // if ( (rank % 2) == 0) {
  //   sendrecv_gossip(rank, rank+1);
  // } else {
  //   sendrecv_gossip(rank, rank-1);
  // }

  // --------- Simple Active Passive??? ----------
  // if (rank != 0 ){
  //   active_send(rank);
  // } else {
  //   passive_recv(nproc);
  // }

  if (rank != 0) {
    any_send(rank);
  } else {
    std::atomic_bool shut_down(false);
    std::queue<Request> queue;
    std::thread communication_thread(DoRecv, std::ref(queue),
                                     std::ref(shut_down));
    std::thread passive_recv_thread(any_recv, nproc, std::ref(queue), std::ref(shut_down));
    passive_recv_thread.detach();
    communication_thread.join();
  }

  MPI_Finalize();
  return 0;
}