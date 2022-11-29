#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>


#define min(x, y) (((x) < (y))? (x) : (y))
void fill_edgeToRow_cuda(int* edgeToRow, int *nodePointer, int num_nodes);
void fill_window_cuda(int* edgeToColumn, int* blockPartition, int* nodePointer,
                      int* edgeList, int blockSize_h, int blockSize_w, int num_nodes);

std::vector<torch::Tensor> spmm_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input
  );

std::vector<torch::Tensor> spmm_balance_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    torch::Tensor tcblock_rowid,
	  torch::Tensor tcblock_colid,
              int tc_count,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input
  );

std::vector<torch::Tensor> spmm_balance_forward_cuda_v1(
	torch::Tensor tcblock_rowid,
	torch::Tensor tcblock_colid,
	torch::Tensor TCblocktile_rowid,
	torch::Tensor TCblocktile_colid,
	torch::Tensor TCblock_offset,
	torch::Tensor sparse_AToX_idx,
	          int tc_count,
              int num_nodes,
              int num_edges,
              int embedding_dim,
  torch::Tensor input
); 

torch::Tensor spmm_cuda_coo_without_v(
	torch::Tensor rowind,
	torch::Tensor colind,
	torch::Tensor dense,
	int m
);



std::vector<torch::Tensor> spmmAGNN_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor edgeAttention,        // *edge attention [n_head, n_e]
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input
    ); 

std::vector<torch::Tensor> sddmm_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,			    // edge list.
    torch::Tensor blockPartition,		// number of TC_blocks (16x8) in each row_window.
    torch::Tensor edgeToColumn, 		// eid -> col within each row_window.
    torch::Tensor edgeToRow, 			  // eid -> col within each row_window.
              int num_nodes,
              int num_edges,
              int embedding_dim,	    // embedding dimension.
	  torch::Tensor input				        // input feature matrix.
); 

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//////////////////////////////////////////
//
// SPMM Foward Pass (GCN, GraphSAGE)
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_forward(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

  return spmm_forward_cuda(nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim,
                            input);
}

//////////////////////////////////////////
//
// SPMM balance Foward Pass (GCN, GraphSAGE)
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_balance_forward(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    torch::Tensor tcblock_rowid,
	  torch::Tensor tcblock_colid,
    int tc_count
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

  return spmm_balance_forward_cuda(nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, tcblock_rowid, tcblock_colid,
                            tc_count, num_nodes, num_edges, embedding_dim,
                            input);
}

std::vector<torch::Tensor> spmm_balance_forward_v1(
    torch::Tensor input,
    torch::Tensor tcblock_rowid,
	  torch::Tensor tcblock_colid,
    torch::Tensor TCblocktile_rowid,
	  torch::Tensor TCblocktile_colid,
    torch::Tensor TCblock_offset,
	  torch::Tensor sparse_AToX_idx,
    int num_nodes
) {
  CHECK_INPUT(input);
  CHECK_INPUT(tcblock_rowid);
  CHECK_INPUT(tcblock_colid);
  CHECK_INPUT(TCblocktile_rowid);
  CHECK_INPUT(TCblocktile_colid);
  CHECK_INPUT(TCblock_offset);
  CHECK_INPUT(sparse_AToX_idx);
  int num_edges = TCblocktile_rowid.size(0);
  int embedding_dim = input.size(1);
  int tc_count = tcblock_rowid.size(0);
  return spmm_balance_forward_cuda_v1(tcblock_rowid, tcblock_colid, 
                            TCblocktile_rowid, TCblocktile_colid, TCblock_offset, sparse_AToX_idx,
                            tc_count, num_nodes, num_edges, embedding_dim,
                            input);
}

////////////////////////////////////////////
//
// SPMM Foward Pass (AGNN, AGNN)
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_forward_AGNN(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor edgeAttention,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(edgeAttention);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);
  
  return spmmAGNN_forward_cuda(nodePointer, edgeList, edgeAttention,
                              blockPartition, edgeToColumn, edgeToRow, 
                              num_nodes, num_edges, embedding_dim,
                              input);
}


////////////////////////////////////////////
//
// SDDMM Foward Pass
//
////////////////////////////////////////////
std::vector<torch::Tensor> sddmm_forward(
  torch::Tensor input,				
  torch::Tensor nodePointer,
  torch::Tensor edgeList,			    
	torch::Tensor blockPartition,		
	torch::Tensor edgeToColumn, 		
	torch::Tensor edgeToRow
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

//   printf("at sddmm_forward\n");
  return sddmm_forward_cuda(nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim,
                            input);
}


// condense an sorted array with duplication: [1,2,2,3,4,5,5]
// after condense, it becomes: [1,2,3,4,5].
// Also, mapping the origin value to the corresponding new location in the new array.
// 1->[0], 2->[1], 3->[2], 4->[3], 5->[4]. 
std::map<unsigned, unsigned> inplace_deduplication(unsigned* array, unsigned length){
    int loc=0, cur=1;
    std::map<unsigned, unsigned> nb2col;
    nb2col[array[0]] = 0;
    while (cur < length){
        if(array[cur] != array[cur - 1]){
            loc++;
            array[loc] = array[cur];
            nb2col[array[cur]] = loc;       // mapping from eid to TC_block column index.[]
        }
        cur++;
    }
    return nb2col;
}

// void preprocess(torch::Tensor edgeList_tensor, 
//                 torch::Tensor nodePointer_tensor, 
//                 int num_nodes, 
//                 int blockSize_h,
//                 int blockSize_w,
//                 torch::Tensor blockPartition_tensor, 
//                 torch::Tensor edgeToColumn_tensor,
//                 torch::Tensor edgeToRow_tensor
//                 ){
//     // input tensors.
//     auto edgeList = edgeList_tensor.accessor<int, 1>();
//     auto nodePointer = nodePointer_tensor.accessor<int, 1>();
//     // output tensors.
//     auto blockPartition = blockPartition_tensor.accessor<int, 1>();
//     auto edgeToColumn = edgeToColumn_tensor.accessor<int, 1>();
//     auto edgeToRow = edgeToRow_tensor.accessor<int, 1>();

//     unsigned block_counter = 0;

//     #pragma omp parallel for 
//     for (unsigned nid = 0; nid < num_nodes; nid++){
//         for (unsigned eid = nodePointer[nid]; eid < nodePointer[nid+1]; eid++)
//             edgeToRow[eid] = nid;
//     }

//     #pragma omp parallel for reduction(+:block_counter)
//     for (unsigned iter = 0; iter < num_nodes + 1; iter +=  blockSize_h){
//         unsigned windowId = iter / blockSize_h;
//         unsigned block_start = nodePointer[iter];
//         unsigned block_end = nodePointer[min(iter + blockSize_h, num_nodes)];
//         unsigned num_window_edges = block_end - block_start;
//         unsigned *neighbor_window = (unsigned *) malloc (num_window_edges * sizeof(unsigned));
//         memcpy(neighbor_window, &edgeList[block_start], num_window_edges * sizeof(unsigned));

//         // Step-1: Sort the neighbor id array of a row window.
//         thrust::sort(neighbor_window, neighbor_window + num_window_edges);

//         // Step-2: Deduplication of the edge id array.
//         // printf("Before dedupblication: %d\n", num_window_edges);
//         // FAN: important! this is a map, not an array! reorder and group to left, the edgeToColumn saves the reordered result. I believe its for sparseA, and need to work together with edgelist(origin col index).
//         // FAN: the origin col index is for loading matrix B. LOAD SPARSE A, needs this edgeToColumn, and nodepointer(rowoffset), blockPartition to work together. 
//         std::map<unsigned, unsigned> clean_edges2col = inplace_deduplication(neighbor_window, num_window_edges);
        
//         // generate blockPartition --> number of TC_blcok in each row window.
//         blockPartition[windowId] = (clean_edges2col.size() + blockSize_w - 1) /blockSize_w;
//         block_counter += blockPartition[windowId];

//         // scan the array and generate edge to column mapping. --> edge_id to compressed_column_id of TC_block.
//         for (unsigned e_index = block_start; e_index < block_end; e_index++){
//             unsigned eid = edgeList[e_index];
//             edgeToColumn[e_index] = clean_edges2col[eid];
//         }
//     }
//     printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter, block_counter * 8 * 16);
// }
int preprocess(torch::Tensor edgeList_tensor, 
                torch::Tensor nodePointer_tensor, 
                int num_nodes, 
                int blockSize_h,
                int blockSize_w,
                torch::Tensor blockPartition_tensor, 
                torch::Tensor edgeToColumn_tensor,
                torch::Tensor edgeToRow_tensor
                ){

    // input tensors.
    auto edgeList = edgeList_tensor.accessor<int, 1>();
    auto nodePointer = nodePointer_tensor.accessor<int, 1>();

    // output tensors.
    auto blockPartition = blockPartition_tensor.accessor<int, 1>();
    auto edgeToColumn = edgeToColumn_tensor.accessor<int, 1>();
    auto edgeToRow = edgeToRow_tensor.accessor<int, 1>();

    unsigned block_counter = 0;

    #pragma omp parallel for 
    for (unsigned nid = 0; nid < num_nodes; nid++){
        for (unsigned eid = nodePointer[nid]; eid < nodePointer[nid+1]; eid++)
            edgeToRow[eid] = nid;
    }

    #pragma omp parallel for reduction(+:block_counter)
    for (unsigned iter = 0; iter < num_nodes + 1; iter +=  blockSize_h){
        unsigned windowId = iter / blockSize_h;
        unsigned block_start = nodePointer[iter];
        unsigned block_end = nodePointer[min(iter + blockSize_h, num_nodes)];
        unsigned num_window_edges = block_end - block_start;
        unsigned *neighbor_window = (unsigned *) malloc (num_window_edges * sizeof(unsigned));
        memcpy(neighbor_window, &edgeList[block_start], num_window_edges * sizeof(unsigned));

        // Step-1: Sort the neighbor id array of a row window.
        thrust::sort(neighbor_window, neighbor_window + num_window_edges);

        // Step-2: Deduplication of the edge id array.
        // printf("Before dedupblication: %d\n", num_window_edges);
        // FAN: important! this is a map, not an array! reorder and group to left, the edgeToColumn saves the reordered result. I believe its for sparseA, and need to work together with edgelist(origin col index).
        // FAN: the origin col index is for loading matrix B. LOAD SPARSE A, needs this edgeToColumn, and nodepointer(rowoffset), blockPartition to work together. 
        std::map<unsigned, unsigned> clean_edges2col = inplace_deduplication(neighbor_window, num_window_edges);
        
        // generate blockPartition --> number of TC_blcok in each row window.
        blockPartition[windowId] = (clean_edges2col.size() + blockSize_w - 1) /blockSize_w;
        block_counter += blockPartition[windowId];

        // scan the array and generate edge to column mapping. --> edge_id to compressed_column_id of TC_block.
        for (unsigned e_index = block_start; e_index < block_end; e_index++){
            unsigned eid = edgeList[e_index];
            edgeToColumn[e_index] = clean_edges2col[eid];
        }
    }
    printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter, block_counter * 8 * 16);
    return block_counter;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int> preprocess_v1(torch::Tensor edgeList_tensor, 
                torch::Tensor nodePointer_tensor, 
                int num_nodes, 
                int blockSize_h,
                int blockSize_w,
                torch::Tensor blockPartition_tensor, 
                torch::Tensor edgeToColumn_tensor,
                torch::Tensor edgeToRow_tensor
                ){
    // input tensors.
    auto edgeList = edgeList_tensor.accessor<int, 1>();
    auto nodePointer = nodePointer_tensor.accessor<int, 1>();
    // output tensors.
    auto blockPartition = blockPartition_tensor.accessor<int, 1>();
    auto edgeToColumn = edgeToColumn_tensor.accessor<int, 1>();
    auto edgeToRow = edgeToRow_tensor.accessor<int, 1>();
    #pragma omp parallel for 
    for (unsigned nid = 0; nid < num_nodes; nid++){
        for (unsigned eid = nodePointer[nid]; eid < nodePointer[nid+1]; eid++)
            edgeToRow[eid] = nid;
    }
    int block_counter = 0;
    #pragma omp parallel for reduction(+:block_counter)
    for (unsigned iter = 0; iter < num_nodes + 1; iter +=  blockSize_h){
        unsigned windowId = iter / blockSize_h;
        unsigned block_start = nodePointer[iter];
        unsigned block_end = nodePointer[min(iter + blockSize_h, num_nodes)];
        unsigned num_window_edges = block_end - block_start;
        unsigned *neighbor_window = (unsigned *) malloc (num_window_edges * sizeof(unsigned));
        memcpy(neighbor_window, &edgeList[block_start], num_window_edges * sizeof(unsigned));
        // Step-1: Sort the neighbor id array of a row window.
        thrust::sort(neighbor_window, neighbor_window + num_window_edges);
        // Step-2: Deduplication of the edge id array.
        // printf("Before dedupblication: %d\n", num_window_edges);
        // FAN: important! this is a map, not an array! reorder and group to left, the edgeToColumn saves the reordered result. I believe its for sparseA, and need to work together with edgelist(origin col index).
        // FAN: the origin col index is for loading matrix B. LOAD SPARSE A, needs this edgeToColumn, and nodepointer(rowoffset), blockPartition to work together. 
        std::map<unsigned, unsigned> clean_edges2col = inplace_deduplication(neighbor_window, num_window_edges);
        // generate blockPartition --> number of TC_blcok in each row window.
        blockPartition[windowId] = (clean_edges2col.size() + blockSize_w - 1) /blockSize_w;
        block_counter += blockPartition[windowId];
        // scan the array and generate edge to column mapping. --> edge_id to compressed_column_id of TC_block.
        for (unsigned e_index = block_start; e_index < block_end; e_index++){
            unsigned eid = edgeList[e_index];
            edgeToColumn[e_index] = clean_edges2col[eid];
        }
    }
    printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter, block_counter * 8 * 16);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto tcblock_rowid_tensor = torch::zeros({block_counter}, options);
    auto tcblock_colid_tensor = torch::zeros({block_counter}, options);
    auto tcblocktile_rowid_tensor = torch::zeros({edgeList_tensor.size(0)}, options);
    auto tcblocktile_colid_tensor = torch::zeros({edgeList_tensor.size(0)}, options);
    auto tcblock_offset_tensor = torch::zeros({block_counter + 1}, options);
    auto sparse_AToX_index_tensor = torch::zeros({block_counter * blockSize_w}, options);
    auto tcblock_rowid = tcblock_rowid_tensor.accessor<int, 1>();
    auto tcblock_colid = tcblock_colid_tensor.accessor<int, 1>();
    auto tcblock_offset = tcblock_offset_tensor.accessor<int, 1>();
    auto sparse_AToX_index = sparse_AToX_index_tensor.accessor<int, 1>();
    auto tcblocktile_rowid = tcblocktile_rowid_tensor.accessor<int, 1>();
    auto tcblocktile_colid = tcblocktile_colid_tensor.accessor<int, 1>();
    int sum = 0;
    for (int iter = 0; iter < num_nodes + 1; iter +=  blockSize_h) {
      int windowId = iter / blockSize_h;
      for (int i = 0; i < blockPartition[windowId]; i++) {
        tcblock_rowid[sum + i] = windowId;
        tcblock_colid[sum + i] = i;
      }
      sum += blockPartition[windowId];
    }
    int current_tc = 0;
    for (unsigned iter = 0; iter < num_nodes + 1; iter +=  blockSize_h) {
        unsigned windowId = iter / blockSize_h;
        unsigned block_start = nodePointer[iter];
        unsigned block_end = nodePointer[min(iter + blockSize_h, num_nodes)];
        for (int i = 0; i < blockPartition[windowId]; i++) {
          int current_tc_nnz = 0;
          for (unsigned e_index = block_start; e_index < block_end; e_index++) {
            unsigned col = edgeToColumn[e_index];
			      if (i * blockSize_w <= col && col < (i + 1) * blockSize_w) {			// if the edge in the current TC_block frame of column.
				      unsigned row_local = edgeToRow[e_index] % blockSize_h;
				      unsigned col_local = col % blockSize_w;
				      sparse_AToX_index[current_tc * blockSize_w + col_local] = edgeList[e_index];		// record the mapping from sparse_A colId to rowId of dense_X.
              tcblocktile_rowid[tcblock_offset[current_tc] + current_tc_nnz] = row_local;
              tcblocktile_colid[tcblock_offset[current_tc] + current_tc_nnz] = col_local;
              current_tc_nnz += 1;
			      }	
          }
          current_tc += 1;
          tcblock_offset[current_tc] = tcblock_offset[current_tc - 1] + current_tc_nnz;
        }
    }
    return std::make_tuple(tcblock_rowid_tensor, tcblock_colid_tensor, tcblocktile_rowid_tensor, tcblocktile_colid_tensor, tcblock_offset_tensor, sparse_AToX_index_tensor, block_counter);
}

// FAN ADD: 
void get_coo_tcblock(int num_nodes, 
                     int blockSize_h,
                     int blockSize_w,
                     torch::Tensor blockPartition_tensor, 
                     torch::Tensor tcblock_rowid_tensor,
                     torch::Tensor tcblock_colid_tensor) {
  auto blockPartition = blockPartition_tensor.accessor<int, 1>();
  auto tcblock_rowid = tcblock_rowid_tensor.accessor<int, 1>();
  auto tcblock_colid = tcblock_colid_tensor.accessor<int, 1>();
  int sum = 0;
  for (int iter = 0; iter < num_nodes + 1; iter +=  blockSize_h) {
    int windowId = iter / blockSize_h;
    for (int i = 0; i < blockPartition[windowId]; i++) {
      tcblock_rowid[sum + i] = windowId;
      tcblock_colid[sum + i] = i;
    }
    sum += blockPartition[windowId];
  }
}

void preprocess_gpu(torch::Tensor edgeList_tensor, 
                torch::Tensor nodePointer_tensor, 
                int num_nodes, 
                int blockSize_h,
                int blockSize_w,
                torch::Tensor blockPartition_tensor, 
                torch::Tensor edgeToColumn_tensor,
                torch::Tensor edgeToRow_tensor
                )
{

    // input tensors.
    auto edgeList = edgeList_tensor.data<int>();
    auto nodePointer = nodePointer_tensor.data<int>();

    // output tensors.
    auto blockPartition = blockPartition_tensor.data<int>();
    auto edgeToColumn = edgeToColumn_tensor.data<int>();
    auto edgeToRow = edgeToRow_tensor.data<int>();

    unsigned block_counter = 0;
    
    fill_edgeToRow_cuda(edgeToRow, nodePointer, num_nodes);
    fill_window_cuda(edgeToColumn, blockPartition, nodePointer, edgeList,
                                blockSize_h, blockSize_w, num_nodes);

    printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter, block_counter * 8 * 16);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess", &preprocess, "Preprocess Step (CPU)");
  m.def("preprocess_v1", &preprocess_v1, "Preprocess v1 Step (CPU)");
  m.def("preprocess_gpu", &preprocess_gpu, "Preprocess Step (CUDA)");
  // ADD FAN:
  m.def("get_coo_tcblock", &get_coo_tcblock, "Get COO format for TC blocks");

  // forward computation
  m.def("forward", &spmm_forward, "TC-GNN SPMM forward (CUDA)");
  m.def("balance_forward", &spmm_balance_forward, "TC-GNN balance SPMM forward (CUDA)");
  m.def("balance_forward_v1", &spmm_balance_forward_v1, "TC-GNN balance SPMM v1 forward (CUDA)");
  m.def("forward_ef", &sddmm_forward, "TC-GNN SDDMM forward (CUDA)");
  m.def("forward_AGNN", &spmm_forward_AGNN, "TC-GNN SPMM (AGNN) forward (CUDA)");

  // backward
  m.def("backward", &spmm_forward, "TC-GNN SPMM backward (CUDA)");
  m.def("balance_backward", &spmm_balance_forward, "TC-GNN balance SPMM backward (CUDA)");
  m.def("balance_backward_v1", &spmm_balance_forward_v1, "TC-GNN balance SPMM v1 backward (CUDA)");
  m.def("backward_ef", &sddmm_forward, "TC-GNN SDDMM backward_ef (CUDA)");
}