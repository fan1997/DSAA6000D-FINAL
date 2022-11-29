#include <torch/extension.h>
#include <stdio.h>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>

#include "config.h"
#define WPB 8

using namespace nvcuda;

////////////////////
// each warp fill neighbor for one nodes
// warp -- all neighbors of a node
////////////////////
__global__ void fill_edgeToRow(int* edgeToRow, int *nodePointer, int num_nodes) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int nid = tid / 32;
	int laneid = tid % 32;
	// check a valid node range.
	if (nid < num_nodes){
		#pragma unroll
		for (int eid = nodePointer[nid] + laneid; eid < nodePointer[nid+1]; eid += 32){
            edgeToRow[eid] = nid;
		}
	}
}

__global__ void fill_window (int* edgeToColumn, 
						     int* blockPartition, 
							 int* nodePointer,
							 int* edgeList, 
							 int blockSize_h,
							 int blockSize_w,
							 int num_nodes
							)
{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;

		int winId = tid / 32;		// each warp one window
		int lanid = tid % 32;

        unsigned block_start = nodePointer[winId * blockSize_h];
        unsigned block_end = nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
        unsigned num_window_edges = block_end - block_start;
    //     unsigned *neighbor_window = (unsigned *) malloc (num_window_edges * sizeof(unsigned));
    //     memcpy(neighbor_window, &edgeList[block_start], num_window_edges * sizeof(unsigned));

		// thrust::device_vector<int>neighbor_window(&edgeList[block_start], &edgeList[block_end]);
		// Step-1: Sort the neighbor id array of a row window.
        // thrust::sort(neighbor_window.begin(), neighbor_window.end());

    //     // Step-2: Deduplication of the edge id array.
    //     // printf("Before dedupblication: %d\n", num_window_edges);
    //     std::map<unsigned, unsigned> clean_edges2col = inplace_deduplication(neighbor_window, num_window_edges);

    //     // generate blockPartition --> number of TC_blcok in each row window.
    //     blockPartition[windowId] = (clean_edges2col.size() + blockSize_w - 1) /blockSize_w;
    //     block_counter += blockPartition[windowId];

    //     // scan the array and generate edge to column mapping. --> edge_id to compressed_column_id of TC_block.
    //     for (unsigned e_index = block_start; e_index < block_end; e_index++){
    //         unsigned eid = edgeList[e_index];
    //         edgeToColumn[e_index] = clean_edges2col[eid];
    //     }			
}

void fill_edgeToRow_cuda(int* edgeToRow, int *nodePointer, int num_nodes){
	int wrap_size = 32; 
	int block_size = 1024;
	int grid_size =  (num_nodes * wrap_size + block_size - 1) / block_size;
	// each warp fill neighbor for one nodes
	fill_edgeToRow <<< grid_size, block_size >>>(edgeToRow, nodePointer, num_nodes);
	
	cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

void fill_window_cuda(	int* edgeToColumn, 
					  	int* blockPartition, 
					  	int* nodePointer,
						int* edgeList, 
					   	int blockSize_h,
						int blockSize_w,
						int num_nodes)
{
	int wrap_size = 32; 
	int block_size = 1024;
	int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
	int grid_size =  (window_count * wrap_size + block_size - 1) / block_size;
	// each warp will handle all neighbor for a TC block window (16 x 8)
	fill_window <<< grid_size, block_size >>> (edgeToColumn, blockPartition, nodePointer, edgeList,
																blockSize_h, blockSize_w, num_nodes);
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
}

//////////////////////
/// SPMM forward (GCN, GraphSAGE)
//////////////////////
__global__ void spmm_forward_cuda_kernel(
	const int * __restrict__ nodePointer,		// node pointer.
	const int *__restrict__ edgeList,			// edge list.
	const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ in_mat,		    // input feature matrix.
	float *out_mat							    // aggreAGNNed output feature matrix.
);

//////////////////////
/// SPMM balance forward (GCN, GraphSAGE)
//////////////////////
__global__ void spmm_balance_forward_cuda_kernel(
	const int * __restrict__ nodePointer,		// node pointer.
	const int *__restrict__ edgeList,			// edge list.
	const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
	const int *__restrict__ TCblock_rowid, 		// rowid of each TC block.
	const int *__restrict__ TCblock_colid, 		// colid of each TC block.
	const int tc_count,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // aggreAGNNed output feature matrix.
);
__global__ void spmm_balance_forward_cuda_kernel_v1(
	const int *__restrict__ TCblock_rowid, 		// rowid of each TC block.
	const int *__restrict__ TCblock_colid, 		// colid of each TC block.
	const int *__restrict__ TCblocktile_rowid, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblocktile_colid, 		// colid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const int tc_count,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);
__global__ void LBSPMMKernel_novalue(
	int k, int NNZ, int nnz_per_warp, const int* __restrict__ A_rowind, const int* __restrict__ A_colind, const  float* __restrict__ A_value, const  float* __restrict__ B,  float* __restrict__ C 
  );
  __global__ void LBSPMMKernel_mix(
	int m, int k, int NNZ, int tc_count, int nnz_per_warp, 
	const int * __restrict__ A_rowind, 
	const int* __restrict__ A_colind,
	const int *__restrict__ TCblock_rowid,
	const int *__restrict__ TCblock_colid,
	const int *__restrict__ TCblocktile_rowid,
	const int *__restrict__ TCblocktile_colid,
	const int *__restrict__ TCblock_offset,
	const int *__restrict__ sparse_AToX_idx, 
	const  float* __restrict__ B,  
	float* __restrict__ C);

__global__ void LBSPMMKernel_CF2_novalue(
	int k, int NNZ, int nnz_per_warp, const int* __restrict__ A_rowind, const int* __restrict__ A_colind, const  float* __restrict__ A_value, const  float* __restrict__ B,  float* __restrict__ C 
  ) ;

//////////////////////
/// SPMM forward (AGNN, AGNN)
//////////////////////
__global__ void spmmAGNN_forward_cuda_kernel(
	const int * __restrict__ nodePointer,		// node pointer.
	const int *__restrict__ edgeList,			// edge list.
    const float *__restrict__ edgeAttention,	// edge attention.
	const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // aggreAGNNed output feature matrix.
);

//////////////////////
/// SDDMM forward
//////////////////////
__global__ void sddmm_forward_cuda_kernel(
    const int * __restrict__ nodePointer,		// node pointer.
	const int *__restrict__ edgeList,			// edge list.
	const int *__restrict__ blockPartition,		// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 			// eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *__restrict__ in_mat,					// input feature matrix.
	float *edgeFeature							// aggreAGNNed output feature matrix.
);

////////////////////////////////////////////
//
// SPMM Foward Pass  (GCN, GraphSAGE)
//
////////////////////////////////////////////
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
) 
{
    auto output = torch::zeros_like(input);
    const int num_row_windows = blockPartition.size(0);
    const int WARPperBlock = WPB;

    dim3 grid(num_row_windows, 1, 1);
    dim3 block(WARP_SIZE, WARPperBlock, 1);

    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
	const int dynamic_shared_size = 2 * dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.

    spmm_forward_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
                                                                    nodePointer.data<int>(), 
                                                                    edgeList.data<int>(),
                                                                    blockPartition.data<int>(), 
                                                                    edgeToColumn.data<int>(), 
                                                                    edgeToRow.data<int>(), 
                                                                    num_nodes,
                                                                    num_edges,
                                                                    embedding_dim,
                                                                    input.data<float>(), 
                                                                    output.data<float>()
                                                                );

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return {output};
}


////////////////////////////////////////////
//
// SPMM balance Foward Pass  (GCN, GraphSAGE)
//
////////////////////////////////////////////
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
) 
{
    auto output = torch::zeros_like(input);
    const int num_row_windows = blockPartition.size(0);
    const int WARPperBlock = WPB;
    std::cout << "TCBLOCK_PER_WARP:" << TCBLOCK_PER_WARP << std::endl;
    dim3 grid((tc_count + TCBLOCK_PER_WARP - 1) / TCBLOCK_PER_WARP, 1, 1);
    dim3 block(WARP_SIZE, WARPperBlock, 1);

    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
	const int dynamic_shared_size = 2 * dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.

    spmm_balance_forward_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
                                                                    nodePointer.data<int>(), 
                                                                    edgeList.data<int>(),
                                                                    blockPartition.data<int>(), 
                                                                    edgeToColumn.data<int>(), 
                                                                    edgeToRow.data<int>(), 
																	tcblock_rowid.data<int>(), 
																	tcblock_colid.data<int>(), 
																	tc_count, 
                                                                    num_nodes,
                                                                    num_edges,
                                                                    embedding_dim,
                                                                    input.data<float>(), 
                                                                    output.data<float>()
                                                                );
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return {output};
}

////////////////////////////////////////////
//
// SPMM balance Foward V1 Pass  (GCN, GraphSAGE)
//
////////////////////////////////////////////
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
) 
{
	auto output = torch::zeros_like(input);
    const int WARPperBlock = WPB;
    dim3 grid((tc_count + TCBLOCK_PER_WARP - 1) / TCBLOCK_PER_WARP, 1, 1);
	std::cout << "TCBLOCK_PER_WARP:" << TCBLOCK_PER_WARP << std::endl;
    dim3 block(WARP_SIZE, WARPperBlock, 1);
    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
	const int dynamic_shared_size = 2 * dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.
    spmm_balance_forward_cuda_kernel_v1<<<grid, block, dynamic_shared_size>>>(
                                                                    tcblock_rowid.data<int>(), 
                                                                    tcblock_colid.data<int>(),
                                                                    TCblocktile_rowid.data<int>(), 
                                                                    TCblocktile_colid.data<int>(), 
                                                                    TCblock_offset.data<int>(), 
																	sparse_AToX_idx.data<int>(), 
																	tc_count, 
                                                                    num_nodes,
                                                                    num_edges,
                                                                    embedding_dim,
                                                                    input.data<float>(), 
                                                                    output.data<float>()
                                                                );
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return {output};
}


////////////////////////////////////////////
//
// SPMM Foward Pass (AGNN, AGNN)
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmmAGNN_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor edgeAttention,        //*edge attention [n_head, n_e]
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input
) 
{
    auto output = torch::zeros_like(input);
    const int num_row_windows = blockPartition.size(0);
    const int WARPperBlock = WPB;
    const int num_attention = edgeAttention.size(0);
 
    cudaStream_t* streams = new cudaStream_t[num_attention];
    dim3 grid(num_row_windows, 1, 1);
    dim3 block(WARP_SIZE, WARPperBlock, 1);

    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
	const int dynamic_shared_size = dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.

    // printf("spmmAGNN_forward_cuda--1\n");
    for (int att_idx = 0; att_idx < num_attention; att_idx++){
        cudaStreamCreate ( &streams[att_idx]) ;
        spmmAGNN_forward_cuda_kernel<<<grid, block, dynamic_shared_size, streams[att_idx]>>>(
                                                                                        nodePointer.data<int>(), 
                                                                                        edgeList.data<int>(),
                                                                                        edgeAttention.data<float>(),
                                                                                        blockPartition.data<int>(), 
                                                                                        edgeToColumn.data<int>(), 
                                                                                        edgeToRow.data<int>(), 
                                                                                        num_nodes,
                                                                                        num_edges,
                                                                                        embedding_dim,
                                                                                        input.data<float>(), 
                                                                                        output.data<float>()
                                                                                    );
    }
    // printf("spmmAGNN_forward_cuda--2\n");
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return {output};
}

////////////////////////////////////////////
//
// SDDMM Foward Pass 
//
////////////////////////////////////////////
std::vector<torch::Tensor> sddmm_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,			    // edge list.
	torch::Tensor blockPartition,		// number of TC_blocks (16x8) in each row_window.
	torch::Tensor edgeToColumn, 		// eid -> col within each row_window.
	torch::Tensor edgeToRow, 			// eid -> col within each row_window.
              int num_nodes,
              int num_edges,
              int embedding_dim,	    // embedding dimension.
	torch::Tensor input				    // input feature matrix.
) 
{
    auto output = torch::zeros_like(edgeList).to(torch::kFloat);
    const int num_row_windows = blockPartition.size(0);

	dim3 grid(num_row_windows, 1, 1);
	dim3 block(WARP_SIZE, 1, 1);
    // printf("at sddmm_forward_cuda\n");

    sddmm_forward_cuda_kernel<<< grid, block>>>(
                                                nodePointer.data<int>(), 
                                                edgeList.data<int>(),
                                                blockPartition.data<int>(), 
                                                edgeToColumn.data<int>(), 
                                                edgeToRow.data<int>(), 
                                                num_nodes,
                                                num_edges,
                                                embedding_dim,
                                                input.data<float>(), 
                                                output.data<float>()
                                                );

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return {output};
}



/*----------- CUDA Kernel ---------*/ 

////////////////////////////////////
/// SPMM forward (GCN, GraphSage)
///////////////////////////////////
__global__ void spmm_forward_cuda_kernel(
	const int * __restrict__ nodePointer,		// node pointer.
	const int *__restrict__ edgeList,			// edge list.
	const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // aggreAGNNed output feature matrix.
) {
    const unsigned bid = blockIdx.x;								// block_index == row_window_index
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
	const unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.
	
	const unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
	const unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
	const unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
	const unsigned dense_bound = numNodes * embedding_dim;

	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	// __shared__ float dense_X[dimTileNum * BLK_W * BLK_H];	// column-major dense tile [dimTileNum, BLK_W, BLK_H]
	extern __shared__ float dense_X[];

	wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
	wmma::fill_fragment(acc_frag, 0.0f);

	// Processing TC_blocks along the column dimension of Sparse A.
	for (unsigned i = 0; i < num_TC_blocks; i++){

		// Init A_colToX_row with dummy values.
		if (tid < BLK_W){
			sparse_AToX_index[tid] = numNodes + 1;
		}

		__syncthreads();

		// Init sparse_A with zero values.
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock){
			sparse_A[idx] = 0;
		}

		// Init dense_X with zero values.
		#pragma unroll
		for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock){
			dense_X[idx] = 0;
		}

		// Initialize sparse_A by using BLK_H (16) threads from the warp-0.
		// currently fetch all neighbors of the current nodes.
		// then to see whether it can fit into current TC_block frame of column.		

		// FAN: important to understand. all threads check all data to check if they are in current tc block
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock){
			unsigned col = edgeToColumn[eIdx];
			if (i * BLK_W <= col && col < (i + 1) * BLK_W){			// if the edge in the current TC_block frame of column.
				unsigned row_local = edgeToRow[eIdx] % BLK_H;
				unsigned col_local = col % BLK_W;
				sparse_A[row_local * BLK_W + col_local] = 1;		// set the edge of the sparse_A.
				sparse_AToX_index[col_local] = edgeList[eIdx];		// record the mapping from sparse_A colId to rowId of dense_X.
			}		
		}

		__syncthreads();

		// Initialize dense_X by column-major store,
		// Threads of a warp for fetching a dense_X.
		// each warp identify by wid.
		if (wid < dimTileNum)
			#pragma unroll
			for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize){
				unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W];						// TC_block_col to dense_tile_row.
				unsigned dense_dimIdx = idx / BLK_W;										// dimIndex of the dense tile.
				unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
				unsigned target_idx = wid * BLK_W * BLK_H + idx;
				// boundary test.
				if (source_idx >= dense_bound)
					dense_X[target_idx] = 0;
				else
					dense_X[target_idx] = input[source_idx];
			}

		__syncthreads();

		if (wid < dimTileNum)
		{
			wmma::load_matrix_sync(a_frag, sparse_A, BLK_W);
			wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);     // [dimTileNum*BLK_H, BLK_W]

			#pragma unroll
			for (unsigned t = 0; t < a_frag.num_elements; t++) {
				a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
			}

			#pragma unroll
			for (unsigned t = 0; t < b_frag.num_elements; t++) {
				b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
			}
			// Perform the matrix multiplication.
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
		}
	}

	if (wid < dimTileNum)
		// Store the matrix to output matrix.
		// * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
		wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
}

////////////////////////////////////
/// FAN ADD SPMM balance forward (GCN, GraphSage)
///////////////////////////////////
__global__ void spmm_balance_forward_cuda_kernel(
	const int * __restrict__ nodePointer,		// node pointer.
	const int *__restrict__ edgeList,			// edge list.
	const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
	const int *__restrict__ TCblock_rowid, 		// rowid of each TC block.
	const int *__restrict__ TCblock_colid, 		// colid of each TC block.
	const int tc_count,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
	__shared__ int tc_rowid[TCBLOCK_PER_WARP];
	__shared__ int tc_colid[TCBLOCK_PER_WARP];
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = bid * TCBLOCK_PER_WARP;
	const unsigned hb = min((bid + 1) * TCBLOCK_PER_WARP, tc_count);
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	#pragma unroll
	for (unsigned idx = tid; idx < TCBLOCK_PER_WARP; idx += threadPerBlock) {
	  int ptr = lb + idx;
	  if (ptr < hb) {
		tc_rowid[idx] = __ldg(TCblock_rowid + ptr);
		tc_colid[idx] = __ldg(TCblock_colid + ptr);
	  }
	}
	__syncthreads();
	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	// __shared__ float dense_X[dimTileNum * BLK_W * BLK_H];	// column-major dense tile [dimTileNum, BLK_W, BLK_H]
	extern __shared__ float dense_X[];
	wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
	wmma::fill_fragment(acc_frag, 0.0f);
	int former_row_id = tc_rowid[0];
	int current_rid = former_row_id;
	int current_cid = 0;
	unsigned nIdx_start = current_rid * BLK_H;					    // starting nodeIdx of current row_window.
	unsigned nIdx_end = min((current_rid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.
	unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
	unsigned eIdx_end = nodePointer[nIdx_end];	            // ending edgeIdx of the current row_window.
	// for (unsigned i = 0; i < TCBLOCK_PER_WARP; i++) {
	for (unsigned i = 0; i < hb - lb; i++) {
	  current_rid = tc_rowid[i];
	  current_cid = tc_colid[i];
	  if (current_rid != former_row_id) {
		nIdx_start = current_rid * BLK_H;					    // starting nodeIdx of current row_window.
		nIdx_end = min((current_rid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.
		eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
		eIdx_end = nodePointer[nIdx_end];	            // ending edgeIdx of the current row_window.
		if (wid < dimTileNum) {
			wmma::store_matrix_sync(dense_X + wid * BLK_H * BLK_H, acc_frag, BLK_H, wmma::mem_row_major);
			#pragma unroll
			for (unsigned idx = laneid; idx < BLK_H * BLK_H; idx += warpSize) {
			  unsigned dense_rowIdx = former_row_id * BLK_H + idx / BLK_H;
			  unsigned dense_dimIdx = idx % BLK_H;
			  unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
			  unsigned target_idx = wid * BLK_H * BLK_H + idx;
			  // boundary test.
			  if (source_idx < dense_bound)
				atomicAdd(output + source_idx, dense_X[target_idx]);
			}   
		}
		wmma::fill_fragment(acc_frag, 0.0f);
		former_row_id = current_rid;
	  } 
      // compute this tile 
	  // Init A_colToX_row with dummy values.
	  if (tid < BLK_W){
		sparse_AToX_index[tid] = numNodes + 1;
	  }
	  __syncthreads();
	  // Init sparse_A with zero values.
	  #pragma unroll
	  for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock){
		sparse_A[idx] = 0;
	  }
	  // Init dense_X with zero values.
	  #pragma unroll
	  for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock){
		dense_X[idx] = 0;
	  }
	  // Initialize sparse_A by using BLK_H (16) threads from the warp-0.
	  // currently fetch all neighbors of the current nodes.
	  // then to see whether it can fit into current TC_block frame of column.		
	  #pragma unroll
	  for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock){
		unsigned col = edgeToColumn[eIdx];
		if (current_cid * BLK_W <= col && col < (current_cid + 1) * BLK_W){			// if the edge in the current TC_block frame of column.
		  unsigned row_local = edgeToRow[eIdx] % BLK_H;
		  unsigned col_local = col % BLK_W;
		  sparse_A[row_local * BLK_W + col_local] = 1;		// set the edge of the sparse_A.
		  sparse_AToX_index[col_local] = edgeList[eIdx];		// record the mapping from sparse_A colId to rowId of dense_X.
		}		
	  }
	  __syncthreads();
	  // Initialize dense_X by column-major store,
	  // Threads of a warp for fetching a dense_X.
	  // each warp identify by wid.
	  if (wid < dimTileNum)
		#pragma unroll
		for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize){
		  unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W];						// TC_block_col to dense_tile_row.
		  unsigned dense_dimIdx = idx / BLK_W;										// dimIndex of the dense tile.
		  unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
		  unsigned target_idx = wid * BLK_W * BLK_H + idx;
		  // boundary test.
		  if (source_idx >= dense_bound)
			dense_X[target_idx] = 0;
		  else
			dense_X[target_idx] = input[source_idx];
		}
	  __syncthreads();
	  if (wid < dimTileNum) {
			wmma::load_matrix_sync(a_frag, sparse_A, BLK_W);
			wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);
			#pragma unroll
			for (unsigned t = 0; t < a_frag.num_elements; t++) {
				a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
			}
			#pragma unroll
			for (unsigned t = 0; t < b_frag.num_elements; t++) {
				b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
			}
			// Perform the matrix multiplication.
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	  }
	}
	if (wid < dimTileNum) {
		wmma::store_matrix_sync(dense_X + wid * BLK_H * BLK_H, acc_frag, BLK_H, wmma::mem_row_major);
		#pragma unroll
		for (unsigned idx = laneid; idx < BLK_H * BLK_H; idx += warpSize) {
		  unsigned dense_rowIdx = current_cid * BLK_H + idx / BLK_H;
		  unsigned dense_dimIdx = idx % BLK_H;
		  unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
		  unsigned target_idx = wid * BLK_H * BLK_H + idx;
		  // boundary test.
		  if (source_idx < dense_bound)
			atomicAdd(output + source_idx, dense_X[target_idx]);
		}   
	}
}



__global__ void spmm_balance_forward_cuda_kernel_v1(
	const int *__restrict__ TCblock_rowid, 		// rowid of each TC block.
	const int *__restrict__ TCblock_colid, 		// colid of each TC block.
	const int *__restrict__ TCblocktile_rowid, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblocktile_colid, 		// colid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const int tc_count,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
	__shared__ int tc_rowid[TCBLOCK_PER_WARP];
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = bid * TCBLOCK_PER_WARP;
	const unsigned hb = min((bid + 1) * TCBLOCK_PER_WARP, tc_count);
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	#pragma unroll
	for (unsigned idx = tid; idx < TCBLOCK_PER_WARP; idx += threadPerBlock) {
	  int ptr = lb + idx;
	  if (ptr < hb) {
		tc_rowid[idx] = __ldg(TCblock_rowid + ptr);
	  }
	}
	__syncthreads();
	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	// __shared__ float dense_X[dimTileNum * BLK_W * BLK_H];	// column-major dense tile [dimTileNum, BLK_W, BLK_H]
	extern __shared__ float dense_X[];
	wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
	wmma::fill_fragment(acc_frag, 0.0f);
	int former_row_id = tc_rowid[0];
	int current_rid = former_row_id;


	for (unsigned i = 0; i < hb - lb; i++) {
	  current_rid = tc_rowid[i];
	  unsigned eIdx_start = TCblock_offset[lb + i];			
	  unsigned eIdx_end = TCblock_offset[lb + 1 + i];
	  unsigned sparse_AToX_idx_start = (lb + i) * BLK_W;	   

	  if (current_rid != former_row_id) {
		if (wid < dimTileNum) {
			wmma::store_matrix_sync(dense_X + wid * BLK_H * BLK_H, acc_frag, BLK_H, wmma::mem_row_major);
			#pragma unroll
			for (unsigned idx = laneid; idx < BLK_H * BLK_H; idx += warpSize) {
			  unsigned dense_rowIdx = former_row_id * BLK_H + idx / BLK_H;
			  unsigned dense_dimIdx = idx % BLK_H;
			  unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
			  unsigned target_idx = wid * BLK_H * BLK_H + idx;
			  // boundary test.
			  if (source_idx < dense_bound)
				atomicAdd(output + source_idx, dense_X[target_idx]);
			}   
		}
		wmma::fill_fragment(acc_frag, 0.0f);
		former_row_id = current_rid;
	  } 
      // compute this tile 
	  // Init A_colToX_row with dummy values.
	  if (tid < BLK_W){
		sparse_AToX_index[tid] = numNodes + 1;
	  }
	  __syncthreads();
	  // Init sparse_A with zero values.
	  #pragma unroll
	  for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock){
		sparse_A[idx] = 0;
	  }
	  // Init dense_X with zero values.
	  #pragma unroll
	  for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock){
		dense_X[idx] = 0;
	  }
	  // Initialize sparse_A by using BLK_H (16) threads from the warp-0.
	  // currently fetch all neighbors of the current nodes.
	  // then to see whether it can fit into current TC_block frame of column.		
	  // FAN: important to understand. all threads check all data to check if they are in current tc block
	  #pragma unroll
	  for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock){
		  unsigned row_local = TCblocktile_rowid[eIdx];
		  unsigned col_local = TCblocktile_colid[eIdx];
		  sparse_A[row_local * BLK_W + col_local] = 1;		// set the edge of the sparse_A.	
	  }
	  #pragma unroll
	  for (unsigned eIdx = sparse_AToX_idx_start + tid; eIdx < sparse_AToX_idx_start + BLK_W; eIdx += threadPerBlock){
		sparse_AToX_index[tid] = sparse_AToX_idx[eIdx];	
	  }
	  __syncthreads();

	  if (wid < dimTileNum)
		#pragma unroll
		for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize){
		  unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W];						// TC_block_col to dense_tile_row.
		  unsigned dense_dimIdx = idx / BLK_W;										// dimIndex of the dense tile.
		  unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
		  unsigned target_idx = wid * BLK_W * BLK_H + idx;
		  // boundary test.
		  if (source_idx >= dense_bound)
			dense_X[target_idx] = 0;
		  else
			dense_X[target_idx] = input[source_idx];
		}
	  __syncthreads();
	  if (wid < dimTileNum) {
			wmma::load_matrix_sync(a_frag, sparse_A, BLK_W);
			wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);
			#pragma unroll
			for (unsigned t = 0; t < a_frag.num_elements; t++) {
				a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
			}
			#pragma unroll
			for (unsigned t = 0; t < b_frag.num_elements; t++) {
				b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
			}
			// Perform the matrix multiplication.
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	  }
	}
	if (wid < dimTileNum) {
		wmma::store_matrix_sync(dense_X + wid * BLK_H * BLK_H, acc_frag, BLK_H, wmma::mem_row_major);
		#pragma unroll
		for (unsigned idx = laneid; idx < BLK_H * BLK_H; idx += warpSize) {
		  unsigned dense_rowIdx = current_rid * BLK_H + idx / BLK_H;
		  unsigned dense_dimIdx = idx % BLK_H;
		  unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
		  unsigned target_idx = wid * BLK_H * BLK_H + idx;
		  // boundary test.
		  if (source_idx < dense_bound)
			atomicAdd(output + source_idx, dense_X[target_idx]);
		}   
	}
}
////////////////////////////////////
/// SPMM forward (AGNN, AGNN)
///////////////////////////////////
__global__ void spmmAGNN_forward_cuda_kernel(
	const int * __restrict__    nodePointer,		// node pointer.
	const int *__restrict__     edgeList,			// edge list.
    const float *__restrict__   edgeAttention,	    // edge attention.
	const int *__restrict__     blockPartition, 	// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__     edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__     edgeToRow, 		    // eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__   input,		    // input feature matrix.
	float *output							    // aggreAGNNed output feature matrix.
) {
    const unsigned bid = blockIdx.x;								// block_index == row_window_index
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
	const unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.
	
	const unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
	const unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
	const unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
	const unsigned dense_bound = numNodes * embedding_dim;

	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	// __shared__ float dense_X[dimTileNum * BLK_W * BLK_H];	// column-major dense tile [dimTileNum, BLK_W, BLK_H]
	extern __shared__ float dense_X[];

	wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
	wmma::fill_fragment(acc_frag, 0.0f);

	// Processing TC_blocks along the column dimension of Sparse A.
	for (unsigned i = 0; i < num_TC_blocks; i++){

		// Init A_colToX_row with dummy values.
		if (tid < BLK_W){
			sparse_AToX_index[tid] = numNodes + 1;
		}

		__syncthreads();

		// Init sparse_A with zero values.
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock){
			sparse_A[idx] = 0;
		}

		// Init dense_X with zero values.
		#pragma unroll
		for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock){
			dense_X[idx] = 0;
		}

		// Initialize sparse_A by using BLK_H (16) threads from the warp-0.
		// currently fetch all neighbors of the current nodes.
		// then to see whether it can fit into current TC_block frame of column.		
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock){
			unsigned col = edgeToColumn[eIdx];
			if (i * BLK_W <= col && col < (i + 1) * BLK_W){			// if the edge in the current TC_block frame of column.
				unsigned row_local = edgeToRow[eIdx] % BLK_H;
				unsigned col_local = col % BLK_W;
				sparse_A[row_local * BLK_W + col_local] = edgeAttention[eIdx];		// sparse_A according to edge_features.
				sparse_AToX_index[col_local] = edgeList[eIdx];		                // record the mapping from sparse_A colId to rowId of dense_X.
			}		
		}

		__syncthreads();

		// Initialize dense_X by column-major store,
		// Threads of a warp for fetching a dense_X.
		// each warp identify by wid.
		if (wid < dimTileNum)
			#pragma unroll
			for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize){
				unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W];						// TC_block_col to dense_tile_row.
				unsigned dense_dimIdx = idx / BLK_W;										// dimIndex of the dense tile.
				unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
				unsigned target_idx = wid * BLK_W * BLK_H + idx;
				// boundary test.
				if (source_idx >= dense_bound)
					dense_X[target_idx] = 0;
				else
					dense_X[target_idx] = input[source_idx];
			}

		__syncthreads();

		if (wid < dimTileNum)
		{
			wmma::load_matrix_sync(a_frag, sparse_A, BLK_W);
			wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);

			#pragma unroll
			for (unsigned t = 0; t < a_frag.num_elements; t++) {
				a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
			}

			#pragma unroll
			for (unsigned t = 0; t < b_frag.num_elements; t++) {
				b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
			}
			// Perform the matrix multiplication.
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
		}
	}

	if (wid < dimTileNum)
		// Store the matrix to output matrix.
		// * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
		wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
}


//////////////////////
/// SDDMM forward
//////////////////////
__global__ void sddmm_forward_cuda_kernel(
    const int *__restrict__ nodePointer,
	const int *__restrict__ edgeList,			// edge list.
	const int *__restrict__ blockPartition,		// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 			// eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *__restrict__ in_mat,					// input feature matrix.
	float *edgeFeature							// aggreAGNNed output feature matrix.
)
{
    // printf("at sddmm_forward_cuda_kernel\n");
    unsigned bid = blockIdx.x;										// block_index == row_window_index
    unsigned wid = threadIdx.y;										// warp_index handling multi-dimension > 16.
    unsigned laneid = threadIdx.x;									// lanid of each warp.
    unsigned tid = threadIdx.y * blockDim.x + laneid;				// threadid of each block.

    unsigned threadPerBlock = blockDim.x * blockDim.y;
    unsigned DimIterations =  (embedding_dim + BLK_W - 1) / BLK_W; 	// dimension iteration for output.

    unsigned nid_start = bid * BLK_H;								// starting node_id of current row_window.
    unsigned nid_end = min((bid + 1) * BLK_H, numNodes);			// ending node_id of the current row_window.

    unsigned eIdx_start = nodePointer[nid_start];					            // starting eIdx of current row_window.
    unsigned eIdx_end = nodePointer[nid_end];						            // ending eIdx of the current row_window.
    unsigned num_TC_blocks = (blockPartition[bid] * BLK_W + BLK_H - 1)/BLK_H; 	// number of TC_blocks of the current row_window.

    __shared__ float sparse_A[BLK_H * BLK_H];					// 16 x 16 output sparse matrix.
    __shared__ float sparse_A_val[BLK_H * BLK_H];				// 16 x 16 output sparse matrix.
    #ifdef verify
    __shared__ float verify_A[BLK_H * BLK_H];					// 16 x 16 output sparse matrix.
    #endif 

    __shared__ unsigned sparse_AToX_index[BLK_H];				// TC_block col to dense_tile row.
    __shared__ float dense_X[BLK_H * BLK_W];
    __shared__ float dense_Y[BLK_W * BLK_H];

    wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Processing TC_blocks along the column dimension of Sparse A.
    // The block step here is 2, which is 16 = 8 + 8. 
    // In order to reuse the edgeToColumn in SpMM. 
    for (unsigned i = 0; i < num_TC_blocks; i++ ){

        if (wid == 0 && laneid < BLK_H){
            sparse_AToX_index[laneid] = numNodes + 1;
        }

        __syncthreads();

        #pragma unroll
        for (unsigned idx = tid; idx < BLK_H * BLK_H; idx += threadPerBlock){
            sparse_A[idx] = numEdges + 1;
            sparse_A_val[idx] = 0.0f;
        }

        #pragma unroll
        for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock){
            dense_X[idx] = 0;
            dense_Y[idx] = 0;
        }

        // Initialize sparse_A by using BLK_H (16) threads from the warp-0.
        // currently fetch all neighbors of the current nodes.
        // then to see whether it can fit into current TC_block frame of column.
        #pragma unroll
        // if (tid < WARP_SI)
        for (unsigned eIdx = tid + eIdx_start; eIdx < eIdx_end; eIdx += threadPerBlock){
            unsigned col = edgeToColumn[eIdx];						// condensed column id in sparse_A.
            if (i * BLK_H <= col && col < (i + 1) * BLK_H){			// if the edge in the current TC_block frame of column.
                unsigned row = edgeToRow[eIdx] % BLK_H;				// reverse indexing the row Id of the edge.
                sparse_A[row * BLK_H + col % BLK_H] = eIdx;			// set the edge of the sparse_A.
                sparse_AToX_index[col % BLK_H] = edgeList[eIdx];	// record the mapping from sparse_A colId to rowId of dense_X.
            }
        }		

        __syncthreads();

        for (unsigned warp_iter = 0; warp_iter < DimIterations; warp_iter++){
            // Initialize dense_X by row-major store,
            // Threads of a warp for fetching a dense_X.
            #pragma unroll
            for (unsigned i = tid; i < BLK_H * BLK_W; i += threadPerBlock){
                unsigned dense_rowIdx = i / BLK_W;					
                unsigned dense_dimIdx = i % BLK_W;					
                unsigned target_idx = i;
                unsigned source_idx = (nid_start + dense_rowIdx) * embedding_dim + warp_iter * BLK_W + dense_dimIdx;
                if (source_idx >= numNodes * embedding_dim)
                    dense_X[target_idx] = 0;
                else
                    dense_X[target_idx] = in_mat[source_idx];
            }

            // Initialize dense_Y by column-major store,
            // Threads of a warp for fetching a dense_Y.
            #pragma unroll
            for (unsigned i = tid; i < BLK_W * BLK_H; i += threadPerBlock){
                unsigned dense_rowIdx = sparse_AToX_index[i / BLK_W];					// TC_block_col to dense_tile_row.
                unsigned dense_dimIdx = i % BLK_W;										// dimIndex of the dense tile.
                unsigned target_idx = i;
                unsigned source_idx = dense_rowIdx * embedding_dim + warp_iter * BLK_W + dense_dimIdx;
                if (source_idx >= numNodes * embedding_dim)
                    dense_Y[target_idx] = 0;
                else
                    dense_Y[target_idx] = in_mat[source_idx];
            }

            __syncthreads();

            wmma::load_matrix_sync(a_frag, dense_X, BLK_W);
            wmma::load_matrix_sync(b_frag, dense_Y, BLK_W);

            #pragma unroll
            for (unsigned t = 0; t < a_frag.num_elements; t++) {
                a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
            }

            #pragma unroll
            for (unsigned t = 0; t < b_frag.num_elements; t++) {
                b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
            }

            // Perform the matrix multiplication on Tensor Core
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);		
        } // <--- ending of warp iteration.

        wmma::store_matrix_sync(sparse_A_val, acc_frag, BLK_H, wmma::mem_row_major);
        wmma::fill_fragment(acc_frag, 0.0f);

        // Output the results to sparse matrix edge featureList.
        for (unsigned t = 0; t < BLK_H * BLK_H; t++) {
            unsigned rowId = t / BLK_H;
            unsigned colId = t % BLK_H;
            if (sparse_A[rowId * BLK_H + colId] < numEdges){
                unsigned eIdx = sparse_A[rowId * BLK_H + colId];
                edgeFeature[eIdx] = sparse_A_val[t];
            }
        } //<-- ending of storing output to global memory.
    }
}