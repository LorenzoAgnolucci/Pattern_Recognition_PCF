#ifndef PATTERNRECOGNITION_H_
#define PATTERNRECOGNITION_H_

void findPatternSharedMemory_wrapper(const float* __restrict__ device_target_matrix, const float* __restrict__ device_query_matrix,
		float *device_sads_matrix, const int target_matrix_rows,const int target_matrix_columns,const int query_matrix_rows,const int query_matrix_columns,
		const int sads_matrix_rows,const int sads_matrix_columns, const int TILE_WIDTH);


void findPatternGlobalMemory_wrapper(const float* __restrict__ device_target_matrix, const float* __restrict__ device_query_matrix,
		float *device_sads_matrix,const int target_matrix_rows,const int target_matrix_columns,const int query_matrix_rows,const int query_matrix_columns,
		const int sads_matrix_rows,const int sads_matrix_columns, const int TILE_WIDTH);


void findPatternGlobalMemoryAliasing_wrapper(float* device_target_matrix,  float* device_query_matrix,
		float *device_sads_matrix,const int target_matrix_rows,const int target_matrix_columns,const int query_matrix_rows,const int query_matrix_columns,
		const int sads_matrix_rows,const int sads_matrix_columns, const int TILE_WIDTH);

#endif /* PATTERNRECOGNITION_H_ */
