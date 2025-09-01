//  RUN: iree-opt %s

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]

util.func @pingpong_dt_medium_f16_subgroup_n8(%lhs: tensor<1x?x8x4x4x4x2x4xf16>, %rhs: tensor<1x?x8x2x4x16x2x4xf16>, %unused_acc: tensor<1x1x8x8x2x4x16x4xf32>) -> tensor<1x1x8x8x2x4x16x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f16
  %lhs_shared = memref.alloc() : memref<1x1x8x4x4x4x2x4xf16, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x1x8x2x4x16x2x4xf16, #gpu.address_space<workgroup>>

  %dim = tensor.dim %lhs, %c1 : tensor<1x?x8x4x4x4x2x4xf16>

  scf.forall (%id) in (512) {
    %delin:4 = affine.delinearize_index %id into (8, 4, 4, 4) : index, index, index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x4x4x4x2x4xf16> to tensor<1x1x1x1x1x1x2x4xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x4xf16>, vector<1x1x1x1x1x1x2x4xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x4xf16>, memref<1x1x8x4x4x4x2x4xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (1024) {
    %delin:4 = affine.delinearize_index %id into (8, 2, 4, 16) : index, index, index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x2x4x16x2x4xf16> to tensor<1x1x1x1x1x1x2x4xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x4xf16>, vector<1x1x1x1x1x1x2x4xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x4xf16>,  memref<1x1x8x2x4x16x2x4xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %0 = tensor.empty() : tensor<1x1x8x8x2x4x16x4xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x1x8x8x2x4x16x4xf32> {
    %ids_lhs:4 = affine.delinearize_index %id into (8, 4, 4, 4) : index, index, index, index
    %ids_rhs:3 = affine.delinearize_index %id into (8, 4, 16) : index, index, index

    // %glb_lhs = arith.addi %ids_lhs#0, %c4 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<8x2x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c1 to %dim step %c1 iter_args(%iter = %2) -> vector<8x2x1x4xf32> {

      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x4x4x4x2x4xf16, #gpu.address_space<workgroup>>, vector<1x1x8x1x1x1x1x4xf16>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x2x4x16x2x4xf16, #gpu.address_space<workgroup>>, vector<1x1x1x2x1x1x1x4xf16>
      %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x1x8x1x1x1x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x1x1x2x1x1x1x4xf16> to vector<2x1x1x4xf16>

      // rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%c0, %i, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 4] [1, 1, 1, 1, 1, 1, 1, 1] :  tensor<1x?x8x2x4x16x2x4xf16> to tensor<1x1x1x1x1x1x2x4xf16>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x4xf16>, vector<1x1x1x1x1x1x2x4xf16>
      %rhs_thread_1 = tensor.extract_slice %rhs [%c0, %i, %ids_rhs#0, %c1, %ids_rhs#1, %ids_rhs#2, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x2x4x16x2x4xf16> to tensor<1x1x1x1x1x1x2x4xf16>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x4xf16>, vector<1x1x1x1x1x1x2x4xf16>

      // rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x4x4x4x2x4xf16, #gpu.address_space<workgroup>>, vector<1x1x8x1x1x1x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x2x4x16x2x4xf16, #gpu.address_space<workgroup>>, vector<1x1x1x2x1x1x1x4xf16>
      %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x1x8x1x1x1x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x1x1x2x1x1x1x4xf16> to vector<2x1x1x4xf16>

      // rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%c0, %i, %ids_lhs#0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x4x4x4x2x4xf16> to tensor<1x1x1x1x1x1x2x4xf16>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x4xf16>, vector<1x1x1x1x1x1x2x4xf16>
      // %lhs_thread_1 = tensor.extract_slice %lhs [%c0, %i, %glb_lhs, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x4x4x4x2x4xf16> to tensor<1x1x1x1x1x1x2x4xf16>
      // %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x4xf16>, vector<1x1x1x1x1x1x2x4xf16>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x4xf16>, memref<1x1x8x2x4x16x2x4xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %c0, %ids_rhs#0, %c1, %ids_rhs#1, %ids_rhs#2, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x4xf16>, memref<1x1x8x2x4x16x2x4xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%c0, %c0, %ids_lhs#0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x4xf16>, memref<1x1x8x4x4x4x2x4xf16, #gpu.address_space<workgroup>>
      // vector.transfer_write %lhs_vec_local_1, %lhs_shared [%c0, %c0, %glb_lhs, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x4xf16>, memref<1x1x8x4x4x4x2x4xf16, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<8x2x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x4x4x4x2x4xf16, #gpu.address_space<workgroup>>, vector<1x1x8x1x1x1x1x4xf16>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x2x4x16x2x4xf16, #gpu.address_space<workgroup>>, vector<1x1x1x2x1x1x1x4xf16>
    %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x1x8x1x1x1x1x4xf16> to vector<8x1x1x4xf16>
    %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x1x1x2x1x1x1x4xf16> to vector<2x1x1x4xf16>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x4x4x4x2x4xf16, #gpu.address_space<workgroup>>, vector<1x1x8x1x1x1x1x4xf16>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x2x4x16x2x4xf16, #gpu.address_space<workgroup>>, vector<1x1x1x2x1x1x1x4xf16>
    %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x1x8x1x1x1x1x4xf16> to vector<8x1x1x4xf16>
    %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x1x1x2x1x1x1x4xf16> to vector<2x1x1x4xf16>
 
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x8x2x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<8x2x1x4xf32> to vector<1x1x1x8x2x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x8x2x1x1x4xf32>, tensor<1x1x1x8x2x1x1x4xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids_rhs#0, %c0, %c0, %ids_rhs#1, %ids_rhs#2, %c0] [1, 1, 1, 8, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x8x2x1x1x4xf32> into tensor<1x1x8x8x2x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x8x8x2x4x16x4xf32>
}

util.func @pingpong_dt_medium_f16_subgroup_n8_reduce_dims(%lhs_base: tensor<1x?x8x4x4x4x2x4xf16>, %rhs_base: tensor<1x?x8x2x4x16x2x4xf16>, %unused_acc: tensor<1x1x8x8x2x4x16x4xf32>) -> tensor<1x1x8x8x2x4x16x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f16

  %dim = tensor.dim %rhs_base, %c1 : tensor<1x?x8x2x4x16x2x4xf16>

  %lhs = tensor.collapse_shape %lhs_base [[0, 1], [2], [3, 4, 5], [6, 7]] : tensor<1x?x8x4x4x4x2x4xf16> into tensor<?x8x64x8xf16>
  %rhs = tensor.collapse_shape %rhs_base [[0, 1], [2, 3], [4, 5], [6, 7]] : tensor<1x?x8x2x4x16x2x4xf16> into tensor<?x16x64x8xf16>

  %lhs_shared = memref.alloc() : memref<2x8x64x8xf16, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<2x16x64x8xf16, #gpu.address_space<workgroup>>

  scf.forall (%id) in (512) {
    %delin:2 = affine.delinearize_index %id into (8, 64) : index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x8x64x8xf16> to tensor<1x1x1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<2x8x64x8xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (1024) {
    %delin:2 = affine.delinearize_index %id into (16, 64) : index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>,  memref<2x16x64x8xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

  gpu.barrier

  %0 = tensor.empty() : tensor<1x1x8x8x2x4x16x4xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x1x8x8x2x4x16x4xf32> {
    %ids:2 = affine.delinearize_index %id into (8, 64) : index, index
    %threads:2 = affine.delinearize_index %ids#1 into (4, 16) : index, index

    %glb_lhs = arith.addi %ids#0, %c4 overflow<nsw, nuw> : index
    
    %glb0_rhs = arith.muli %ids#0, %c2 overflow<nsw, nuw> : index
    %glb1_rhs = arith.addi %glb0_rhs, %c1 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<8x2x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c1 to %dim step %c1 iter_args(%iter = %2) -> vector<8x2x1x4xf32> {

      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<2x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
      %lhs_vec_0_t = vector.transpose %lhs_vec_0, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>

      // rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%i, %glb0_rhs, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] :  tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
      %rhs_thread_1 = tensor.extract_slice %rhs [%i, %glb1_rhs, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>

      // rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<2x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
      %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>

      // rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %ids#0, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x8x64x8xf16> to tensor<1x1x1x8xf16>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %glb0_rhs, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<2x16x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %glb1_rhs, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<2x16x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%c0, %ids#0, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<2x8x64x8xf16, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<8x2x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<2x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
    %lhs_vec_0_t = vector.transpose %lhs_vec_0, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
    %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<2x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
    %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
    %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>
 
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x8x2x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<8x2x1x4xf32> to vector<1x1x1x8x2x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x8x2x1x1x4xf32>, tensor<1x1x1x8x2x1x1x4xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#0, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 8, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x8x2x1x1x4xf32> into tensor<1x1x8x8x2x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x8x8x2x4x16x4xf32>
}

util.func @pingpong_dt_medium_f16_subgroup_n8_double_buffer(%lhs_base: tensor<1x?x8x4x4x4x2x4xf16>, %rhs_base: tensor<1x?x8x2x4x16x2x4xf16>, %unused_acc: tensor<1x1x8x8x2x4x16x4xf32>) -> tensor<1x1x8x8x2x4x16x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f16

  %dim = tensor.dim %rhs_base, %c1 : tensor<1x?x8x2x4x16x2x4xf16>

  %lhs = tensor.collapse_shape %lhs_base [[0, 1], [2], [3, 4, 5], [6, 7]] : tensor<1x?x8x4x4x4x2x4xf16> into tensor<?x8x64x8xf16>
  %rhs = tensor.collapse_shape %rhs_base [[0, 1], [2, 3], [4, 5], [6, 7]] : tensor<1x?x8x2x4x16x2x4xf16> into tensor<?x16x64x8xf16>

  %lhs_shared = memref.alloc() : memref<2x8x64x8xf16, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>

  scf.forall (%id) in (512) {
    %delin:2 = affine.delinearize_index %id into (8, 64) : index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x8x64x8xf16> to tensor<1x1x1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<2x8x64x8xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (1024) {
    %delin:2 = affine.delinearize_index %id into (16, 64) : index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>,  memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

  gpu.barrier

  %0 = tensor.empty() : tensor<1x1x8x8x2x4x16x4xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x1x8x8x2x4x16x4xf32> {
    %ids:2 = affine.delinearize_index %id into (8, 64) : index, index
    %threads:2 = affine.delinearize_index %ids#1 into (4, 16) : index, index

    %glb_lhs = arith.addi %ids#0, %c4 overflow<nsw, nuw> : index
    
    %glb0_rhs = arith.muli %ids#0, %c2 overflow<nsw, nuw> : index
    %glb1_rhs = arith.addi %glb0_rhs, %c1 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<8x2x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c1 to %dim step %c1 iter_args(%iter = %2) -> vector<8x2x1x4xf32> {
      
      %rem0 = arith.remui %i, %c2 : index
      %add0 = arith.addi %i, %c1 : index
      %rem1 = arith.remui %add0, %c2 : index
      

      %lhs_vec_0 = vector.transfer_read %lhs_shared[%rem1, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
      %lhs_vec_0_t = vector.transpose %lhs_vec_0, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>

      // rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%i, %glb0_rhs, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] :  tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
      %rhs_thread_1 = tensor.extract_slice %rhs [%i, %glb1_rhs, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>

      // rocdl.sched.barrier 0

      // %lhs_vec_2 = vector.transfer_read %lhs_shared[%rem1, %c0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
      // %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>

      // rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %ids#0, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x8x64x8xf16> to tensor<1x1x1x8xf16>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared[%rem1, %c0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %glb0_rhs, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %glb1_rhs, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%rem0, %ids#0, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<2x8x64x8xf16, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<8x2x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c1, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
    %lhs_vec_0_t = vector.transpose %lhs_vec_0, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
    %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c1, %c0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
    %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
    %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>
 
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x8x2x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<8x2x1x4xf32> to vector<1x1x1x8x2x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x8x2x1x1x4xf32>, tensor<1x1x1x8x2x1x1x4xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#0, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 8, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x8x2x1x1x4xf32> into tensor<1x1x8x8x2x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x8x8x2x4x16x4xf32>
}

util.func @pingpong_dt_medium_f16_subgroup_n8_intrinsics_m16(%lhs_base: tensor<1x?x16x4x4x4x2x4xf16>, %rhs_base: tensor<1x?x8x2x4x16x2x4xf16>, %unused_acc: tensor<1x1x8x16x2x4x16x4xf32>) -> tensor<1x1x8x16x2x4x16x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f16

  %dim = tensor.dim %rhs_base, %c1 : tensor<1x?x8x2x4x16x2x4xf16>

  %lhs = tensor.collapse_shape %lhs_base [[0, 1], [2], [3, 4, 5], [6, 7]] : tensor<1x?x16x4x4x4x2x4xf16> into tensor<?x16x64x8xf16>
  %rhs = tensor.collapse_shape %rhs_base [[0, 1], [2, 3], [4, 5], [6, 7]] : tensor<1x?x8x2x4x16x2x4xf16> into tensor<?x16x64x8xf16>

  %lhs_shared = memref.alloc() : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>

  scf.forall (%id) in (1024) {
    %delin:2 = affine.delinearize_index %id into (16, 64) : index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (1024) {
    %delin:2 = affine.delinearize_index %id into (16, 64) : index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>,  memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

  gpu.barrier

  %0 = tensor.empty() : tensor<1x1x8x16x2x4x16x4xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x1x8x16x2x4x16x4xf32> {
    %ids:2 = affine.delinearize_index %id into (8, 64) : index, index
    %threads:2 = affine.delinearize_index %ids#1 into (4, 16) : index, index

    // %glb_lhs = arith.addi %ids#0, %c4 overflow<nsw, nuw> : index
    
    %glb0 = arith.muli %ids#0, %c2 overflow<nsw, nuw> : index
    %glb1 = arith.addi %glb0, %c1 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<16x2x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c1 to %dim step %c1 iter_args(%iter = %2) -> vector<16x2x1x4xf32> {

//      %rhs_vec = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x8xf16>
//      %rhs_vec_t = vector.transpose %rhs_vec, [1, 2, 0, 3] : vector<1x2x1x8xf16> to vector<2x1x1x8xf16>
//
//      %rhs_vec_0 =  vector.extract_strided_slice %rhs_vec_t {offsets = [0, 0, 0, 0], sizes = [2, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<2x1x1x8xf16> to vector<2x1x1x4xf16>
//      %rhs_vec_2 =  vector.extract_strided_slice %rhs_vec_t {offsets = [0, 0, 0, 4], sizes = [2, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<2x1x1x8xf16> to vector<2x1x1x4xf16>

      rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %glb0, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
      %lhs_thread_1 = tensor.extract_slice %lhs [%i, %glb1, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>


      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%i, %glb0, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] :  tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
      %rhs_thread_1 = tensor.extract_slice %rhs [%i, %glb1, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>

      rocdl.sched.barrier 0

      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x16x1x4xf16>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
      %lhs_vec_0_t = vector.transpose %lhs_vec_0, [1, 2, 0, 3] : vector<1x16x1x4xf16> to vector<16x1x1x4xf16>
      %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>

      // %lhs_vec = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x16x1x8xf16>
      // %lhs_vec_t = vector.transpose %lhs_vec, [1, 2, 0, 3] : vector<1x16x1x8xf16> to vector<16x1x1x8xf16>
      // %lhs_vec_0 = vector.extract_strided_slice %lhs_vec_t {offsets = [0, 0, 0, 0], sizes = [16, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<16x1x1x8xf16> to vector<16x1x1x4xf16>
      // %lhs_vec_2 = vector.extract_strided_slice %lhs_vec_t {offsets = [0, 0, 0, 4], sizes = [16, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<16x1x1x8xf16> to vector<16x1x1x4xf16>
//       

      // %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x16x1x4xf16>
      // %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
      // %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x16x1x4xf16> to vector<16x1x1x4xf16>
      // %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>

      // rocdl.sched.barrier 0

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<16x1x1x4xf16>, vector<2x1x1x4xf16> into vector<16x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x16x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
      %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x16x1x4xf16> to vector<16x1x1x4xf16>
      %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>
      
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %glb0, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %glb1, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%c0, %glb0, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%c0, %glb1, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<16x1x1x4xf16>, vector<2x1x1x4xf16> into vector<16x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<16x2x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x16x1x8xf16>
    %rhs_vec = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x8xf16>
    %lhs_vec_t = vector.transpose %lhs_vec, [1, 2, 0, 3] : vector<1x16x1x8xf16> to vector<16x1x1x8xf16>
    %rhs_vec_t = vector.transpose %rhs_vec, [1, 2, 0, 3] : vector<1x2x1x8xf16> to vector<2x1x1x8xf16>

    %lhs_vec_0 = vector.extract_strided_slice %lhs_vec_t {offsets = [0, 0, 0, 0], sizes = [16, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<16x1x1x8xf16> to vector<16x1x1x4xf16>
    %rhs_vec_0 =  vector.extract_strided_slice %rhs_vec_t {offsets = [0, 0, 0, 0], sizes = [2, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<2x1x1x8xf16> to vector<2x1x1x4xf16>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0, %rhs_vec_0) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<16x1x1x4xf16>, vector<2x1x1x4xf16> into vector<16x2x1x4xf32>

    %lhs_vec_2 = vector.extract_strided_slice %lhs_vec_t {offsets = [0, 0, 0, 4], sizes = [16, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<16x1x1x8xf16> to vector<16x1x1x4xf16>
    %rhs_vec_2 =  vector.extract_strided_slice %rhs_vec_t {offsets = [0, 0, 0, 4], sizes = [2, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<2x1x1x8xf16> to vector<2x1x1x4xf16>

    // %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x16x1x4xf16>
    // %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
    // %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x16x1x4xf16> to vector<16x1x1x4xf16>
    // %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>
 
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2, %rhs_vec_2) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<16x1x1x4xf16>, vector<2x1x1x4xf16> into vector<16x2x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x16x2x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<16x2x1x4xf32> to vector<1x1x1x16x2x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x16x2x1x1x4xf32>, tensor<1x1x1x16x2x1x1x4xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#0, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 16, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x16x2x1x1x4xf32> into tensor<1x1x8x16x2x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x8x16x2x4x16x4xf32>
}

util.func @pingpong_dt_medium_f16_subgroup_n8_intrinsics_m16_split(%lhs_base: tensor<1x?x16x4x4x4x2x4xf16>, %rhs_base: tensor<1x?x8x2x4x16x2x4xf16>, %unused_acc: tensor<1x1x8x16x2x4x16x4xf32>) -> tensor<1x1x8x16x2x4x16x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f16

  %dim = tensor.dim %rhs_base, %c1 : tensor<1x?x8x2x4x16x2x4xf16>

  %lhs = tensor.collapse_shape %lhs_base [[0, 1], [2], [3, 4, 5], [6, 7]] : tensor<1x?x16x4x4x4x2x4xf16> into tensor<?x16x64x8xf16>
  %rhs = tensor.collapse_shape %rhs_base [[0, 1], [2, 3], [4, 5], [6, 7]] : tensor<1x?x8x2x4x16x2x4xf16> into tensor<?x16x64x8xf16>

  %lhs_shared = memref.alloc() : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>

  scf.forall (%id) in (1024) {
    %delin:2 = affine.delinearize_index %id into (16, 64) : index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (1024) {
    %delin:2 = affine.delinearize_index %id into (16, 64) : index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>,  memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

  gpu.barrier

  %0 = tensor.empty() : tensor<1x1x8x16x2x4x16x4xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x1x8x16x2x4x16x4xf32> {
    %ids:2 = affine.delinearize_index %id into (8, 64) : index, index
    %threads:2 = affine.delinearize_index %ids#1 into (4, 16) : index, index

    // %glb_lhs = arith.addi %ids#0, %c4 overflow<nsw, nuw> : index
    
    // %glb_lhs0 = arith.muli %ids#0, %c2 overflow<nsw, nuw> : index
    %glb0 = arith.muli %ids#0, %c2 overflow<nsw, nuw> : index
    %glb1 = arith.addi %glb0, %c1 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<16x2x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c1 to %dim step %c1 iter_args(%iter = %2) -> vector<16x2x1x4xf32> {

      %in_0 =  vector.extract_strided_slice %iter {offsets = [0, 0, 0, 0], sizes = [8, 2, 1, 4], strides = [1, 1, 1, 1]} : vector<16x2x1x4xf32> to vector<8x2x1x4xf32>
      %in_2 =  vector.extract_strided_slice %iter {offsets = [8, 0, 0, 0], sizes = [8, 2, 1, 4], strides = [1, 1, 1, 1]} : vector<16x2x1x4xf32> to vector<8x2x1x4xf32>

//      %rhs_vec = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x8xf16>
//      %rhs_vec_t = vector.transpose %rhs_vec, [1, 2, 0, 3] : vector<1x2x1x8xf16> to vector<2x1x1x8xf16>
//
//      %rhs_vec_0 =  vector.extract_strided_slice %rhs_vec_t {offsets = [0, 0, 0, 0], sizes = [2, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<2x1x1x8xf16> to vector<2x1x1x4xf16>
//      %rhs_vec_2 =  vector.extract_strided_slice %rhs_vec_t {offsets = [0, 0, 0, 4], sizes = [2, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<2x1x1x8xf16> to vector<2x1x1x4xf16>

      rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%i, %glb0, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] :  tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
      %rhs_thread_1 = tensor.extract_slice %rhs [%i, %glb1, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %glb0, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
      %lhs_thread_1 = tensor.extract_slice %lhs [%i, %glb1, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>

      rocdl.sched.barrier 0

      %lhs_vec_0_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
      %lhs_vec_0_0_t = vector.transpose %lhs_vec_0_0, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>

      rocdl.sched.barrier 0

      // rocdl.sched.barrier 0

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0_0 = iree_codegen.inner_tiled ins(%lhs_vec_0_0_t, %rhs_vec_0_t) outs(%in_0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      %lhs_vec_0_2 = vector.transfer_read %lhs_shared[%c0, %c8, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
      %lhs_vec_0_2_t = vector.transpose %lhs_vec_0_2, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>
      
      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0_2 = iree_codegen.inner_tiled ins(%lhs_vec_0_2_t, %rhs_vec_0_t) outs(%in_2) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      %lhs_vec_2_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %lhs_vec_2_0_t = vector.transpose %lhs_vec_2_0, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>

      %lhs_vec_2_2 = vector.transfer_read %lhs_shared[%c0, %c8, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %lhs_vec_2_2_t = vector.transpose %lhs_vec_2_2, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2_0 = iree_codegen.inner_tiled ins(%lhs_vec_2_0_t, %rhs_vec_2_t) outs(%dot0_0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %glb0, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %glb1, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%c0, %glb0, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%c0, %glb1, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2_2 = iree_codegen.inner_tiled ins(%lhs_vec_2_2_t, %rhs_vec_2_t) outs(%dot0_2) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<2x1x1x4xf16> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      %out_0 = vector.insert_strided_slice %dot2_0, %2 {offsets = [0, 0, 0, 0], strides = [1, 1, 1, 1]} : vector<8x2x1x4xf32> into vector<16x2x1x4xf32>
      %out_2 = vector.insert_strided_slice %dot2_2, %out_0 {offsets = [8, 0, 0, 0], strides = [1, 1, 1, 1]} : vector<8x2x1x4xf32> into vector<16x2x1x4xf32>

      scf.yield %out_2 : vector<16x2x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x16x1x8xf16>
    %rhs_vec = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x8xf16>
    %lhs_vec_t = vector.transpose %lhs_vec, [1, 2, 0, 3] : vector<1x16x1x8xf16> to vector<16x1x1x8xf16>
    %rhs_vec_t = vector.transpose %rhs_vec, [1, 2, 0, 3] : vector<1x2x1x8xf16> to vector<2x1x1x8xf16>

    %lhs_vec_0 = vector.extract_strided_slice %lhs_vec_t {offsets = [0, 0, 0, 0], sizes = [16, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<16x1x1x8xf16> to vector<16x1x1x4xf16>
    %rhs_vec_0 =  vector.extract_strided_slice %rhs_vec_t {offsets = [0, 0, 0, 0], sizes = [2, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<2x1x1x8xf16> to vector<2x1x1x4xf16>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0, %rhs_vec_0) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<16x1x1x4xf16>, vector<2x1x1x4xf16> into vector<16x2x1x4xf32>

    %lhs_vec_2 = vector.extract_strided_slice %lhs_vec_t {offsets = [0, 0, 0, 4], sizes = [16, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<16x1x1x8xf16> to vector<16x1x1x4xf16>
    %rhs_vec_2 =  vector.extract_strided_slice %rhs_vec_t {offsets = [0, 0, 0, 4], sizes = [2, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<2x1x1x8xf16> to vector<2x1x1x4xf16>

    // %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x16x1x4xf16>
    // %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
    // %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x16x1x4xf16> to vector<16x1x1x4xf16>
    // %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>
 
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2, %rhs_vec_2) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<16x1x1x4xf16>, vector<2x1x1x4xf16> into vector<16x2x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x16x2x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<16x2x1x4xf32> to vector<1x1x1x16x2x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x16x2x1x1x4xf32>, tensor<1x1x1x16x2x1x1x4xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#0, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 16, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x16x2x1x1x4xf32> into tensor<1x1x8x16x2x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x8x16x2x4x16x4xf32>
}

util.func @pingpong_dt_medium_f16_subgroup_n8_intrinsics_m16_intrinsics_n4_split(%lhs_base: tensor<1x?x16x4x4x4x2x4xf16>, %rhs_base: tensor<1x?x8x4x4x16x2x4xf16>, %unused_acc: tensor<1x1x8x16x4x4x16x4xf32>) -> tensor<1x1x8x16x4x4x16x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f16

  %dim = tensor.dim %rhs_base, %c1 : tensor<1x?x8x4x4x16x2x4xf16>

  %lhs = tensor.collapse_shape %lhs_base [[0, 1], [2], [3, 4, 5], [6, 7]] : tensor<1x?x16x4x4x4x2x4xf16> into tensor<?x16x64x8xf16>
  %rhs = tensor.collapse_shape %rhs_base [[0, 1], [2, 3], [4, 5], [6, 7]] : tensor<1x?x8x4x4x16x2x4xf16> into tensor<?x32x64x8xf16>

  %lhs_shared = memref.alloc() : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x32x64x8xf16, #gpu.address_space<workgroup>>

  scf.forall (%id) in (1024) {
    %delin:2 = affine.delinearize_index %id into (16, 64) : index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (32, 64) : index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x32x64x8xf16> to tensor<1x1x1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>,  memref<1x32x64x8xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

  gpu.barrier

  %0 = tensor.empty() : tensor<1x1x8x16x4x4x16x4xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x1x8x16x4x4x16x4xf32> {
    %ids:2 = affine.delinearize_index %id into (8, 64) : index, index
    %threads:2 = affine.delinearize_index %ids#1 into (4, 16) : index, index

    // %glb_lhs = arith.addi %ids#0, %c4 overflow<nsw, nuw> : index
    
    // %glb_lhs0 = arith.muli %ids#0, %c2 overflow<nsw, nuw> : index
    %glb0_rhs = arith.muli %ids#0, %c4 overflow<nsw, nuw> : index
    %glb1_rhs = arith.addi %glb0_rhs, %c1 overflow<nsw, nuw> : index
    %glb2_rhs = arith.addi %glb0_rhs, %c2 overflow<nsw, nuw> : index
    %glb3_rhs = arith.addi %glb0_rhs, %c3 overflow<nsw, nuw> : index

    %glb0 = arith.muli %ids#0, %c2 overflow<nsw, nuw> : index
    %glb1 = arith.addi %glb0, %c1 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<16x4x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c1 to %dim step %c1 iter_args(%iter = %2) -> vector<16x4x1x4xf32> {

      %in_0 =  vector.extract_strided_slice %iter {offsets = [0, 0, 0, 0], sizes = [8, 4, 1, 4], strides = [1, 1, 1, 1]} : vector<16x4x1x4xf32> to vector<8x4x1x4xf32>
      %in_2 =  vector.extract_strided_slice %iter {offsets = [8, 0, 0, 0], sizes = [8, 4, 1, 4], strides = [1, 1, 1, 1]} : vector<16x4x1x4xf32> to vector<8x4x1x4xf32>

      rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%i, %glb0_rhs, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] :  tensor<?x32x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
      %rhs_thread_1 = tensor.extract_slice %rhs [%i, %glb1_rhs, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x32x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
      %rhs_thread_2 = tensor.extract_slice %rhs [%i, %glb2_rhs, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x32x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_2 = vector.transfer_read %rhs_thread_2 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
      %rhs_thread_3 = tensor.extract_slice %rhs [%i, %glb3_rhs, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x32x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_3 = vector.transfer_read %rhs_thread_3 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %glb0, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
      %lhs_thread_1 = tensor.extract_slice %lhs [%i, %glb1, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x16x64x8xf16> to tensor<1x1x1x8xf16>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>

      rocdl.sched.barrier 0

      %lhs_vec_0_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x32x64x8xf16, #gpu.address_space<workgroup>>, vector<1x4x1x4xf16>
      %lhs_vec_0_0_t = vector.transpose %lhs_vec_0_0, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

      rocdl.sched.barrier 0

      // rocdl.sched.barrier 0

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0_0 = iree_codegen.inner_tiled ins(%lhs_vec_0_0_t, %rhs_vec_0_t) outs(%in_0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      %lhs_vec_0_2 = vector.transfer_read %lhs_shared[%c0, %c8, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x32x64x8xf16, #gpu.address_space<workgroup>>, vector<1x4x1x4xf16>
      %lhs_vec_0_2_t = vector.transpose %lhs_vec_0_2, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>
      
      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0_2 = iree_codegen.inner_tiled ins(%lhs_vec_0_2_t, %rhs_vec_0_t) outs(%in_2) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      %lhs_vec_2_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %lhs_vec_2_0_t = vector.transpose %lhs_vec_2_0, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>

      %lhs_vec_2_2 = vector.transfer_read %lhs_shared[%c0, %c8, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %lhs_vec_2_2_t = vector.transpose %lhs_vec_2_2, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2_0 = iree_codegen.inner_tiled ins(%lhs_vec_2_0_t, %rhs_vec_2_t) outs(%dot0_0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %glb0_rhs, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x32x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %glb1_rhs, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x32x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_2, %rhs_shared [%c0, %glb2_rhs, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x32x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_3, %rhs_shared [%c0, %glb3_rhs, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x32x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%c0, %glb0, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%c0, %glb1, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x16x64x8xf16, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2_2 = iree_codegen.inner_tiled ins(%lhs_vec_2_2_t, %rhs_vec_2_t) outs(%dot0_2) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      %out_0 = vector.insert_strided_slice %dot2_0, %2 {offsets = [0, 0, 0, 0], strides = [1, 1, 1, 1]} : vector<8x4x1x4xf32> into vector<16x4x1x4xf32>
      %out_2 = vector.insert_strided_slice %dot2_2, %out_0 {offsets = [8, 0, 0, 0], strides = [1, 1, 1, 1]} : vector<8x4x1x4xf32> into vector<16x4x1x4xf32>

      scf.yield %out_2 : vector<16x4x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x16x1x8xf16>
    %rhs_vec = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x32x64x8xf16, #gpu.address_space<workgroup>>, vector<1x4x1x8xf16>
    %lhs_vec_t = vector.transpose %lhs_vec, [1, 2, 0, 3] : vector<1x16x1x8xf16> to vector<16x1x1x8xf16>
    %rhs_vec_t = vector.transpose %rhs_vec, [1, 2, 0, 3] : vector<1x4x1x8xf16> to vector<4x1x1x8xf16>

    %lhs_vec_0 = vector.extract_strided_slice %lhs_vec_t {offsets = [0, 0, 0, 0], sizes = [16, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<16x1x1x8xf16> to vector<16x1x1x4xf16>
    %rhs_vec_0 =  vector.extract_strided_slice %rhs_vec_t {offsets = [0, 0, 0, 0], sizes = [4, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<4x1x1x8xf16> to vector<4x1x1x4xf16>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0, %rhs_vec_0) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<16x1x1x4xf16>, vector<4x1x1x4xf16> into vector<16x4x1x4xf32>

    %lhs_vec_2 = vector.extract_strided_slice %lhs_vec_t {offsets = [0, 0, 0, 4], sizes = [16, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<16x1x1x8xf16> to vector<16x1x1x4xf16>
    %rhs_vec_2 =  vector.extract_strided_slice %rhs_vec_t {offsets = [0, 0, 0, 4], sizes = [4, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<4x1x1x8xf16> to vector<4x1x1x4xf16>

    // %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x16x1x4xf16>
    // %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x8xf16, #gpu.address_space<workgroup>>, vector<1x2x1x4xf16>
    // %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x16x1x4xf16> to vector<16x1x1x4xf16>
    // %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x4xf16> to vector<2x1x1x4xf16>
 
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2, %rhs_vec_2) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<16x1x1x4xf16>, vector<4x1x1x4xf16> into vector<16x4x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x16x4x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<16x4x1x4xf32> to vector<1x1x1x16x4x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x16x4x1x1x4xf32>, tensor<1x1x1x16x4x1x1x4xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#0, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 16, 4, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x16x4x1x1x4xf32> into tensor<1x1x8x16x4x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x8x16x4x4x16x4xf32>
}


util.func @pingpong_dt_medium_f16_subgroup_n8_intrinsics_n4(%lhs_base: tensor<1x?x8x4x4x4x2x4xf16>, %rhs_base: tensor<1x?x8x4x4x16x2x4xf16>, %unused_acc: tensor<1x1x8x8x4x4x16x4xf32>) -> tensor<1x1x8x8x4x4x16x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f16

  %dim = tensor.dim %rhs_base, %c1 : tensor<1x?x8x4x4x16x2x4xf16>

  %lhs = tensor.collapse_shape %lhs_base [[0, 1], [2], [3, 4, 5], [6, 7]] : tensor<1x?x8x4x4x4x2x4xf16> into tensor<?x8x64x8xf16>
  %rhs = tensor.collapse_shape %rhs_base [[0, 1], [2, 3], [4, 5], [6, 7]] : tensor<1x?x8x4x4x16x2x4xf16> into tensor<?x32x64x8xf16>

  %lhs_shared = memref.alloc() : memref<1x8x64x8xf16, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x32x64x8xf16, #gpu.address_space<workgroup>>

  scf.forall (%id) in (512) {
    %delin:2 = affine.delinearize_index %id into (8, 64) : index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x8x64x8xf16> to tensor<1x1x1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x8x64x8xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:2 = affine.delinearize_index %id into (32, 64) : index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x32x64x8xf16> to tensor<1x1x1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>,  memref<1x32x64x8xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

  gpu.barrier

  %0 = tensor.empty() : tensor<1x1x8x8x4x4x16x4xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x1x8x8x4x4x16x4xf32> {
    %ids:2 = affine.delinearize_index %id into (8, 64) : index, index
    %threads:2 = affine.delinearize_index %ids#1 into (4, 16) : index, index

    %glb_lhs = arith.addi %ids#0, %c4 overflow<nsw, nuw> : index
    
    %glb0_rhs = arith.muli %ids#0, %c4 overflow<nsw, nuw> : index
    %glb1_rhs = arith.addi %glb0_rhs, %c1 overflow<nsw, nuw> : index
    %glb2_rhs = arith.addi %glb0_rhs, %c2 overflow<nsw, nuw> : index
    %glb3_rhs = arith.addi %glb0_rhs, %c3 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<8x4x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c1 to %dim step %c1 iter_args(%iter = %2) -> vector<8x4x1x4xf32> {

      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x32x64x8xf16, #gpu.address_space<workgroup>>, vector<1x4x1x4xf16>
      %lhs_vec_0_t = vector.transpose %lhs_vec_0, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

      rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%i, %glb0_rhs, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] :  tensor<?x32x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
      %rhs_thread_1 = tensor.extract_slice %rhs [%i, %glb1_rhs, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x32x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
      %rhs_thread_2 = tensor.extract_slice %rhs [%i, %glb2_rhs, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x32x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_2 = vector.transfer_read %rhs_thread_2 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>
      %rhs_thread_3 = tensor.extract_slice %rhs [%i, %glb3_rhs, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x32x64x8xf16> to tensor<1x1x1x8xf16>
      %rhs_vec_local_3 = vector.transfer_read %rhs_thread_3 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>

      rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x32x64x8xf16, #gpu.address_space<workgroup>>, vector<1x4x1x4xf16>
      %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

      rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %ids#0, %ids#1, %c0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<?x8x64x8xf16> to tensor<1x1x1x8xf16>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf16>, vector<1x1x1x8xf16>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %glb0_rhs, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x32x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %glb1_rhs, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x32x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_2, %rhs_shared [%c0, %glb2_rhs, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x32x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_3, %rhs_shared [%c0, %glb3_rhs, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x32x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%c0, %ids#0, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<1x8x64x8xf16, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<8x4x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x32x64x8xf16, #gpu.address_space<workgroup>>, vector<1x4x1x4xf16>
    %lhs_vec_0_t = vector.transpose %lhs_vec_0, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
    %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x8x1x4xf16>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c4], %cst {in_bounds = [true, true, true, true]} : memref<1x32x64x8xf16, #gpu.address_space<workgroup>>, vector<1x4x1x4xf16>
    %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
    %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>
 
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x8x4x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<8x4x1x4xf32> to vector<1x1x1x8x4x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x8x4x1x1x4xf32>, tensor<1x1x1x8x4x1x1x4xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#0, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 8, 4, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x8x4x1x1x4xf32> into tensor<1x1x8x8x4x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x8x8x4x4x16x4xf32>
}

util.func @pingpong_dt_medium_f16_subgroup_m2_n4_intrinsics_n4(%lhs_base: tensor<1x?x2x8x4x4x4x2x4xf16>, %rhs_base: tensor<1x?x4x4x4x16x2x4xf16>, %unused_acc: tensor<1x1x2x4x8x4x4x16x4xf32>) -> tensor<1x1x2x4x8x4x4x16x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f16

  %dim = tensor.dim %rhs_base, %c1 : tensor<1x?x4x4x4x16x2x4xf16>

  %lhs = tensor.collapse_shape %lhs_base [[0, 1], [2], [3], [4, 5, 6], [7, 8]] : tensor<1x?x2x8x4x4x4x2x4xf16> into tensor<?x2x8x64x8xf16>
  %rhs = tensor.collapse_shape %rhs_base [[0, 1], [2], [3], [4, 5], [6, 7]] : tensor<1x?x4x4x4x16x2x4xf16> into tensor<?x4x4x64x8xf16>

  %lhs_shared = memref.alloc() : memref<1x2x8x64x8xf16, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x4x4x64x8xf16, #gpu.address_space<workgroup>>

  scf.forall (%id) in (1024) {
    %delin:3 = affine.delinearize_index %id into (2, 8, 64) : index, index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %delin#2, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : tensor<?x2x8x64x8xf16> to tensor<1x1x1x1x8xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true]} : tensor<1x1x1x1x8xf16>, vector<1x1x1x1x8xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %delin#0, %delin#1, %delin#2, %c0] {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x8xf16>, memref<1x2x8x64x8xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (1024) {
    %delin:3 = affine.delinearize_index %id into (4, 4, 64) : index, index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %delin#2, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : tensor<?x4x4x64x8xf16> to tensor<1x1x1x1x8xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true]} : tensor<1x1x1x1x8xf16>, vector<1x1x1x1x8xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %delin#0, %delin#1, %delin#2, %c0] {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x8xf16>,  memref<1x4x4x64x8xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

  gpu.barrier

  %0 = tensor.empty() : tensor<1x1x2x4x8x4x4x16x4xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x1x2x4x8x4x4x16x4xf32> {
    %ids:3 = affine.delinearize_index %id into (2, 4, 64) : index, index, index
    %threads:2 = affine.delinearize_index %ids#2 into (4, 16) : index, index

    %glb_lhs:2 = affine.delinearize_index %id into (8, 64) : index, index
    
    %glb0_rhs = arith.muli %ids#0, %c2 overflow<nsw, nuw> : index
    %glb1_rhs = arith.addi %glb0_rhs, %c1 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<8x4x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c1 to %dim step %c1 iter_args(%iter = %2) -> vector<8x4x1x4xf32> {

      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true]} : memref<1x2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x8x1x4xf16>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true]} : memref<1x4x4x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x4x1x4xf16>
      %lhs_vec_0_cast = vector.shape_cast %lhs_vec_0 : vector<1x1x8x1x4xf16> to vector<1x8x1x4xf16>
      %rhs_vec_0_cast = vector.shape_cast %rhs_vec_0 : vector<1x1x4x1x4xf16> to vector<1x4x1x4xf16>
      %lhs_vec_0_t = vector.transpose %lhs_vec_0_cast, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_0_t = vector.transpose %rhs_vec_0_cast, [1, 2, 0, 3] : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

      rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%i, %glb0_rhs, %ids#1, %ids#2, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] :  tensor<?x4x4x64x8xf16> to tensor<1x1x1x1x8xf16>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true]} : tensor<1x1x1x1x8xf16>, vector<1x1x1x1x8xf16>
      %rhs_thread_1 = tensor.extract_slice %rhs [%i, %glb1_rhs, %ids#1, %ids#2, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : tensor<?x4x4x64x8xf16> to tensor<1x1x1x1x8xf16>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true]} : tensor<1x1x1x1x8xf16>, vector<1x1x1x1x8xf16>

      rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %ids#0, %c0, %ids#2, %c4], %cst {in_bounds = [true, true, true, true, true]} : memref<1x2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x8x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %ids#1, %c0, %ids#2, %c4], %cst {in_bounds = [true, true, true, true, true]} : memref<1x4x4x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x4x1x4xf16>
      %lhs_vec_2_cast = vector.shape_cast %lhs_vec_2 : vector<1x1x8x1x4xf16> to vector<1x8x1x4xf16>
      %rhs_vec_2_cast = vector.shape_cast %rhs_vec_2 : vector<1x1x4x1x4xf16> to vector<1x4x1x4xf16>
      %lhs_vec_2_t = vector.transpose %lhs_vec_2_cast, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_2_t = vector.transpose %rhs_vec_2_cast, [1, 2, 0, 3] : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

      rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %c0, %glb_lhs#0, %glb_lhs#1, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : tensor<?x2x8x64x8xf16> to tensor<1x1x1x1x8xf16>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true]} : tensor<1x1x1x1x8xf16>, vector<1x1x1x1x8xf16>
      %lhs_thread_1 = tensor.extract_slice %lhs [%i, %c1, %glb_lhs#0, %glb_lhs#1, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : tensor<?x2x8x64x8xf16> to tensor<1x1x1x1x8xf16>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true]} : tensor<1x1x1x1x8xf16>, vector<1x1x1x1x8xf16>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %glb0_rhs, %ids#1, %ids#2, %c0] {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x8xf16>, memref<1x4x4x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %glb1_rhs, %ids#1, %ids#2, %c0] {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x8xf16>, memref<1x4x4x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%c0, %c0, %glb_lhs#0, %glb_lhs#1, %c0] {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x8xf16>, memref<1x2x8x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%c0, %c1, %glb_lhs#0, %glb_lhs#1, %c0] {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x8xf16>, memref<1x2x8x64x8xf16, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<8x4x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true]} : memref<1x2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x8x1x4xf16>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true]} : memref<1x4x4x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x4x1x4xf16>
    %lhs_vec_0_cast = vector.shape_cast %lhs_vec_0 : vector<1x1x8x1x4xf16> to vector<1x8x1x4xf16>
    %rhs_vec_0_cast = vector.shape_cast %rhs_vec_0 : vector<1x1x4x1x4xf16> to vector<1x4x1x4xf16>
    %lhs_vec_0_t = vector.transpose %lhs_vec_0_cast, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
    %rhs_vec_0_t = vector.transpose %rhs_vec_0_cast, [1, 2, 0, 3] : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %ids#0, %c0, %ids#2, %c4], %cst {in_bounds = [true, true, true, true, true]} : memref<1x2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x8x1x4xf16>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %ids#1, %c0, %ids#2, %c4], %cst {in_bounds = [true, true, true, true, true]} : memref<1x4x4x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x4x1x4xf16>
    %lhs_vec_2_cast = vector.shape_cast %lhs_vec_2 : vector<1x1x8x1x4xf16> to vector<1x8x1x4xf16>
    %rhs_vec_2_cast = vector.shape_cast %rhs_vec_2 : vector<1x1x4x1x4xf16> to vector<1x4x1x4xf16>
    %lhs_vec_2_t = vector.transpose %lhs_vec_2_cast, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
    %rhs_vec_2_t = vector.transpose %rhs_vec_2_cast, [1, 2, 0, 3] : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>
 
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x1x8x4x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<8x4x1x4xf32> to vector<1x1x1x1x8x4x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true, true]} : vector<1x1x1x1x8x4x1x1x4xf32>, tensor<1x1x1x1x8x4x1x1x4xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#0, %ids#1, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 1, 8, 4, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x8x4x1x1x4xf32> into tensor<1x1x2x4x8x4x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x2x4x8x4x4x16x4xf32>
}

util.func @pingpong_dt_medium_f16_subgroup_m2_n4_intrinsics_k1_n4(%lhs_base: tensor<1x?x2x8x4x4x4x4xf16>, %rhs_base: tensor<1x?x4x4x4x16x4xf16>, %unused_acc: tensor<1x1x2x4x8x4x4x16x4xf32>) -> tensor<1x1x2x4x8x4x4x16x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f16

  %dim = tensor.dim %rhs_base, %c1 : tensor<1x?x4x4x4x16x4xf16>
  %nDim =  arith.divui %dim, %c2 : index

  %lhs_expand = tensor.expand_shape %lhs_base [[0], [1, 2], [3], [4], [5], [6], [7], [8]] output_shape [1, %nDim, 2, 2, 8, 4, 4, 4, 4] : tensor<1x?x2x8x4x4x4x4xf16> into tensor<1x?x2x2x8x4x4x4x4xf16>
  %rhs_expand = tensor.expand_shape rlhs_base [[0], [1, 2], [3], [4], [5], [6], [7]] output_shape [1, %nDim, 2, 4, 4, 4, 16, 4] : tensor<1x?x4x4x4x16x4xf16> into tensor<1x?x2x4x4x4x16x4xf16>

  %lhs = tensor.collapse_shape %lhs_expand [[0, 1], [2], [3], [4], [5, 6, 7], [8]] : tensor<1x?x2x2x8x4x4x4x4xf16> into tensor<?x2x2x8x64x4xf16>
  %rhs = tensor.collapse_shape %rhs_expand [[0, 1], [2], [3], [4], [5, 6], [7]] : tensor<1x?x2x4x4x4x16x4xf16> into tensor<?x2x4x4x64x4xf16>

  %lhs_shared = memref.alloc() : memref<1x2x2x8x64x4xf16, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x2x4x4x64x4xf16, #gpu.address_space<workgroup>>

  scf.forall (%id) in (1024) {
    %delin:4 = affine.delinearize_index %id into (2, 2, 8, 32) : index, index, index, index
    %inner = arith.muli %delin#3, %c2 overflow<nsw, nuw> : index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %delin#2, %inner, %c0] [1, 1, 1, 1, 2, 4] [1, 1, 1, 1, 1, 1] : tensor<?x2x2x8x64x4xf16> to tensor<1x1x1x1x2x4xf16>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x4xf16>, vector<1x1x1x1x2x4xf16>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %delin#0, %delin#1, %delin#2, %inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x4xf16>, memref<1x2x2x8x64x4xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (1024) {
    %delin:4 = affine.delinearize_index %id into (2, 4, 4, 32) : index, index, index, index
    %inner = arith.muli %delin#3, %c2 overflow<nsw, nuw> : index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %delin#2, %inner, %c0] [1, 1, 1, 1, 2, 4] [1, 1, 1, 1, 1, 1] : tensor<?x2x4x4x64x4xf16> to tensor<1x1x1x1x2x4xf16>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x4xf16>, vector<1x1x1x1x2x4xf16>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %delin#0, %delin#1, %delin#2, %inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x4xf16>,  memref<1x2x4x4x64x4xf16, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

  gpu.barrier

  %0 = tensor.empty() : tensor<1x1x2x4x8x4x4x16x4xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x1x2x4x8x4x4x16x4xf32> {
    %ids:3 = affine.delinearize_index %id into (2, 4, 64) : index, index, index
    %threads:2 = affine.delinearize_index %ids#2 into (4, 16) : index, index

    %glb_lhs:2 = affine.delinearize_index %id into (8, 64) : index, index
    
    %glb0_rhs = arith.muli %ids#0, %c2 overflow<nsw, nuw> : index
    %glb1_rhs = arith.addi %glb0_rhs, %c1 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<8x4x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c1 to %dim step %c1 iter_args(%iter = %2) -> vector<8x4x1x4xf32> {

      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true]} : memref<1x2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x8x1x4xf16>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true]} : memref<1x4x4x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x4x1x4xf16>
      %lhs_vec_0_cast = vector.shape_cast %lhs_vec_0 : vector<1x1x8x1x4xf16> to vector<1x8x1x4xf16>
      %rhs_vec_0_cast = vector.shape_cast %rhs_vec_0 : vector<1x1x4x1x4xf16> to vector<1x4x1x4xf16>
      %lhs_vec_0_t = vector.transpose %lhs_vec_0_cast, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_0_t = vector.transpose %rhs_vec_0_cast, [1, 2, 0, 3] : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

      rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%i, %glb0_rhs, %ids#1, %ids#2, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] :  tensor<?x4x4x64x8xf16> to tensor<1x1x1x1x8xf16>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true]} : tensor<1x1x1x1x8xf16>, vector<1x1x1x1x8xf16>
      %rhs_thread_1 = tensor.extract_slice %rhs [%i, %glb1_rhs, %ids#1, %ids#2, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : tensor<?x4x4x64x8xf16> to tensor<1x1x1x1x8xf16>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true]} : tensor<1x1x1x1x8xf16>, vector<1x1x1x1x8xf16>

      rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %ids#0, %c0, %ids#2, %c4], %cst {in_bounds = [true, true, true, true, true]} : memref<1x2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x8x1x4xf16>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %ids#1, %c0, %ids#2, %c4], %cst {in_bounds = [true, true, true, true, true]} : memref<1x4x4x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x4x1x4xf16>
      %lhs_vec_2_cast = vector.shape_cast %lhs_vec_2 : vector<1x1x8x1x4xf16> to vector<1x8x1x4xf16>
      %rhs_vec_2_cast = vector.shape_cast %rhs_vec_2 : vector<1x1x4x1x4xf16> to vector<1x4x1x4xf16>
      %lhs_vec_2_t = vector.transpose %lhs_vec_2_cast, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
      %rhs_vec_2_t = vector.transpose %rhs_vec_2_cast, [1, 2, 0, 3] : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

      rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %c0, %glb_lhs#0, %glb_lhs#1, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : tensor<?x2x8x64x8xf16> to tensor<1x1x1x1x8xf16>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true]} : tensor<1x1x1x1x8xf16>, vector<1x1x1x1x8xf16>
      %lhs_thread_1 = tensor.extract_slice %lhs [%i, %c1, %glb_lhs#0, %glb_lhs#1, %c0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : tensor<?x2x8x64x8xf16> to tensor<1x1x1x1x8xf16>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true]} : tensor<1x1x1x1x8xf16>, vector<1x1x1x1x8xf16>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %glb0_rhs, %ids#1, %ids#2, %c0] {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x8xf16>, memref<1x4x4x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %glb1_rhs, %ids#1, %ids#2, %c0] {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x8xf16>, memref<1x4x4x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%c0, %c0, %glb_lhs#0, %glb_lhs#1, %c0] {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x8xf16>, memref<1x2x8x64x8xf16, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%c0, %c1, %glb_lhs#0, %glb_lhs#1, %c0] {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x8xf16>, memref<1x2x8x64x8xf16, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
      } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<8x4x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true]} : memref<1x2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x8x1x4xf16>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true]} : memref<1x4x4x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x4x1x4xf16>
    %lhs_vec_0_cast = vector.shape_cast %lhs_vec_0 : vector<1x1x8x1x4xf16> to vector<1x8x1x4xf16>
    %rhs_vec_0_cast = vector.shape_cast %rhs_vec_0 : vector<1x1x4x1x4xf16> to vector<1x4x1x4xf16>
    %lhs_vec_0_t = vector.transpose %lhs_vec_0_cast, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
    %rhs_vec_0_t = vector.transpose %rhs_vec_0_cast, [1, 2, 0, 3] : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %ids#0, %c0, %ids#2, %c4], %cst {in_bounds = [true, true, true, true, true]} : memref<1x2x8x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x8x1x4xf16>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %ids#1, %c0, %ids#2, %c4], %cst {in_bounds = [true, true, true, true, true]} : memref<1x4x4x64x8xf16, #gpu.address_space<workgroup>>, vector<1x1x4x1x4xf16>
    %lhs_vec_2_cast = vector.shape_cast %lhs_vec_2 : vector<1x1x8x1x4xf16> to vector<1x8x1x4xf16>
    %rhs_vec_2_cast = vector.shape_cast %rhs_vec_2 : vector<1x1x4x1x4xf16> to vector<1x4x1x4xf16>
    %lhs_vec_2_t = vector.transpose %lhs_vec_2_cast, [1, 2, 0, 3] : vector<1x8x1x4xf16> to vector<8x1x1x4xf16>
    %rhs_vec_2_t = vector.transpose %rhs_vec_2_cast, [1, 2, 0, 3] : vector<1x4x1x4xf16> to vector<4x1x1x4xf16>
 
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    } : vector<8x1x1x4xf16>, vector<4x1x1x4xf16> into vector<8x4x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x1x8x4x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<8x4x1x4xf32> to vector<1x1x1x1x8x4x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true, true]} : vector<1x1x1x1x8x4x1x1x4xf32>, tensor<1x1x1x1x8x4x1x1x4xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#0, %ids#1, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 1, 8, 4, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x8x4x1x1x4xf32> into tensor<1x1x2x4x8x4x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x2x4x8x4x4x16x4xf32>
}