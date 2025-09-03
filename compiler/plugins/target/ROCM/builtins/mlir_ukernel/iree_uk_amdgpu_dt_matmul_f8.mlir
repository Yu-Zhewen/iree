//  RUN: iree-opt %s

!in_ty_f8 = tensor<1x?x4x2x4x16x2x8xf8E4M3FNUZ>
!exp_in_ty_f8 = tensor<1x256x?xf8E4M3FNUZ>
!block_in_f8 = tensor<2x4x2x4x16x2x8xf8E4M3FNUZ>
!exp_block_in_f8 = tensor<1x256x128xf8E4M3FNUZ>
!flat_shared_f8 = memref<16384xf8E4M3FNUZ, #gpu.address_space<workgroup>>
!shared_f8 = memref<1x1x4x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
!shared_exp_f8 = memref<16x16x4x32xf8E4M3FNUZ, #gpu.address_space<workgroup>>

!mexp_in_ty_f8 = tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ>
!mexp_block_in_f8 = tensor<2x8x4x4x4x2x8xf8E4M3FNUZ>
!mflat_shared_f8 = memref<16384xf8E4M3FNUZ, #gpu.address_space<workgroup>>
!mshared_f8 = memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
!mshared_exp_f8 = memref<8x16x4x32xf8E4M3FNUZ, #gpu.address_space<workgroup>>

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]

util.func @pingpong_large_f8_expanded_data_tiling(%lhs_base: tensor<1x?x2x8x4x4x4x8xf8E4M3FNUZ>, %rhs_base: tensor<1x?x4x4x4x16x8xf8E4M3FNUZ>, %unused_acc: tensor<1x1x2x4x8x4x4x16x4xf32>) -> tensor<1x1x2x4x8x4x4x16x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f8E4M3FNUZ

  %dim = tensor.dim %rhs_base, %c1 : tensor<1x?x4x4x4x16x8xf8E4M3FNUZ>
  %nDim = arith.divui %dim, %c4 : index

  %lhs_expand = tensor.expand_shape %lhs_base [[0], [1, 2], [3], [4], [5], [6], [7], [8]] output_shape [1, %nDim, 2, 2, 8, 4, 4, 4, 8] : tensor<1x?x2x8x4x4x4x8xf8E4M3FNUZ> into tensor<1x?x4x2x8x4x4x4x8xf8E4M3FNUZ>
  %rhs_expand = tensor.expand_shape %rhs_base [[0], [1, 2], [3], [4], [5], [6], [7]] output_shape [1, %nDim, 2, 4, 4, 4, 16, 8] : tensor<1x?x4x4x4x16x8xf8E4M3FNUZ> into tensor<1x?x4x4x4x4x16x8xf8E4M3FNUZ>

  %lhs = tensor.collapse_shape %lhs_expand [[0, 1], [2], [3], [4], [5, 6, 7], [8]] : tensor<1x?x4x2x8x4x4x4x8xf8E4M3FNUZ> into tensor<?x4x2x8x64x8xf8E4M3FNUZ>
  %rhs = tensor.collapse_shape %rhs_expand [[0, 1], [2], [3], [4], [5, 6], [7]] : tensor<1x?x4x4x4x4x16x8xf8E4M3FNUZ> into tensor<?x4x4x4x64x8xf8E4M3FNUZ>

  %lhs_shared = memref.alloc() : memref<1x4x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x4x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>

  scf.forall (%id) in (2048) {
    %delin:4 = affine.delinearize_index %id into (4, 2, 8, 32) : index, index, index, index
    %inner = arith.muli %delin#3, %c2 overflow<nsw, nuw> : index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %delin#2, %inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1]  : tensor<?x4x2x8x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %delin#0, %delin#1, %delin#2, %inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x4x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (2048) {
    %delin:4 = affine.delinearize_index %id into (4, 4, 4, 32) : index, index, index, index
    %inner = arith.muli %delin#3, %c2 overflow<nsw, nuw> : index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %delin#2, %inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1] : tensor<?x4x4x4x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %delin#0, %delin#1, %delin#2, %inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x4x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %0 = tensor.empty() : tensor<1x1x2x4x8x4x4x16x4xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x1x2x4x8x4x4x16x4xf32> {
    %ids:3 = affine.delinearize_index %id into (2, 4, 64) : index, index, index
    %threads:2 = affine.delinearize_index %ids#2 into (4, 16) : index, index

    %glb_rhs:3 = affine.delinearize_index %id into (4, 4, 32) : index, index, index
    %glb_rhs_inner = arith.muli %glb_rhs#2, %c2 overflow<nsw, nuw> : index

    %glb_lhs:3 = affine.delinearize_index %id into (2, 8, 32) : index, index, index
    %glb_lhs_inner = arith.muli %glb_lhs#2, %c2 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<8x4x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }

    %3 = scf.for %i = %c1 to %nDim step %c1 iter_args(%iter = %2) -> vector<8x4x1x4xf32> {
      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %c0, %glb_lhs#0, %glb_lhs#1, %glb_lhs_inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1] :  tensor<?x4x2x8x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_thread_1 = tensor.extract_slice %lhs [%i, %c1, %glb_lhs#0, %glb_lhs#1, %glb_lhs_inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1] : tensor<?x4x2x8x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_thread_2 = tensor.extract_slice %lhs [%i, %c2, %glb_lhs#0, %glb_lhs#1, %glb_lhs_inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1] :  tensor<?x4x2x8x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_vec_local_2 = vector.transfer_read %lhs_thread_2 [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_thread_3 = tensor.extract_slice %lhs [%i, %c3, %glb_lhs#0, %glb_lhs#1, %glb_lhs_inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1] : tensor<?x4x2x8x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_vec_local_3 = vector.transfer_read %lhs_thread_3 [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>
      
      // Local loads.
      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x8x1x8xf8E4M3FNUZ>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %c0, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x4x1x8xf8E4M3FNUZ>
      %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x1x1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x1x1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%i, %c0, %glb_rhs#0, %glb_rhs#1, %glb_rhs_inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1] :  tensor<?x4x4x4x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_thread_1 = tensor.extract_slice %rhs [%i, %c1, %glb_rhs#0, %glb_rhs#1, %glb_rhs_inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1] : tensor<?x4x4x4x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_thread_2 = tensor.extract_slice %rhs [%i, %c2, %glb_rhs#0, %glb_rhs#1, %glb_rhs_inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1] :  tensor<?x4x4x4x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_vec_local_2 = vector.transfer_read %rhs_thread_2 [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_thread_3 = tensor.extract_slice %rhs [%i, %c3, %glb_rhs#0, %glb_rhs#1, %glb_rhs_inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1] : tensor<?x4x4x4x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_vec_local_3 = vector.transfer_read %rhs_thread_3 [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>

      // Local loads.
      %lhs_vec_1 = vector.transfer_read %lhs_shared[%c0, %c1, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x8x1x8xf8E4M3FNUZ>
      %rhs_vec_1 = vector.transfer_read %rhs_shared[%c0, %c1, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x4x1x8xf8E4M3FNUZ>
      %lhs_vec_1_t = vector.shape_cast %lhs_vec_1 : vector<1x1x1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_1_t = vector.shape_cast %rhs_vec_1 : vector<1x1x1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot1 = iree_codegen.inner_tiled ins(%lhs_vec_1_t, %rhs_vec_1_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      // Local loads.
      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c2, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x8x1x8xf8E4M3FNUZ>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %c2, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x4x1x8xf8E4M3FNUZ>
      %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x1x1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x1x1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

      // Local loads.
      %lhs_vec_3 = vector.transfer_read %lhs_shared[%c0, %c3, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x8x1x8xf8E4M3FNUZ>
      %rhs_vec_3 = vector.transfer_read %rhs_shared[%c0, %c3, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x4x1x8xf8E4M3FNUZ>
      %lhs_vec_3_t = vector.shape_cast %lhs_vec_3 : vector<1x1x1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_3_t = vector.shape_cast %rhs_vec_3 : vector<1x1x1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot1) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %c0, %glb_rhs#0, %glb_rhs#1, %glb_rhs_inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x4x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %c1, %glb_rhs#0, %glb_rhs#1, %glb_rhs_inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x4x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_2, %rhs_shared [%c0, %c2, %glb_rhs#0, %glb_rhs#1, %glb_rhs_inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x4x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_3, %rhs_shared [%c0, %c3, %glb_rhs#0, %glb_rhs#1, %glb_rhs_inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x4x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
        
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%c0, %c0, %glb_lhs#0, %glb_lhs#1, %glb_lhs_inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x4x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%c0, %c1, %glb_lhs#0, %glb_lhs#1, %glb_lhs_inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x4x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_2, %lhs_shared [%c0, %c2, %glb_lhs#0, %glb_lhs#1, %glb_lhs_inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x4x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_3, %lhs_shared [%c0, %c3, %glb_lhs#0, %glb_lhs#1, %glb_lhs_inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x4x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot3 = iree_codegen.inner_tiled ins(%lhs_vec_3_t, %rhs_vec_3_t) outs(%dot2) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot3 : vector<8x4x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x8x1x8xf8E4M3FNUZ>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %c0, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x4x1x8xf8E4M3FNUZ>
    %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x1x1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x1x1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

    %lhs_vec_1 = vector.transfer_read %lhs_shared[%c0, %c1, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x8x1x8xf8E4M3FNUZ>
    %rhs_vec_1 = vector.transfer_read %rhs_shared[%c0, %c1, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x4x1x8xf8E4M3FNUZ>
    %lhs_vec_1_t = vector.shape_cast %lhs_vec_1 : vector<1x1x1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_1_t = vector.shape_cast %rhs_vec_1 : vector<1x1x1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

    %dot1 = iree_codegen.inner_tiled ins(%lhs_vec_1_t, %rhs_vec_1_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c2, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x8x1x8xf8E4M3FNUZ>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %c2, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x4x1x8xf8E4M3FNUZ>
    %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x1x1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x1x1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot1) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

    %lhs_vec_3 = vector.transfer_read %lhs_shared[%c0, %c3, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x8x1x8xf8E4M3FNUZ>
    %rhs_vec_3 = vector.transfer_read %rhs_shared[%c0, %c3, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x4x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x4x1x8xf8E4M3FNUZ>
    %lhs_vec_3_t = vector.shape_cast %lhs_vec_3 : vector<1x1x1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_3_t = vector.shape_cast %rhs_vec_3 : vector<1x1x1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

    %dot3 = iree_codegen.inner_tiled ins(%lhs_vec_3_t, %rhs_vec_3_t) outs(%dot2) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x1x8x4x1x1x4xf32>
    %cast = vector.shape_cast %dot3 : vector<8x4x1x4xf32> to vector<1x1x1x1x8x4x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true, true]} : vector<1x1x1x1x8x4x1x1x4xf32>, tensor<1x1x1x1x8x4x1x1x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#0, %ids#1, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 1, 8, 4, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x8x4x1x1x4xf32> into tensor<1x1x2x4x8x4x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x2x4x8x4x4x16x4xf32>
}

util.func @pingpong_dt_medium_f8_expanded_3(%lhs: !mexp_in_ty_f8, %rhs: !in_ty_f8, %unused_acc: tensor<1x1x4x8x2x4x16x4xf32>) -> tensor<1x1x4x8x2x4x16x4xf32> {
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
  %cst = arith.constant 0.0 : f8E4M3FNUZ
  // %lhs_shared_base = memref.alloc() : !mflat_shared_f8
  // %rhs_shared_base = memref.alloc() : !flat_shared_f8
  %lhs_shared = memref.alloc() : memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x1x4x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>

  %dim = tensor.dim %rhs, %c1 : !in_ty_f8

  // %lhs_shared = memref.expand_shape %lhs_shared_base [[0, 1, 2, 3, 4, 5, 6]] output_shape [2, 8, 4, 4, 4, 2, 8] : !mflat_shared_f8 into !mshared_f8
  // %rhs_shared = memref.expand_shape %rhs_shared_base [[0, 1, 2, 3, 4, 5, 6]] output_shape [2, 4, 2, 4, 16, 2, 8] : !flat_shared_f8 into !shared_f8

  // %lhs_init = tensor.extract_slice %lhs [0, 0, 0, 0, 0, 0, 0, 0] [1, 2, 8, 4, 4, 4, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : !mexp_in_ty_f8 to !mexp_block_in_f8
  // %rhs_init = tensor.extract_slice %rhs [0, 0, 0, 0, 0, 0, 0, 0] [1, 2, 4, 2, 4, 16, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : !in_ty_f8 to !block_in_f8

  scf.forall (%id) in (512) {
    %delin:4 = affine.delinearize_index %id into (8, 4, 4, 4) : index, index, index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : !mexp_in_ty_f8 to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, !mshared_f8
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (512) {
    %delin:4 = affine.delinearize_index %id into (4, 2, 4, 16) : index, index, index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : !in_ty_f8 to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, !shared_f8
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %0 = tensor.empty() : tensor<1x1x4x8x2x4x16x4xf32>
  %1 = scf.forall (%id) in (256) shared_outs(%out = %0) -> tensor<1x1x4x8x2x4x16x4xf32> {
    %ids_lhs:4 = affine.delinearize_index %id into (4, 4, 4, 4) : index, index, index, index
    %ids_rhs:3 = affine.delinearize_index %id into (4, 4, 16) : index, index, index

    // %ids:4 = affine.delinearize_index %id into (2, 4, 4, 16) : index, index, index, index
    // %inner_id = arith.muli %ids#2, %c8 overflow<nsw, nuw> : index
    // %inner_id_acc = arith.muli %ids#2, %c4 overflow<nsw, nuw> : index
    // %m_outer_id = arith.muli %ids#0, %c4 overflow<nsw, nuw> : index
    // %n_outer_id = arith.muli %ids#1, %c4 overflow<nsw, nuw> : index
    // %delin:2 = affine.delinearize_index %id into (64, 8) : index, index
    // %wt:3 = affine.delinearize_index %id into (8, 8, 8) : index, index, index

    // // Inner 64 loads 8 threads x 16 elements.
    // %gko = arith.muli %wt#2, %c16 overflow<nsw, nuw> : index
    // // RHS indexing. Each subgroup loads 32 contiguous rows out of 256.
    // %bpo = arith.muli %wt#0, %c32 overflow<nsw, nuw> : index
    // // Base index is remaining outer 8 lanes + subgroup base.
    // %glb0 = arith.addi %wt#1, %bpo overflow<nsw, nuw> : index
    // %glb1 = arith.addi %glb0, %c8 overflow<nsw, nuw> : index
    // %glb2 = arith.addi %glb1, %c8 overflow<nsw, nuw> : index
    // %glb3 = arith.addi %glb2, %c8 overflow<nsw, nuw> : index
    // // LHS indexing.
    // %bpo_lhs = arith.muli %wt#0, %c16 overflow<nsw, nuw> : index
    // %glb0_lhs = arith.addi %wt#1, %bpo_lhs overflow<nsw, nuw> : index
    // %glb1_lhs = arith.addi %glb0_lhs, %c8 overflow<nsw, nuw> : index

    %glb_rhs:5 = affine.delinearize_index %id into (2, 4, 2, 4, 4) : index, index, index, index, index
    %glb_rhs0 = arith.muli %glb_rhs#4, %c4 overflow<nsw, nuw> : index
    %glb_rhs1 = arith.addi %glb_rhs0, %c1 overflow<nsw, nuw> : index
    %glb_rhs2 = arith.addi %glb_rhs0, %c2 overflow<nsw, nuw> : index
    %glb_rhs3 = arith.addi %glb_rhs0, %c3 overflow<nsw, nuw> : index

    %glb_lhs:4 = affine.delinearize_index %id into (2, 8, 4, 4) : index, index, index, index

    %2 = arith.constant dense<0.0> : vector<8x2x1x4xf32>
    // %2 = arith.constant dense<0.0> : vector<1x1x1x8x2x1x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c128 : index
    %cmp1 = arith.cmpi sge, %id, %c128 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
//     %3 = scf.for %i = %c2 to %dim step %c2 iter_args(%iter = %2) -> vector<8x2x1x1x1x1x1x4xf32> {
// 
//       // %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !mshared_f8, vector<1x8x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !mshared_f8, vector<1x8x1x1x1x2x8xf8E4M3FNUZ>
//       // %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !shared_f8, vector<1x1x2x1x1x2x8xf8E4M3FNUZ>
//       %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !shared_f8, vector<1x1x2x1x1x2x8xf8E4M3FNUZ>
//       %lhs_vec_0_t = vector.transpose %lhs_vec_0, [1, 5, 0, 2, 3, 4, 6] : vector<1x8x1x1x1x2x8xf8E4M3FNUZ> to vector<8x2x1x1x1x1x8xf8E4M3FNUZ>
//       %rhs_vec_0_t = vector.transpose %rhs_vec_0, [2, 5, 0, 1, 3, 4, 6] : vector<1x1x2x1x1x2x8xf8E4M3FNUZ> to vector<2x2x1x1x1x1x8xf8E4M3FNUZ>
// 
//       rocdl.sched.barrier 0
// 
//       // Global loads of rhs.
//       %rhs_block = tensor.extract_slice %rhs [0, %i, 0, 0, 0, 0, 0, 0] [1, 2, 4, 2, 4, 16, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : !in_ty_f8 to !block_in_f8
//       %rhs_thread_0 = tensor.extract_slice %rhs_block [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs0, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_thread_1 = tensor.extract_slice %rhs_block [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs1, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_thread_2 = tensor.extract_slice %rhs_block [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs2, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_vec_local_2 = vector.transfer_read %rhs_thread_2 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_thread_3 = tensor.extract_slice %rhs_block [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs3, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_vec_local_3 = vector.transfer_read %rhs_thread_3 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
// 
//       rocdl.sched.barrier 0
// 
//       // %lhs_vec_2 = vector.transfer_read %lhs_shared[%c1, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !mshared_f8, vector<1x8x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_vec_2 = vector.transfer_read %lhs_shared[%c1, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !mshared_f8, vector<1x8x1x1x1x2x8xf8E4M3FNUZ>
//       // %rhs_vec_2 = vector.transfer_read %rhs_shared[%c1, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !shared_f8, vector<1x1x2x1x1x2x8xf8E4M3FNUZ>
//       %rhs_vec_2 = vector.transfer_read %rhs_shared[%c1, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !shared_f8, vector<1x1x2x1x1x2x8xf8E4M3FNUZ>
//       %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 5, 0, 2, 3, 4, 6] : vector<1x8x1x1x1x2x8xf8E4M3FNUZ> to vector<8x2x1x1x1x1x8xf8E4M3FNUZ>
//       %rhs_vec_2_t = vector.transpose %rhs_vec_2, [2, 5, 0, 1, 3, 4, 6] : vector<1x1x2x1x1x2x8xf8E4M3FNUZ> to vector<2x2x1x1x1x1x8xf8E4M3FNUZ>
// 
//       rocdl.sched.barrier 0
// 
//       // Global loads of lhs.
//       %lhs_block = tensor.extract_slice %lhs [0, %i, 0, 0, 0, 0, 0, 0] [1, 2, 8, 4, 4, 4, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1]  : !mexp_in_ty_f8 to !mexp_block_in_f8
//       %lhs_thread_0 = tensor.extract_slice %lhs_block [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c0, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !mexp_block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_thread_1 = tensor.extract_slice %lhs_block [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c1, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !mexp_block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_thread_2 = tensor.extract_slice %lhs_block [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c2, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !mexp_block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_vec_local_2 = vector.transfer_read %lhs_thread_2 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_thread_3 = tensor.extract_slice %lhs_block [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c3, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !mexp_block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_vec_local_3 = vector.transfer_read %lhs_thread_3 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
// 
//       gpu.barrier
//       rocdl.sched.barrier 0
//       rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }
// 
//       %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
//         indexing_maps = #contraction_accesses,
//         iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
//         kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
//       } : vector<8x2x1x1x1x1x8xf8E4M3FNUZ>, vector<2x2x1x1x1x1x8xf8E4M3FNUZ> into vector<8x2x1x1x1x1x1x4xf32>
// //      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0, %rhs_vec_0) outs(%iter) {
// //        indexing_maps = #contraction_accesses,
// //        iterator_types = [],
// //        kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2>
// //      } : vector<1x8x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x2x1x1x2x8xf8E4M3FNUZ> into vector<1x1x1x8x2x1x1x4xf32>
// 
//       rocdl.s.setprio 0
//       gpu.barrier
//       rocdl.sched.barrier 0
// 
//       vector.transfer_write %rhs_vec_local_0, %rhs_shared [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !shared_f8
//       vector.transfer_write %rhs_vec_local_1, %rhs_shared [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs1, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !shared_f8
//       vector.transfer_write %rhs_vec_local_2, %rhs_shared [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs2, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !shared_f8
//       vector.transfer_write %rhs_vec_local_3, %rhs_shared [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !shared_f8
// 
//       vector.transfer_write %lhs_vec_local_0, %lhs_shared [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !mshared_f8
//       vector.transfer_write %lhs_vec_local_1, %lhs_shared [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c1, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !mshared_f8
//       vector.transfer_write %lhs_vec_local_2, %lhs_shared [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c2, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !mshared_f8
//       vector.transfer_write %lhs_vec_local_3, %lhs_shared [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !mshared_f8
// 
//       gpu.barrier
//       rocdl.sched.barrier 0
//       rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }
// 
//       %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
//         indexing_maps = #contraction_accesses,
//         iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
//         kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major = true>
//       } : vector<8x2x1x1x1x1x8xf8E4M3FNUZ>, vector<2x2x1x1x1x1x8xf8E4M3FNUZ> into vector<8x2x1x1x1x1x1x4xf32>
// //      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2, %rhs_vec_2) outs(%dot0) {
// //        indexing_maps = #contraction_accesses,
// //        iterator_types = [],
// //        kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2>
// //      } : vector<1x8x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x2x1x1x2x8xf8E4M3FNUZ> into vector<1x1x1x8x2x1x1x4xf32>
// 
//       rocdl.s.setprio 0
//       gpu.barrier
//       rocdl.sched.barrier 0
// 
//       scf.yield %dot2 : vector<8x2x1x1x1x1x1x4xf32>
//     }
//     scf.if %cmp1 {
//       rocdl.s.barrier
//     }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : !mshared_f8, vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : !shared_f8, vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ>
    %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%2) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : !mshared_f8, vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : !shared_f8, vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ>
    %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>
 
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x8x2x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<8x2x1x4xf32> to vector<1x1x1x8x2x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x8x2x1x1x4xf32>, tensor<1x1x1x8x2x1x1x4xf32>

    // %tp = vector.transpose %dot0, [2, 3, 4, 0, 1, 5, 6, 7] : vector<8x2x1x1x1x1x1x4xf32> to vector<1x1x1x8x2x1x1x4xf32>
    // %empty = tensor.empty() : tensor<1x1x1x8x2x1x1x4xf32>
    // %4 = vector.transfer_write %tp, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x8x2x1x1x4xf32>, tensor<1x1x1x8x2x1x1x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids_rhs#0, %c0, %c0, %ids_rhs#1, %ids_rhs#2, %c0] [1, 1, 1, 8, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x8x2x1x1x4xf32> into tensor<1x1x4x8x2x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x4x8x2x4x16x4xf32>
}

util.func @pingpong_dt_medium_f8_expanded(%lhs: !mexp_in_ty_f8, %rhs: !in_ty_f8, %unused_acc: tensor<1x1x4x8x2x4x16x4xf32>) -> tensor<1x1x4x8x2x4x16x4xf32> {
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
  %cst = arith.constant 0.0 : f8E4M3FNUZ
  %lhs_shared = memref.alloc() : memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x1x4x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>

  %dim = tensor.dim %rhs, %c1 : !in_ty_f8

  scf.forall (%id) in (512) {
    %delin:4 = affine.delinearize_index %id into (8, 4, 4, 4) : index, index, index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : !mexp_in_ty_f8 to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, !mshared_f8
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (512) {
    %delin:4 = affine.delinearize_index %id into (4, 2, 4, 16) : index, index, index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : !in_ty_f8 to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, !shared_f8
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %0 = tensor.empty() : tensor<1x1x4x8x2x4x16x4xf32>
  %1 = scf.forall (%id) in (256) shared_outs(%out = %0) -> tensor<1x1x4x8x2x4x16x4xf32> {
    %ids_lhs:4 = affine.delinearize_index %id into (4, 4, 4, 4) : index, index, index, index
    %ids_rhs:3 = affine.delinearize_index %id into (4, 4, 16) : index, index, index

    %glb_lhs = arith.addi %ids_lhs#0, %c4 overflow<nsw, nuw> : index

    // %glb_rhs:5 = affine.delinearize_index %id into (2, 4, 2, 4, 4) : index, index, index, index, index
    // %glb_rhs0 = arith.muli %glb_rhs#4, %c4 overflow<nsw, nuw> : index
    // %glb_rhs1 = arith.addi %glb_rhs0, %c1 overflow<nsw, nuw> : index
    // %glb_rhs2 = arith.addi %glb_rhs0, %c2 overflow<nsw, nuw> : index
    // %glb_rhs3 = arith.addi %glb_rhs0, %c3 overflow<nsw, nuw> : index

    // %glb_lhs:4 = affine.delinearize_index %id into (4, 4, 4, 4) : index, index, index, index

    %2 = arith.constant dense<0.0> : vector<8x2x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c128 : index
    %cmp1 = arith.cmpi sge, %id, %c128 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %3 = scf.for %i = %c1 to %dim step %c1 iter_args(%iter = %2) -> vector<8x2x1x4xf32> {

      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : !mshared_f8, vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : !shared_f8, vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ>
      %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

      //rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%c0, %i, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] :  tensor<1x?x4x2x4x16x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_thread_1 = tensor.extract_slice %rhs [%c0, %i, %ids_rhs#0, %c1, %ids_rhs#1, %ids_rhs#2, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x4x2x4x16x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>

      //rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : !mshared_f8, vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : !shared_f8, vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ>
      %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

      //rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%c0, %i, %ids_lhs#0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_thread_1 = tensor.extract_slice %lhs [%c0, %i, %glb_lhs, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x4x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %c0, %ids_rhs#0, %c1, %ids_rhs#1, %ids_rhs#2, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x4x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%c0, %c0, %ids_lhs#0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%c0, %c0, %glb_lhs, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<8x2x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : !mshared_f8, vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : !shared_f8, vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ>
    %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : !mshared_f8, vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : !shared_f8, vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ>
    %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>
 
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x8x2x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<8x2x1x4xf32> to vector<1x1x1x8x2x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x8x2x1x1x4xf32>, tensor<1x1x1x8x2x1x1x4xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids_rhs#0, %c0, %c0, %ids_rhs#1, %ids_rhs#2, %c0] [1, 1, 1, 8, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x8x2x1x1x4xf32> into tensor<1x1x4x8x2x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x4x8x2x4x16x4xf32>
}

util.func @pingpong_dt_medium_f8_expanded_subgroup_n8(%lhs: tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ>, %rhs: tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ>, %unused_acc: tensor<1x1x8x8x2x4x16x4xf32>) -> tensor<1x1x8x8x2x4x16x4xf32> {
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
  %cst = arith.constant 0.0 : f8E4M3FNUZ
  %lhs_shared = memref.alloc() : memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x1x8x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>

  %dim = tensor.dim %rhs, %c1 : tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ>

  scf.forall (%id) in (512) {
    %delin:4 = affine.delinearize_index %id into (8, 4, 4, 4) : index, index, index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (1024) {
    %delin:4 = affine.delinearize_index %id into (8, 2, 4, 16) : index, index, index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>,  memref<1x1x8x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
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

      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ>
      %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

      // rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%c0, %i, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] :  tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_thread_1 = tensor.extract_slice %rhs [%c0, %i, %ids_rhs#0, %c1, %ids_rhs#1, %ids_rhs#2, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>

      // rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ>
      %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

      // rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%c0, %i, %ids_lhs#0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      // %lhs_thread_1 = tensor.extract_slice %lhs [%c0, %i, %glb_lhs, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      // %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x8x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %c0, %ids_rhs#0, %c1, %ids_rhs#1, %ids_rhs#2, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x8x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%c0, %c0, %ids_lhs#0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      // vector.transfer_write %lhs_vec_local_1, %lhs_shared [%c0, %c0, %glb_lhs, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<8x2x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ>
    %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %c0, %ids_rhs#0, %c0, %ids_rhs#1, %ids_rhs#2, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ>
    %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>
 
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x8x2x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<8x2x1x4xf32> to vector<1x1x1x8x2x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x8x2x1x1x4xf32>, tensor<1x1x1x8x2x1x1x4xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids_rhs#0, %c0, %c0, %ids_rhs#1, %ids_rhs#2, %c0] [1, 1, 1, 8, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x8x2x1x1x4xf32> into tensor<1x1x8x8x2x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x8x8x2x4x16x4xf32>
}

util.func @pingpong_dt_medium_f8_expanded_subgroup_n8_reduce_dims(%lhs_base: tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ>, %rhs_base: tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ>, %unused_acc: tensor<1x1x8x8x2x4x16x4xf32>) -> tensor<1x1x8x8x2x4x16x4xf32> {
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
  %cst = arith.constant 0.0 : f8E4M3FNUZ

  %dim = tensor.dim %rhs_base, %c1 : tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ>

  %lhs = tensor.collapse_shape %lhs_base [[0, 1], [2], [3, 4, 5], [6, 7]] : tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ> into tensor<?x8x64x16xf8E4M3FNUZ>
  %rhs = tensor.collapse_shape %rhs_base [[0, 1], [2, 3], [4, 5], [6, 7]] : tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ> into tensor<?x16x64x16xf8E4M3FNUZ>

  %lhs_shared = memref.alloc() : memref<1x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>

  scf.forall (%id) in (512) {
    %delin:2 = affine.delinearize_index %id into (8, 64) : index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<?x8x64x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, memref<1x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (1024) {
    %delin:2 = affine.delinearize_index %id into (16, 64) : index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<?x16x64x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>,  memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

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

      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x8x1x8xf8E4M3FNUZ>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x2x1x8xf8E4M3FNUZ>
      %lhs_vec_0_t = vector.transpose %lhs_vec_0, [1, 2, 0, 3] : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x2x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

      // rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%i, %glb0_rhs, %ids#1, %c0] [1, 1, 1, 16] [1, 1, 1, 1] :  tensor<?x16x64x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %rhs_thread_1 = tensor.extract_slice %rhs [%i, %glb1_rhs, %ids#1, %c0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<?x16x64x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>

      // rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c8], %cst {in_bounds = [true, true, true, true]} : memref<1x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x8x1x8xf8E4M3FNUZ>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c8], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x2x1x8xf8E4M3FNUZ>
      %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

      // rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %ids#0, %ids#1, %c0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<?x8x64x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %glb0_rhs, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %glb1_rhs, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%c0, %ids#0, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, memref<1x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<8x2x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x8x1x8xf8E4M3FNUZ>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x2x1x8xf8E4M3FNUZ>
    %lhs_vec_0_t = vector.transpose %lhs_vec_0, [1, 2, 0, 3] : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x2x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c8], %cst {in_bounds = [true, true, true, true]} : memref<1x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x8x1x8xf8E4M3FNUZ>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#1, %c8], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x2x1x8xf8E4M3FNUZ>
    %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>
 
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x8x2x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<8x2x1x4xf32> to vector<1x1x1x8x2x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x8x2x1x1x4xf32>, tensor<1x1x1x8x2x1x1x4xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#0, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 8, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x8x2x1x1x4xf32> into tensor<1x1x8x8x2x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x8x8x2x4x16x4xf32>
}

util.func @pingpong_dt_medium_f8_expanded_subgroup_n8_intrinsic_m16(%lhs_base: tensor<1x?x16x4x4x4x2x8xf8E4M3FNUZ>, %rhs_base: tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ>, %unused_acc: tensor<1x1x8x16x2x4x16x4xf32>) -> tensor<1x1x8x16x2x4x16x4xf32> {
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
  %cst = arith.constant 0.0 : f8E4M3FNUZ

  %dim = tensor.dim %rhs_base, %c1 : tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ>

  %lhs = tensor.collapse_shape %lhs_base [[0, 1], [2], [3, 4, 5], [6, 7]] : tensor<1x?x16x4x4x4x2x8xf8E4M3FNUZ> into tensor<?x16x64x16xf8E4M3FNUZ>
  %rhs = tensor.collapse_shape %rhs_base [[0, 1], [2, 3], [4, 5], [6, 7]] : tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ> into tensor<?x16x64x16xf8E4M3FNUZ>

  %lhs_shared = memref.alloc() : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>

  scf.forall (%id) in (1024) {
    %delin:2 = affine.delinearize_index %id into (16, 64) : index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<?x16x64x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (1024) {
    %delin:2 = affine.delinearize_index %id into (16, 64) : index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<?x16x64x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>,  memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

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

      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x16x1x8xf8E4M3FNUZ>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x2x1x8xf8E4M3FNUZ>
      %lhs_vec_0_t = vector.transpose %lhs_vec_0, [1, 2, 0, 3] : vector<1x16x1x8xf8E4M3FNUZ> to vector<16x1x1x8xf8E4M3FNUZ>
      %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x2x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

      // rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%i, %glb0, %ids#1, %c0] [1, 1, 1, 16] [1, 1, 1, 1] :  tensor<?x16x64x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %rhs_thread_1 = tensor.extract_slice %rhs [%i, %glb1, %ids#1, %c0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<?x16x64x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>

      // rocdl.sched.barrier 0

      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c8], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x16x1x8xf8E4M3FNUZ>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c8], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x2x1x8xf8E4M3FNUZ>
      %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x16x1x8xf8E4M3FNUZ> to vector<16x1x1x8xf8E4M3FNUZ>
      %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

      // rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %glb0, %ids#1, %c0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<?x16x64x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %lhs_thread_1 = tensor.extract_slice %lhs [%i, %glb1, %ids#1, %c0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<?x16x64x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<16x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<16x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %glb0, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %glb1, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%c0, %glb0, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%c0, %glb1, %ids#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<16x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<16x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<16x2x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x16x1x8xf8E4M3FNUZ>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x2x1x8xf8E4M3FNUZ>
    %lhs_vec_0_t = vector.transpose %lhs_vec_0, [1, 2, 0, 3] : vector<1x16x1x8xf8E4M3FNUZ> to vector<16x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x2x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<16x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<16x2x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#1, %c8], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x16x1x8xf8E4M3FNUZ>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0, %ids#1, %c8], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x2x1x8xf8E4M3FNUZ>
    %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x16x1x8xf8E4M3FNUZ> to vector<16x1x1x8xf8E4M3FNUZ>
    %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>
 
    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<16x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<16x2x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x16x2x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<16x2x1x4xf32> to vector<1x1x1x16x2x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x16x2x1x1x4xf32>, tensor<1x1x1x16x2x1x1x4xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#0, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 16, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x16x2x1x1x4xf32> into tensor<1x1x8x16x2x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x8x16x2x4x16x4xf32>
}

util.func private @pingpong_dt_medium_f8_expanded_subgroup_n8_reduce_dims_zhewen(%lhs_base: tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ>, %rhs_base: tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ>, %unused_acc: tensor<1x1x8x8x2x4x16x4xf32>) -> tensor<1x1x8x8x2x4x16x4xf32> {
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
  %c4096 = arith.constant 4096 : index
  %cst = arith.constant 0.0 : f8E4M3FNUZ

  %dim = tensor.dim %rhs_base, %c1 : tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ>

  %lhs = tensor.collapse_shape %lhs_base [[0, 1], [2], [3, 4, 5], [6, 7]] : tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ> into tensor<?x8x64x16xf8E4M3FNUZ>
  %rhs = tensor.collapse_shape %rhs_base [[0, 1], [2, 3], [4, 5], [6, 7]] : tensor<1x?x8x2x4x16x2x8xf8E4M3FNUZ> into tensor<?x16x64x16xf8E4M3FNUZ>

  %lhs_shared = memref.alloc() : memref<1x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>

  scf.forall (%id) in (512) {
    %delin:2 = affine.delinearize_index %id into (8, 64) : index, index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<?x8x64x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, memref<1x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (1024) {
    %delin:2 = affine.delinearize_index %id into (16, 64) : index, index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %c0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<?x16x64x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %delin#0, %delin#1, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %0 = tensor.empty() : tensor<1x1x8x8x2x4x16x4xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x1x8x8x2x4x16x4xf32> {
    %ids:3 = affine.delinearize_index %id into (1, 8, 64) : index, index, index
    %threads:2 = affine.delinearize_index %ids#2 into (4, 16) : index, index

    %glb0_rhs = arith.muli %ids#1, %c2 overflow<nsw, nuw> : index
    %glb1_rhs = arith.addi %glb0_rhs, %c1 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<8x2x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }

    %3 = scf.for %i = %c1 to %dim step %c1 iter_args(%iter = %2) -> vector<8x2x1x4xf32> {
      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %ids#0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x8x1x8xf8E4M3FNUZ>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x2x1x8xf8E4M3FNUZ>
      %lhs_vec_0_t = vector.transpose %lhs_vec_0, [1, 2, 0, 3] : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x2x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

      // Global loads of rhs.
      %rhs_block = tensor.extract_slice %rhs [%i, %glb0_rhs, %ids#2, %c0] [1, 2, 1, 16] [1, 1, 1, 1] : tensor<?x16x64x16xf8E4M3FNUZ> to tensor<1x2x1x16xf8E4M3FNUZ>
      %rhs_thread_0 = tensor.extract_slice %rhs_block [%c0, %c0, %c0, %c0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x2x1x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>
      %rhs_thread_1 = tensor.extract_slice %rhs_block [%c0, %c1, %c0, %c0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x2x1x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>

      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %ids#0, %ids#2, %c8], %cst {in_bounds = [true, true, true, true]} : memref<1x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x8x1x8xf8E4M3FNUZ>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#2, %c8], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x2x1x8xf8E4M3FNUZ>
      %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %ids#1, %ids#2, %c0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<?x8x64x16xf8E4M3FNUZ> to tensor<1x1x1x16xf8E4M3FNUZ>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x1x16xf8E4M3FNUZ>, vector<1x1x1x16xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared[%c0, %glb0_rhs, %ids#2, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared[%c0, %glb1_rhs, %ids#2, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>

      vector.transfer_write %lhs_vec_local_0, %lhs_shared[%c0, %ids#1, %ids#2, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf8E4M3FNUZ>, memref<1x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<8x2x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %ids#0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x8x1x8xf8E4M3FNUZ>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#2, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x2x1x8xf8E4M3FNUZ>
    %lhs_vec_0_t = vector.transpose %lhs_vec_0, [1, 2, 0, 3] : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0_t = vector.transpose %rhs_vec_0, [1, 2, 0, 3] : vector<1x2x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %ids#0, %ids#2, %c8], %cst {in_bounds = [true, true, true, true]} : memref<1x8x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x8x1x8xf8E4M3FNUZ>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %glb0_rhs, %ids#2, %c8], %cst {in_bounds = [true, true, true, true]} : memref<1x16x64x16xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x2x1x8xf8E4M3FNUZ>
    %lhs_vec_2_t = vector.transpose %lhs_vec_2, [1, 2, 0, 3] : vector<1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_2_t = vector.transpose %rhs_vec_2, [1, 2, 0, 3] : vector<1x2x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x8x2x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<8x2x1x4xf32> to vector<1x1x1x8x2x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x8x2x1x1x4xf32>, tensor<1x1x1x8x2x1x1x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#1, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 8, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x8x2x1x1x4xf32> into tensor<1x1x8x8x2x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x8x8x2x4x16x4xf32>
}


// util.func @pingpong_dt_medium_f8_expanded(%lhs: !mexp_in_ty_f8, %rhs: !in_ty_f8, %unused_acc: tensor<1x1x4x8x2x4x16x4xf32>) -> tensor<1x1x4x8x2x4x16x4xf32> {
//   %7 = tensor.empty() : tensor<1x1x4x8x2x4x16x4xf32>
//   %cst_2 = arith.constant 1.0 : f32
//   %8 = linalg.fill ins(%cst_2 : f32) outs(%7 : tensor<1x1x4x8x2x4x16x4xf32>) -> tensor<1x1x4x8x2x4x16x4xf32>
//   util.return %8 : tensor<1x1x4x8x2x4x16x4xf32>
// }

// util.func @pingpong_dt_medium_f8_expanded(%lhs: !mexp_in_ty_f8, %rhs: !in_ty_f8, %unused_acc: tensor<1x1x4x8x2x4x16x4xf32>) -> tensor<1x1x4x8x2x4x16x4xf32> {
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c2 = arith.constant 2 : index
//   %c3 = arith.constant 3 : index
//   %c4 = arith.constant 4 : index
//   %c8 = arith.constant 8 : index
//   %c16 = arith.constant 16 : index
//   %c32 = arith.constant 32 : index
//   %c64 = arith.constant 64 : index
//   %c128 = arith.constant 128 : index
//   %c256 = arith.constant 256 : index
//   %cst = arith.constant 0.000000e+00 : f32
//   %50 = tensor.empty() : tensor<1x1x4x8x2x4x16x4xf32>
//   %51 = linalg.fill ins(%cst : f32) outs(%50 : tensor<1x1x4x8x2x4x16x4xf32>) -> tensor<1x1x4x8x2x4x16x4xf32>
//   %52 = scf.for %arg3 = %c0 to %c64 step %c1 iter_args(%arg4 = %51) -> (tensor<1x1x4x8x2x4x16x4xf32>) {
//     %56 = tensor.empty() : tensor<1x1x8x4x4x4x2x8xf8E4M3FNUZ>
//     %57 = scf.forall (%arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0, 0, 0, 0) to (1, 1, 8, 4, 4, 4, 2, 8) step (1, 1, 1, 1, 1, 1, 2, 8) shared_outs(%arg13 = %56) -> (tensor<1x1x8x4x4x4x2x8xf8E4M3FNUZ>) {
//       %extracted_slice_5 = tensor.extract_slice %lhs[0, %arg3, %arg7, %arg8, %arg9, %arg10, 0, 0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : !mexp_in_ty_f8 to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %extracted_slice_6 = tensor.extract_slice %arg13[0, 0, %arg7, %arg8, %arg9, %arg10, 0, 0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x8x4x4x4x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %61 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice_5 : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>) outs(%extracted_slice_6 : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>) -> tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
//       scf.forall.in_parallel {
//         tensor.parallel_insert_slice %61 into %arg13[0, 0, %arg7, %arg8, %arg9, %arg10, 0, 0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ> into tensor<1x1x8x4x4x4x2x8xf8E4M3FNUZ>
//       }
//     } {mapping = [#gpu.thread<linear_dim_7>, #gpu.thread<linear_dim_6>, #gpu.thread<linear_dim_5>, #gpu.thread<linear_dim_4>, #gpu.thread<linear_dim_3>, #gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
//     %58 = tensor.empty() : tensor<1x1x4x2x4x16x2x8xf8E4M3FNUZ>
//     %59 = scf.forall (%arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0, 0, 0, 0) to (1, 1, 4, 2, 4, 16, 2, 8) step (1, 1, 1, 1, 1, 1, 2, 8) shared_outs(%arg13 = %58) -> (tensor<1x1x4x2x4x16x2x8xf8E4M3FNUZ>) {
//       %extracted_slice_5 = tensor.extract_slice %rhs[0, %arg3, %arg7, %arg8, %arg9, %arg10, 0, 0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : !in_ty_f8 to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %extracted_slice_6 = tensor.extract_slice %arg13[0, 0, %arg7, %arg8, %arg9, %arg10, 0, 0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x4x2x4x16x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %61 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice_5 : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>) outs(%extracted_slice_6 : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>) -> tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
//       scf.forall.in_parallel {
//         tensor.parallel_insert_slice %61 into %arg13[0, 0, %arg7, %arg8, %arg9, %arg10, 0, 0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ> into tensor<1x1x4x2x4x16x2x8xf8E4M3FNUZ>
//       }
//     } {mapping = [#gpu.thread<linear_dim_7>, #gpu.thread<linear_dim_6>, #gpu.thread<linear_dim_5>, #gpu.thread<linear_dim_4>, #gpu.thread<linear_dim_3>, #gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
//     %60 = scf.forall (%arg5) in (256) shared_outs(%arg6 = %arg4) -> (tensor<1x1x4x8x2x4x16x4xf32>) {
//       %61:7 = affine.delinearize_index %arg5 into (1, 4, 4, 4, 1, 1) : index, index, index, index, index, index, index
//       %extracted_slice_5 = tensor.extract_slice %57[0, 0, %61#1, %61#2, %61#3, %61#4, %61#5, %61#6] [1, 1, 8, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x8x4x4x4x2x8xf8E4M3FNUZ> to tensor<1x1x8x1x1x1x2x8xf8E4M3FNUZ>
//       %62:7 = affine.delinearize_index %arg5 into (4, 1, 4, 16, 1, 1) : index, index, index, index, index, index, index
//       %extracted_slice_6 = tensor.extract_slice %59[0, 0, %62#1, %62#2, %62#3, %62#4, %62#5, %62#6] [1, 1, 1, 2, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x4x2x4x16x2x8xf8E4M3FNUZ> to tensor<1x1x1x2x1x1x2x8xf8E4M3FNUZ>
//       %63:7 = affine.delinearize_index %arg5 into (4, 1, 1, 4, 16, 1) : index, index, index, index, index, index, index
//       %extracted_slice_7 = tensor.extract_slice %arg6[0, 0, %63#1, %63#2, %63#3, %63#4, %63#5, %63#6] [1, 1, 1, 8, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x4x8x2x4x16x4xf32> to tensor<1x1x1x8x2x1x1x4xf32>
//       %64 = iree_codegen.inner_tiled ins(%extracted_slice_5, %extracted_slice_6) outs(%extracted_slice_7) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2>, lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1], reduction = [0, 0, 1], workgroup = [1, 1, 0]}>} : tensor<1x1x8x1x1x1x2x8xf8E4M3FNUZ>, tensor<1x1x1x2x1x1x2x8xf8E4M3FNUZ> into tensor<1x1x1x8x2x1x1x4xf32>
//       scf.forall.in_parallel {
//         tensor.parallel_insert_slice %64 into %arg6[0, 0, %63#1, %63#2, %63#3, %63#4, %63#5, %63#6] [1, 1, 1, 8, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x8x2x1x1x4xf32> into tensor<1x1x4x8x2x4x16x4xf32>
//       }
//     } {mapping = [#gpu.thread<linear_dim_0>]}
//     scf.yield %60 : tensor<1x1x4x8x2x4x16x4xf32>
//   }
//   util.return %52 : tensor<1x1x4x8x2x4x16x4xf32>
// }

util.func private @pingpong_dt_medium_f8_expanded_2(%lhs_base: tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ>, %rhs_base: tensor<1x?x4x2x4x16x2x8xf8E4M3FNUZ>, %unused_acc: tensor<1x1x4x8x2x4x16x4xf32>) -> tensor<1x1x4x8x2x4x16x4xf32> {
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
  %c4096 = arith.constant 4096 : index
  %cst = arith.constant 0.0 : f8E4M3FNUZ

  %dim = tensor.dim %rhs_base, %c1 : tensor<1x?x4x2x4x16x2x8xf8E4M3FNUZ>

  %lhs_shared = memref.alloc() : memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x1x4x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>

  scf.forall (%id) in (512) {
    %delin:4 = affine.delinearize_index %id into (8, 4, 4, 4) : index, index, index, index
    %lhs_thread_local = tensor.extract_slice %lhs_base [%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (512) {
    %delin:4 = affine.delinearize_index %id into (4, 2, 4, 16) : index, index, index, index
    %rhs_thread_local = tensor.extract_slice %rhs_base [%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x4x2x4x16x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %c0, %delin#0, %delin#1, %delin#2, %delin#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x4x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %0 = tensor.empty() : tensor<1x1x4x8x2x4x16x4xf32>
  %1 = scf.forall (%id) in (256) shared_outs(%out = %0) -> tensor<1x1x4x8x2x4x16x4xf32> {
    %ids:4 = affine.delinearize_index %id into (1, 4, 4, 16) : index, index, index, index
    %m_inner_ids:2 = affine.delinearize_index %ids#3 into (4, 4) : index, index

    %glb0_lhs = arith.muli %ids#1, %c2 overflow<nsw, nuw> : index
    %glb1_lhs = arith.addi %glb0_lhs, %c1 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<8x2x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c128 : index
    %cmp1 = arith.cmpi sge, %id, %c128 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }

    %3 = scf.for %i = %c1 to %dim step %c1 iter_args(%iter = %2) -> vector<8x2x1x4xf32> {
      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids#2, %m_inner_ids#0, %m_inner_ids#1, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %c0, %ids#1, %c0, %ids#2, %ids#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x4x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ>
      %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs_base [%c0, %i, %ids#1, %c0, %ids#2, %ids#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x4x2x4x16x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_thread_1 = tensor.extract_slice %rhs_base [%c0, %i, %ids#1, %c1, %ids#2, %ids#3, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x4x2x4x16x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>

      %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids#2, %m_inner_ids#0, %m_inner_ids#1, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ>
      %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %c0, %ids#1, %c0, %ids#2, %ids#3, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x4x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ>
      %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs_base [%c0, %i, %glb0_lhs, %ids#2, %m_inner_ids#0, %m_inner_ids#1, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_thread_1 = tensor.extract_slice %lhs_base [%c0, %i, %glb1_lhs, %ids#2, %m_inner_ids#0, %m_inner_ids#1, %c0, %c0] [1, 1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x8x4x4x4x2x8xf8E4M3FNUZ> to tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared[%c0, %c0, %ids#1, %c0, %ids#2, %ids#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x4x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared[%c0, %c0, %ids#1, %c1, %ids#2, %ids#3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x4x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>

      vector.transfer_write %lhs_vec_local_0, %lhs_shared[%c0, %c0, %glb0_lhs, %ids#2, %m_inner_ids#0, %m_inner_ids#1, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_1, %lhs_shared[%c0, %c0, %glb1_lhs, %ids#2, %m_inner_ids#0, %m_inner_ids#1, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>  

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot2 : vector<8x2x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids#2, %m_inner_ids#0, %m_inner_ids#1, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %c0, %ids#1, %c0, %ids#2, %ids#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x4x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ>
    %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

    %lhs_vec_2 = vector.transfer_read %lhs_shared[%c0, %c0, %c0, %ids#2, %m_inner_ids#0, %m_inner_ids#1, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x8x4x4x4x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ>
    %rhs_vec_2 = vector.transfer_read %rhs_shared[%c0, %c0, %ids#1, %c0, %ids#2, %ids#3, %c1, %c0], %cst {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x4x2x4x16x2x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ>
    %lhs_vec_2_t = vector.shape_cast %lhs_vec_2 : vector<1x1x8x1x1x1x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_2_t = vector.shape_cast %rhs_vec_2 : vector<1x1x1x2x1x1x1x8xf8E4M3FNUZ> to vector<2x1x1x8xf8E4M3FNUZ>

    %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2_t, %rhs_vec_2_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<2x1x1x8xf8E4M3FNUZ> into vector<8x2x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x8x2x1x1x4xf32>
    %cast = vector.shape_cast %dot2 : vector<8x2x1x4xf32> to vector<1x1x1x8x2x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x8x2x1x1x4xf32>, tensor<1x1x1x8x2x1x1x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#1, %c0, %c0, %ids#2, %ids#3, %c0] [1, 1, 1, 8, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x8x2x1x1x4xf32> into tensor<1x1x4x8x2x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x4x8x2x4x16x4xf32>
}


// util.func @pingpong_dt_medium_f8_expanded(%lhs: !mexp_in_ty_f8, %rhs: !in_ty_f8, %unused_acc: tensor<1x1x4x8x2x4x16x4xf32>) -> tensor<1x1x4x8x2x4x16x4xf32> {
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c2 = arith.constant 2 : index
//   %c3 = arith.constant 3 : index
//   %c4 = arith.constant 4 : index
//   %c8 = arith.constant 8 : index
//   %c16 = arith.constant 16 : index
//   %c32 = arith.constant 32 : index
//   %c64 = arith.constant 64 : index
//   %c128 = arith.constant 128 : index
//   %c256 = arith.constant 256 : index
//   %cst = arith.constant 0.0 : f8E4M3FNUZ
//   %lhs_shared_base = memref.alloc() : !mflat_shared_f8
//   %rhs_shared_base = memref.alloc() : !flat_shared_f8
// 
//   // %dim = tensor.dim %rhs_base, %c1 : !in_ty_f8
//   // %lhs = iree_gpu.buffer_resource_cast %lhs_base cacheSwizzleStride(%dim) : !mexp_in_ty_f8
//   // %rhs = iree_gpu.buffer_resource_cast %rhs_base cacheSwizzleStride(%dim) : !in_ty_f8
// 
//   // %lhs_shared_swizzle = iree_codegen.swizzle_hint %lhs_shared_base[#iree_codegen.rotate_rows<128, 8>] : !mflat_shared_f8
//   // %rhs_shared_swizzle = iree_codegen.swizzle_hint %rhs_shared_base[#iree_codegen.rotate_rows<128, 8>] : !flat_shared_f8
// 
//   %lhs_shared = memref.expand_shape %lhs_shared_base [[0, 1, 2, 3, 4, 5, 6]] output_shape [2, 8, 4, 4, 4, 2, 8] : !mflat_shared_f8 into !mshared_f8
//   %rhs_shared = memref.expand_shape %rhs_shared_base [[0, 1, 2, 3, 4, 5, 6]] output_shape [2, 4, 2, 4, 16, 2, 8] : !flat_shared_f8 into !shared_f8
// 
//   %lhs_init = tensor.extract_slice %lhs [0, 0, 0, 0, 0, 0, 0, 0] [1, 2, 8, 4, 4, 4, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : !mexp_in_ty_f8 to !mexp_block_in_f8
//   %rhs_init = tensor.extract_slice %rhs [0, 0, 0, 0, 0, 0, 0, 0] [1, 2, 4, 2, 4, 16, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : !in_ty_f8 to !block_in_f8
// 
//   scf.forall (%id) in (1024) {
//     %delin:7 = affine.delinearize_index %id into (2, 8, 4, 4, 4, 2, 8) : index, index, index, index, index, index, index
//     // %vec = arith.muli %delin#1, %c16 overflow<nsw, nuw> : index
//     %lhs_thread_local = tensor.extract_slice %lhs_init [%delin#0, %delin#1, %delin#2, %delin#3, %delin#4, 0, 0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !mexp_block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//     %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
//     vector.transfer_write %lhs_vec_local, %lhs_shared[%delin#0, %delin#1, %delin#2, %delin#3, %delin#4, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !mshared_f8
//   } {mapping = [#gpu.thread<linear_dim_0>]}
//   scf.forall (%id) in (1024) {
//     %delin:7 = affine.delinearize_index %id into (2, 4, 2, 4, 16, 2, 8) : index, index, index, index, index, index, index
//     // %vec = arith.muli %delin#1, %c16 overflow<nsw, nuw> : index
//     %rhs_thread_local = tensor.extract_slice %rhs_init [%delin#0, %delin#1, %delin#2, %delin#3, %delin#4, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//     %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
//     vector.transfer_write %rhs_vec_local, %rhs_shared[%delin#0, %delin#1, %delin#2, %delin#3, %delin#4, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !shared_f8
//   } {mapping = [#gpu.thread<linear_dim_0>]}
// 
//   // %lhs_shared_expand = memref.expand_shape %lhs_shared [[0, 1], [2, 3]] output_shape [8, 16, 4, 32] : !mshared_f8 into !mshared_exp_f8
//   // %rhs_shared_expand = memref.expand_shape %rhs_shared [[0, 1], [2, 3]] output_shape [16, 16, 4, 32] : !shared_f8 into !shared_exp_f8
// 
//   %0 = tensor.empty() : tensor<1x1x4x8x2x4x16x4xf32>
//   %1 = scf.forall (%id) in (256) shared_outs(%out = %0) -> tensor<1x1x4x8x2x4x16x4xf32> {
//     %ids_lhs:4 = affine.delinearize_index %id into (4, 4, 4) : index, index, index, index
//     %ids_rhs:4 = affine.delinearize_index %id into (4, 4, 16) : index, index, index, index
// 
//     // %ids:4 = affine.delinearize_index %id into (2, 4, 4, 16) : index, index, index, index
//     // %inner_id = arith.muli %ids#2, %c8 overflow<nsw, nuw> : index
//     // %inner_id_acc = arith.muli %ids#2, %c4 overflow<nsw, nuw> : index
//     // %m_outer_id = arith.muli %ids#0, %c4 overflow<nsw, nuw> : index
//     // %n_outer_id = arith.muli %ids#1, %c4 overflow<nsw, nuw> : index
//     // %delin:2 = affine.delinearize_index %id into (64, 8) : index, index
//     // %wt:3 = affine.delinearize_index %id into (8, 8, 8) : index, index, index
// 
//     // // Inner 64 loads 8 threads x 16 elements.
//     // %gko = arith.muli %wt#2, %c16 overflow<nsw, nuw> : index
//     // // RHS indexing. Each subgroup loads 32 contiguous rows out of 256.
//     // %bpo = arith.muli %wt#0, %c32 overflow<nsw, nuw> : index
//     // // Base index is remaining outer 8 lanes + subgroup base.
//     // %glb0 = arith.addi %wt#1, %bpo overflow<nsw, nuw> : index
//     // %glb1 = arith.addi %glb0, %c8 overflow<nsw, nuw> : index
//     // %glb2 = arith.addi %glb1, %c8 overflow<nsw, nuw> : index
//     // %glb3 = arith.addi %glb2, %c8 overflow<nsw, nuw> : index
//     // // LHS indexing.
//     // %bpo_lhs = arith.muli %wt#0, %c16 overflow<nsw, nuw> : index
//     // %glb0_lhs = arith.addi %wt#1, %bpo_lhs overflow<nsw, nuw> : index
//     // %glb1_lhs = arith.addi %glb0_lhs, %c8 overflow<nsw, nuw> : index
// 
//     %glb_rhs:5 = affine.delinearize_index %id into (2, 4, 2, 4, 4) : index, index, index, index, index
//     %glb_rhs1 = arith.addi %glb_rhs#4, %c1 overflow<nsw, nuw> : index
//     %glb_rhs2 = arith.addi %glb_rhs#4, %c2 overflow<nsw, nuw> : index
//     %glb_rhs3 = arith.addi %glb_rhs#4, %c3 overflow<nsw, nuw> : index
// 
//     %glb_lhs:4 = affine.delinearize_index %id into (2, 8, 4, 4) : index, index, index, index
// 
//     %2 = arith.constant dense<0.0> : vector<1x1x1x8x2x1x1x4xf32>
// 
//     %cmp0 = arith.cmpi slt, %id, %c128 : index
//     %cmp1 = arith.cmpi sge, %id, %c128 : index
//     scf.if %cmp0 {
//       rocdl.s.barrier
//     }
//     %3 = scf.for %i = %c2 to %c32 step %c2 iter_args(%iter = %2) -> vector<1x1x1x8x2x1x1x4xf32> {
// 
//       %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !mshared_f8, vector<1x8x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %ids_rhs#1, %c0, %ids_rhs#2, %ids_rhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !shared_f8, vector<1x1x2x1x1x2x8xf8E4M3FNUZ>
// 
//       rocdl.sched.barrier 0
// 
//       // Global loads of rhs.
//       %rhs_block = tensor.extract_slice %rhs [0, %i, 0, 0, 0, 0, 0, 0] [1, 2, 4, 2, 4, 16, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : !in_ty_f8 to !block_in_f8
//       %rhs_thread_0 = tensor.extract_slice %rhs_block [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs#4, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_thread_1 = tensor.extract_slice %rhs_block [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs1, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_thread_2 = tensor.extract_slice %rhs_block [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs2, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_vec_local_2 = vector.transfer_read %rhs_thread_2 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_thread_3 = tensor.extract_slice %rhs_block [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs3, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_vec_local_3 = vector.transfer_read %rhs_thread_3 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
// 
//       rocdl.sched.barrier 0
// 
//       %lhs_vec_2 = vector.transfer_read %lhs_shared[%c1, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !mshared_f8, vector<1x8x1x1x1x2x8xf8E4M3FNUZ>
//       %rhs_vec_2 = vector.transfer_read %rhs_shared[%c1, %ids_rhs#1, %c0, %ids_rhs#2, %ids_rhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !shared_f8, vector<1x1x2x1x1x2x8xf8E4M3FNUZ>
// 
//       rocdl.sched.barrier 0
// 
//       // Global loads of lhs.
//       %lhs_block = tensor.extract_slice %lhs [0, %i, 0, 0, 0, 0, 0, 0] [1, 2, 8, 4, 4, 4, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1]  : !mexp_in_ty_f8 to !mexp_block_in_f8
//       %lhs_thread_0 = tensor.extract_slice %lhs_block [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c0, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !mexp_block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_thread_1 = tensor.extract_slice %lhs_block [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c1, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !mexp_block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_thread_2 = tensor.extract_slice %lhs_block [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c2, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !mexp_block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_vec_local_2 = vector.transfer_read %lhs_thread_2 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_thread_3 = tensor.extract_slice %lhs_block [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c3, %c0, %c0] [1, 1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : !mexp_block_in_f8 to tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>
//       %lhs_vec_local_3 = vector.transfer_read %lhs_thread_3 [%c0, %c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : tensor<1x1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x1x2x8xf8E4M3FNUZ>
// 
//       gpu.barrier
//       rocdl.sched.barrier 0
//       rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }
// 
//       %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0, %rhs_vec_0) outs(%iter) {
//         indexing_maps = #contraction_accesses,
//         iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
//         kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2>
//       } : vector<1x8x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x2x1x1x2x8xf8E4M3FNUZ> into vector<1x1x1x8x2x1x1x4xf32>
// 
//       rocdl.s.setprio 0
//       gpu.barrier
//       rocdl.sched.barrier 0
// 
//       vector.transfer_write %rhs_vec_local_0, %rhs_shared [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs#4, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !shared_f8
//       vector.transfer_write %rhs_vec_local_1, %rhs_shared [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs1, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !shared_f8
//       vector.transfer_write %rhs_vec_local_2, %rhs_shared [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs2, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !shared_f8
//       vector.transfer_write %rhs_vec_local_3, %rhs_shared [%glb_rhs#0, %glb_rhs#1, %glb_rhs#2, %glb_rhs#3, %glb_rhs3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !shared_f8
// 
//       vector.transfer_write %lhs_vec_local_0, %lhs_shared [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !mshared_f8
//       vector.transfer_write %lhs_vec_local_1, %lhs_shared [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c1, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !mshared_f8
//       vector.transfer_write %lhs_vec_local_2, %lhs_shared [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c2, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !mshared_f8
//       vector.transfer_write %lhs_vec_local_3, %lhs_shared [%glb_lhs#0, %glb_lhs#1, %glb_lhs#2, %glb_lhs#3, %c3, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x2x8xf8E4M3FNUZ>, !mshared_f8
// 
//       gpu.barrier
//       rocdl.sched.barrier 0
//       rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }
// 
//       %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2, %rhs_vec_2) outs(%dot0) {
//         indexing_maps = #contraction_accesses,
//         iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
//         kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2>
//       } : vector<1x8x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x2x1x1x2x8xf8E4M3FNUZ> into vector<1x1x1x8x2x1x1x4xf32>
// 
//       rocdl.s.setprio 0
//       gpu.barrier
//       rocdl.sched.barrier 0
// 
//       scf.yield %dot2 : vector<1x1x1x8x2x1x1x4xf32>
//     }
//     scf.if %cmp1 {
//       rocdl.s.barrier
//     }
// 
//     // Epilogue
//     %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !mshared_f8, vector<1x8x1x1x1x2x8xf8E4M3FNUZ>
//     %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %ids_rhs#1, %c0, %ids_rhs#2, %ids_rhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !shared_f8, vector<1x1x2x1x1x2x8xf8E4M3FNUZ>
// 
//     %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0, %rhs_vec_0) outs(%3) {
//       indexing_maps = #contraction_accesses,
//       iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
//       kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2>
//     } : vector<1x8x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x2x1x1x2x8xf8E4M3FNUZ> into vector<1x1x1x8x2x1x1x4xf32>
// 
//     %lhs_vec_2 = vector.transfer_read %lhs_shared[%c1, %c0, %ids_lhs#1, %ids_lhs#2, %ids_lhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !mshared_f8, vector<1x8x1x1x1x2x8xf8E4M3FNUZ>
//     %rhs_vec_2 = vector.transfer_read %rhs_shared[%c1, %ids_rhs#1, %c0, %ids_rhs#2, %ids_rhs#3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true, true]} : !shared_f8, vector<1x1x2x1x1x2x8xf8E4M3FNUZ>
// 
//     %dot2 = iree_codegen.inner_tiled ins(%lhs_vec_2, %rhs_vec_2) outs(%dot0) {
//       indexing_maps = #contraction_accesses,
//       iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
//       kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2>
//     } : vector<1x8x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x2x1x1x2x8xf8E4M3FNUZ> into vector<1x1x1x8x2x1x1x4xf32>
//     
//     %empty = tensor.empty() : tensor<1x1x1x8x2x1x1x4xf32>
//     %4 = vector.transfer_write %dot2, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x8x2x1x1x4xf32>, tensor<1x1x1x8x2x1x1x4xf32>
//     scf.forall.in_parallel {
//       tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids_rhs#0, %c0, %c0, %ids_rhs#1, %ids_rhs#2, %c0] [1, 1, 1, 8, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x8x2x1x1x4xf32> into tensor<1x1x4x8x2x4x16x4xf32>
//     }
//   } {mapping = [#gpu.thread<linear_dim_0>]}
//   util.return %1 : tensor<1x1x4x8x2x4x16x4xf32>
// }



util.func private @pingpong_medium_f8_expanded_data_tiling(%lhs_base: tensor<1x?x2x8x4x4x4x8xf8E4M3FNUZ>, %rhs_base: tensor<1x?x4x4x4x16x8xf8E4M3FNUZ>, %unused_acc: tensor<1x1x2x4x8x4x4x16x4xf32>) -> tensor<1x1x2x4x8x4x4x16x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f8E4M3FNUZ

  %dim = tensor.dim %rhs_base, %c1 : tensor<1x?x4x4x4x16x8xf8E4M3FNUZ>
  %nDim = arith.divui %dim, %c2 : index

  %lhs_expand = tensor.expand_shape %lhs_base [[0], [1, 2], [3], [4], [5], [6], [7], [8]] output_shape [1, %nDim, 2, 2, 8, 4, 4, 4, 8] : tensor<1x?x2x8x4x4x4x8xf8E4M3FNUZ> into tensor<1x?x2x2x8x4x4x4x8xf8E4M3FNUZ>
  %rhs_expand = tensor.expand_shape %rhs_base [[0], [1, 2], [3], [4], [5], [6], [7]] output_shape [1, %nDim, 2, 4, 4, 4, 16, 8] : tensor<1x?x4x4x4x16x8xf8E4M3FNUZ> into tensor<1x?x2x4x4x4x16x8xf8E4M3FNUZ>

  %lhs = tensor.collapse_shape %lhs_expand [[0, 1], [2], [3], [4], [5, 6, 7], [8]] : tensor<1x?x2x2x8x4x4x4x8xf8E4M3FNUZ> into tensor<?x2x2x8x64x8xf8E4M3FNUZ>
  %rhs = tensor.collapse_shape %rhs_expand [[0, 1], [2], [3], [4], [5, 6], [7]] : tensor<1x?x2x4x4x4x16x8xf8E4M3FNUZ> into tensor<?x2x4x4x64x8xf8E4M3FNUZ>

  %lhs_shared = memref.alloc() : memref<1x2x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  %rhs_shared = memref.alloc() : memref<1x2x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>

  scf.forall (%id) in (1024) {
    %delin:4 = affine.delinearize_index %id into (2, 2, 8, 32) : index, index, index, index
    %inner = arith.muli %delin#3, %c2 overflow<nsw, nuw> : index
    %lhs_thread_local = tensor.extract_slice %lhs [%c0, %delin#0, %delin#1, %delin#2, %inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1]  : tensor<?x2x2x8x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
    %lhs_vec_local = vector.transfer_read %lhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>
    vector.transfer_write %lhs_vec_local, %lhs_shared[%c0, %delin#0, %delin#1, %delin#2, %inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x2x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  scf.forall (%id) in (1024) {
    %delin:4 = affine.delinearize_index %id into (2, 4, 4, 32) : index, index, index, index
    %inner = arith.muli %delin#3, %c2 overflow<nsw, nuw> : index
    %rhs_thread_local = tensor.extract_slice %rhs [%c0, %delin#0, %delin#1, %delin#2, %inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1] : tensor<?x2x4x4x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
    %rhs_vec_local = vector.transfer_read %rhs_thread_local [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>
    vector.transfer_write %rhs_vec_local, %rhs_shared[%c0, %delin#0, %delin#1, %delin#2, %inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x2x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %0 = tensor.empty() : tensor<1x1x2x4x8x4x4x16x4xf32>
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> tensor<1x1x2x4x8x4x4x16x4xf32> {
    %ids:3 = affine.delinearize_index %id into (2, 4, 64) : index, index, index
    %threads:2 = affine.delinearize_index %ids#2 into (4, 16) : index, index

    %glb_rhs:3 = affine.delinearize_index %id into (4, 4, 32) : index, index, index
    %glb_rhs_inner = arith.muli %glb_rhs#2, %c2 overflow<nsw, nuw> : index

    %glb_lhs:3 = affine.delinearize_index %id into (2, 8, 32) : index, index, index
    %glb_lhs_inner = arith.muli %glb_lhs#2, %c2 overflow<nsw, nuw> : index

    %2 = arith.constant dense<0.0> : vector<8x4x1x4xf32>

    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }

    %3 = scf.for %i = %c1 to %nDim step %c1 iter_args(%iter = %2) -> vector<8x4x1x4xf32> {
      // Local loads.
      %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x2x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x8x1x8xf8E4M3FNUZ>
      %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %c0, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x2x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x4x1x8xf8E4M3FNUZ>
      %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x1x1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x1x1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

      rocdl.sched.barrier 0

      // Global loads of rhs.
      %rhs_thread_0 = tensor.extract_slice %rhs [%i, %c0, %glb_rhs#0, %glb_rhs#1, %glb_rhs_inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1] :  tensor<?x2x4x4x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_vec_local_0 = vector.transfer_read %rhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_thread_1 = tensor.extract_slice %rhs [%i, %c1, %glb_rhs#0, %glb_rhs#1, %glb_rhs_inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1] : tensor<?x2x4x4x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
      %rhs_vec_local_1 = vector.transfer_read %rhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>

      rocdl.sched.barrier 0

      // Local loads.
      %lhs_vec_1 = vector.transfer_read %lhs_shared[%c0, %c1, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x2x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x8x1x8xf8E4M3FNUZ>
      %rhs_vec_1 = vector.transfer_read %rhs_shared[%c0, %c1, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x2x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x4x1x8xf8E4M3FNUZ>
      %lhs_vec_1_t = vector.shape_cast %lhs_vec_1 : vector<1x1x1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
      %rhs_vec_1_t = vector.shape_cast %rhs_vec_1 : vector<1x1x1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

      rocdl.sched.barrier 0

      // Global loads of lhs.
      %lhs_thread_0 = tensor.extract_slice %lhs [%i, %c0, %glb_lhs#0, %glb_lhs#1, %glb_lhs_inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1] :  tensor<?x2x2x8x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_vec_local_0 = vector.transfer_read %lhs_thread_0 [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_thread_1 = tensor.extract_slice %lhs [%i, %c1, %glb_lhs#0, %glb_lhs#1, %glb_lhs_inner, %c0] [1, 1, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1] : tensor<?x2x2x8x64x8xf8E4M3FNUZ> to tensor<1x1x1x1x2x8xf8E4M3FNUZ>
      %lhs_vec_local_1 = vector.transfer_read %lhs_thread_1 [%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : tensor<1x1x1x1x2x8xf8E4M3FNUZ>, vector<1x1x1x1x2x8xf8E4M3FNUZ>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      vector.transfer_write %rhs_vec_local_0, %rhs_shared [%c0, %c0, %glb_rhs#0, %glb_rhs#1, %glb_rhs_inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x2x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %rhs_vec_local_1, %rhs_shared [%c0, %c1, %glb_rhs#0, %glb_rhs#1, %glb_rhs_inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x2x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      
      vector.transfer_write %lhs_vec_local_0, %lhs_shared [%c0, %c0, %glb_lhs#0, %glb_lhs#1, %glb_lhs_inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x2x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>
      vector.transfer_write %lhs_vec_local_1, %lhs_shared [%c0, %c1, %glb_lhs#0, %glb_lhs#1, %glb_lhs_inner, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x2x8xf8E4M3FNUZ>, memref<1x2x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>

      gpu.barrier
      rocdl.sched.barrier 0
      rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

      %dot1 = iree_codegen.inner_tiled ins(%lhs_vec_1_t, %rhs_vec_1_t) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
      } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

      rocdl.s.setprio 0
      gpu.barrier
      rocdl.sched.barrier 0

      scf.yield %dot1 : vector<8x4x1x4xf32>
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    // Epilogue
    %lhs_vec_0 = vector.transfer_read %lhs_shared[%c0, %c0, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x2x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x8x1x8xf8E4M3FNUZ>
    %rhs_vec_0 = vector.transfer_read %rhs_shared[%c0, %c0, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x2x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x4x1x8xf8E4M3FNUZ>
    %lhs_vec_0_t = vector.shape_cast %lhs_vec_0 : vector<1x1x1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_0_t = vector.shape_cast %rhs_vec_0 : vector<1x1x1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

    %dot0 = iree_codegen.inner_tiled ins(%lhs_vec_0_t, %rhs_vec_0_t) outs(%3) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

    %lhs_vec_1 = vector.transfer_read %lhs_shared[%c0, %c1, %ids#0, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x2x2x8x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x8x1x8xf8E4M3FNUZ>
    %rhs_vec_1 = vector.transfer_read %rhs_shared[%c0, %c1, %ids#1, %c0, %ids#2, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x2x4x4x64x8xf8E4M3FNUZ, #gpu.address_space<workgroup>>, vector<1x1x1x4x1x8xf8E4M3FNUZ>
    %lhs_vec_1_t = vector.shape_cast %lhs_vec_1 : vector<1x1x1x8x1x8xf8E4M3FNUZ> to vector<8x1x1x8xf8E4M3FNUZ>
    %rhs_vec_1_t = vector.shape_cast %rhs_vec_1 : vector<1x1x1x4x1x8xf8E4M3FNUZ> to vector<4x1x1x8xf8E4M3FNUZ>

    %dot1 = iree_codegen.inner_tiled ins(%lhs_vec_1_t, %rhs_vec_1_t) outs(%dot0) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
    } : vector<8x1x1x8xf8E4M3FNUZ>, vector<4x1x1x8xf8E4M3FNUZ> into vector<8x4x1x4xf32>

    %empty = tensor.empty() : tensor<1x1x1x1x8x4x1x1x4xf32>
    %cast = vector.shape_cast %dot1 : vector<8x4x1x4xf32> to vector<1x1x1x1x8x4x1x1x4xf32>
    %4 = vector.transfer_write %cast, %empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true, true, true, true]} : vector<1x1x1x1x8x4x1x1x4xf32>, tensor<1x1x1x1x8x4x1x1x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %out[%c0, %c0, %ids#0, %ids#1, %c0, %c0, %threads#0, %threads#1, %c0] [1, 1, 1, 1, 8, 4, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x8x4x1x1x4xf32> into tensor<1x1x2x4x8x4x4x16x4xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %1 : tensor<1x1x2x4x8x4x4x16x4xf32>
}
