// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- SinkBitCast.cpp - Sink bitcast through reshapes ------------------===//
//
// This pass sinks iree_tensor_ext.bitcast operations through
// tensor.expand_shape and tensor.collapse_shape operations.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/DispatchCreation/Passes.h"

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_SINKBITCASTPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc" // IWYU pragma: keep

using IREE::TensorExt::BitCastOp;

namespace {

/// Propagate bitcast through expand_shape: expand_shape(bitcast(x)) ->
/// bitcast(expand_shape(x)). Supports two cases:
/// 1. 1D bitcast (e.g., tensor<?xf8> -> tensor<?xi8>)
/// 2. Bitcast where only the last dim differs (last dim must be static)
/// (e.g., tensor<4x?x512x32xf4> -> tensor<4x?x512x16xi8>)
struct PropagateBitCastThroughExpandShape final
    : public OpRewritePattern<BitCastOp> {
  using OpRewritePattern<BitCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const override {
    if (!bitcastOp->hasOneUse()) {
      return failure();
    }

    auto expandOp = dyn_cast<tensor::ExpandShapeOp>(*bitcastOp->user_begin());
    if (!expandOp) {
      return failure();
    }

    RankedTensorType bitcastSrcType = bitcastOp.getSource().getType();
    RankedTensorType bitcastDstType = bitcastOp.getResult().getType();
    RankedTensorType expandResultType = expandOp.getResultType();

    int64_t expandResultRank = expandResultType.getRank();
    bool isBitcastRank1 =
        bitcastSrcType.getRank() == 1 && bitcastDstType.getRank() == 1;

    if (!isBitcastRank1) {
      // Only handle bitcast where source and dest differ only in the last dim.
      // Compare shapes directly for all dims except the last.
      if (bitcastSrcType.getShape().drop_back() !=
              bitcastDstType.getShape().drop_back() ||
          ShapedType::isDynamic(bitcastSrcType.getShape().back())) {
        return failure();
      }

      // Check that the last dimension of the bitcast result is in its own
      // reassociation group, so we don't need to deal with the case where it is
      // split into multiple dimensions.
      SmallVector<ReassociationIndices> reassoc =
          expandOp.getReassociationIndices();
      if (reassoc.empty()) {
        return failure();
      }

      const ReassociationIndices &lastGroup = reassoc.back();
      if (lastGroup.size() != 1 ||
          lastGroup[0] != static_cast<int64_t>(expandResultRank - 1)) {
        return failure();
      }
    }

    // Compute the new expand output shape: same as original but with the last
    // dim scaled according to the bitcast source and destination element type
    // bit widths.
    SmallVector<int64_t> newOutputShape(expandResultType.getShape());
    newOutputShape.back() =
        (newOutputShape.back() * bitcastDstType.getElementTypeBitWidth()) /
        bitcastSrcType.getElementTypeBitWidth();

    SmallVector<OpFoldResult> oldMixedOutputShape =
        expandOp.getMixedOutputShape();
    SmallVector<OpFoldResult> newMixedOutputShape(oldMixedOutputShape);
    newMixedOutputShape.back() = rewriter.getIndexAttr(newOutputShape.back());

    // Create the new expand_shape with the original bitcast source, but with
    // the new output shape.
    auto newExpandResultType =
        RankedTensorType::get(newOutputShape, bitcastSrcType.getElementType());
    rewriter.setInsertionPoint(expandOp);
    auto newExpandOp = tensor::ExpandShapeOp::create(
        rewriter, expandOp.getLoc(), newExpandResultType, bitcastOp.getSource(),
        expandOp.getReassociationIndices(), newMixedOutputShape);

    // Build the dynamic dims for the new bitcast.
    SmallVector<Value> newDynSrcDims;
    SmallVector<Value> newDynDstDims;

    for (int64_t i = 0; i < expandResultRank; ++i) {
      if (ShapedType::isDynamic(newOutputShape[i])) {
        Value dimVal = getValueOrCreateConstantIndexOp(
            rewriter, bitcastOp.getLoc(), newMixedOutputShape[i]);
        newDynSrcDims.push_back(dimVal);
      }
    }

    for (int64_t i = 0; i < expandResultRank; ++i) {
      if (expandResultType.isDynamicDim(i)) {
        Value dimVal = getValueOrCreateConstantIndexOp(
            rewriter, bitcastOp.getLoc(), oldMixedOutputShape[i]);
        newDynDstDims.push_back(dimVal);
      }
    }

    rewriter.replaceOpWithNewOp<BitCastOp>(
        expandOp, expandResultType, newExpandOp, newDynSrcDims, newDynDstDims);
    rewriter.eraseOp(bitcastOp);
    return success();
  }
};

/// Propagate bitcast through collapse_shape: collapse_shape(bitcast(x)) ->
/// bitcast(collapse_shape(x)). Only supports bitcast where only the last dim
/// differs (last dim must also be static).
/// (e.g., tensor<4x?x512x32xf4> -> tensor<4x?x512x16xi8>)
struct PropagateBitCastThroughCollapseShape final
    : public OpRewritePattern<BitCastOp> {
  using OpRewritePattern<BitCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const override {
    if (!bitcastOp->hasOneUse()) {
      return failure();
    }

    auto collapseOp =
        dyn_cast<tensor::CollapseShapeOp>(*bitcastOp->user_begin());
    if (!collapseOp) {
      return failure();
    }

    RankedTensorType bitcastSrcType = bitcastOp.getSource().getType();
    RankedTensorType bitcastDstType = bitcastOp.getResult().getType();
    RankedTensorType collapseResultType = collapseOp.getResultType();

    // Only handle bitcast where source and dest differ only in the last dim.
    // Compare shapes directly for all dims except the last.
    if (bitcastSrcType.getShape().drop_back() !=
            bitcastDstType.getShape().drop_back() ||
        ShapedType::isDynamic(bitcastSrcType.getShape().back())) {
      return failure();
    }

    // Check that all dynamic dimensions are in their own reassociation groups,
    // so we don't need to deal with the case where it is merged with other
    // dimensions.
    SmallVector<ReassociationIndices> reassoc =
        collapseOp.getReassociationIndices();
    for (const auto &group : reassoc) {
      for (int64_t dim : group) {
        if (bitcastDstType.isDynamicDim(dim) && group.size() != 1) {
          return failure();
        }
      }
    }

    // Compute the new collapsed shape. The last dim of the new collapse result
    // should be scaled by the bitcast source and destination element type bit
    // widths. Multiply first to avoid integer division truncation.
    SmallVector<int64_t> newCollapseResultShape(collapseResultType.getShape());
    newCollapseResultShape.back() = (newCollapseResultShape.back() *
                                     bitcastDstType.getElementTypeBitWidth()) /
                                    bitcastSrcType.getElementTypeBitWidth();

    // Create the new collapse_shape operating on the original bitcast source,
    // but with the new result shape.
    auto newCollapseResultType = RankedTensorType::get(
        newCollapseResultShape, bitcastSrcType.getElementType());
    rewriter.setInsertionPoint(collapseOp);
    auto newCollapseOp = tensor::CollapseShapeOp::create(
        rewriter, collapseOp.getLoc(), newCollapseResultType,
        bitcastOp.getSource(), collapseOp.getReassociationIndices());

    // Use the original bitcast's source and result dynamic dims, since we
    // already check they are in their own reassociation groups (no merging with
    // other dims).
    rewriter.replaceOpWithNewOp<BitCastOp>(
        collapseOp, collapseResultType, newCollapseOp,
        bitcastOp.getSourceDims(), bitcastOp.getResultDims());
    rewriter.eraseOp(bitcastOp);
    return success();
  }
};

struct SinkBitCastPass final
    : public impl::SinkBitCastPassBase<SinkBitCastPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<PropagateBitCastThroughExpandShape,
                 PropagateBitCastThroughCollapseShape>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
