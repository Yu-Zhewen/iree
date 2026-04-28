// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EmulateNarrowType.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_AMDGPUEMULATENARROWTYPEPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct ConvertRawBufferCast final
    : OpConversionPattern<amdgpu::FatRawBufferCastOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(amdgpu::FatRawBufferCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to convert memref type: {0}",
                                      op.getResult().getType()));
    }
    if (newTy == op.getResult().getType()) {
      // Nothing to do.
      return failure();
    }

    // |validBytes| and |cacheSwizzleStride| are independent of element type
    // and don't need to be updated.
    rewriter.replaceOpWithNewOp<amdgpu::FatRawBufferCastOp>(
        op, newTy, adaptor.getSource(), adaptor.getValidBytes(),
        adaptor.getCacheSwizzleStride(), adaptor.getBoundsCheck(),
        adaptor.getResetOffset());
    return success();
  }
};

/// Converts GatherToLDSOp when its memrefs change from sub-byte types
/// (e.g. f4E2M1FN) to byte-sized types (i8) during narrow type emulation.
/// The pattern linearizes multi-dimensional indices into the converted 1D
/// memref space and adjusts the transfer type accordingly.
struct ConvertGatherToLDS final : OpConversionPattern<amdgpu::GatherToLDSOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(amdgpu::GatherToLDSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType origSrcType = op.getSrc().getType();
    MemRefType origDstType = op.getDst().getType();
    auto newSrcType = cast<MemRefType>(adaptor.getSrc().getType());
    auto newDstType = cast<MemRefType>(adaptor.getDst().getType());

    // Only convert sub-byte element types.
    if (origSrcType.getElementTypeBitWidth() >= 8 &&
        origDstType.getElementTypeBitWidth() >= 8) {
      return failure();
    }

    // If types didn't change, nothing to do.
    if (newSrcType == origSrcType && newDstType == origDstType) {
      return failure();
    }

    Location loc = op.getLoc();
    int64_t origSrcBits = origSrcType.getElementTypeBitWidth();
    int64_t newSrcBits = newSrcType.getElementTypeBitWidth();
    int64_t origDstBits = origDstType.getElementTypeBitWidth();
    int64_t newDstBits = newDstType.getElementTypeBitWidth();

    // Only convert when the transfer vector's total bits are a multiple of
    // the new element bit width. E.g. vector<3xf4E2M1FN> (12 bits) cannot
    // be cleanly packed into i8 elements.
    if (auto vecType = dyn_cast<VectorType>(op.getTransferType())) {
      int64_t totalBits =
          vecType.getNumElements() * vecType.getElementTypeBitWidth();
      if (totalBits % newSrcBits != 0) {
        return rewriter.notifyMatchFailure(
            op,
            "transfer vector bit-width is not a multiple of the new element "
            "bit width");
      }
    }

    // Check both source and destination convertibility before modifying IR.
    if (!canLinearizeAndPack(op.getSrcIndices(), origSrcType, origSrcBits,
                             newSrcBits)) {
      return rewriter.notifyMatchFailure(
          op, "failed to linearize source indices (dynamic or mismatched "
              "strides/offset, or invalid bit-width ratio)");
    }
    if (!canLinearizeAndPack(op.getDstIndices(), origDstType, origDstBits,
                             newDstBits)) {
      return rewriter.notifyMatchFailure(
          op, "failed to linearize destination indices (dynamic or mismatched "
              "strides/offset, or invalid bit-width ratio)");
    }

    // Linearize source indices into a 1D byte-offset index.
    Value srcIdx = linearizeAndPack(rewriter, loc, op.getSrcIndices(),
                                    origSrcType, origSrcBits, newSrcBits);

    // Linearize destination indices.
    Value dstIdx = linearizeAndPack(rewriter, loc, op.getDstIndices(),
                                    origDstType, origDstBits, newDstBits);

    // Adjust transfer type to use the new element type.
    Type newTransferType = convertTransferType(
        rewriter.getContext(), op.getTransferType(), origSrcBits, newSrcBits);

    amdgpu::GatherToLDSOp::create(
        rewriter, loc, adaptor.getSrc(), ValueRange{srcIdx}, adaptor.getDst(),
        ValueRange{dstIdx}, TypeAttr::get(newTransferType), op.getAsyncAttr());

    rewriter.eraseOp(op);
    return success();
  }

private:
  // Checks whether linearizeAndPack can succeed without modifying IR.
  // Dynamic offsets are permitted because the adapted (type-converted) memref
  // already carries the packed offset; we only linearize the indices here.
  static bool canLinearizeAndPack(ValueRange indices, MemRefType origType,
                                  int64_t origBits, int64_t newBits) {
    auto [strides, offset] = origType.getStridesAndOffset();
    for (int64_t stride : strides) {
      if (ShapedType::isDynamic(stride)) {
        return false;
      }
    }
    if (indices.size() != strides.size()) {
      return false;
    }
    if (origBits != newBits &&
        (newBits <= origBits || newBits % origBits != 0)) {
      return false;
    }
    return true;
  }

  // Linearizes multi-dimensional indices into a 1D index for the packed
  // byte-addressable memref. The offset is NOT included because the adapted
  // (type-converted) memref already carries the packed offset.
  //   linearIdx = sum(idx[i] * stride[i])
  //   packedIdx = linearIdx / (newBits / origBits)
  static Value linearizeAndPack(ConversionPatternRewriter &rewriter,
                                Location loc, ValueRange indices,
                                MemRefType origType, int64_t origBits,
                                int64_t newBits) {
    auto [strides, offset] = origType.getStridesAndOffset();

    auto overflowFlags =
        arith::IntegerOverflowFlags::nsw | arith::IntegerOverflowFlags::nuw;
    Value linearIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (auto [idx, stride] : llvm::zip(indices, strides)) {
      Value strideVal = arith::ConstantIndexOp::create(rewriter, loc, stride);
      Value product =
          arith::MulIOp::create(rewriter, loc, idx, strideVal, overflowFlags);
      linearIdx = arith::AddIOp::create(rewriter, loc, linearIdx, product,
                                        overflowFlags);
    }

    // Pack: convert from origBits-element units to newBits-element units.
    if (origBits != newBits) {
      int64_t packRatio = newBits / origBits;
      Value ratioVal = arith::ConstantIndexOp::create(rewriter, loc, packRatio);
      linearIdx = arith::DivUIOp::create(rewriter, loc, linearIdx, ratioVal);
    }

    return linearIdx;
  }

  // Converts the transfer type from sub-byte elements to byte-sized elements,
  // preserving the total transfer size in bits. The caller must ensure
  // totalBits is a multiple of newBits (the op verifier enforces that
  // transfer sizes are 8, 16, 32, 96, or 128 bits, all multiples of 8).
  static Type convertTransferType(MLIRContext *context, Type origType,
                                  int64_t origBits, int64_t newBits) {
    if (auto vecType = dyn_cast<VectorType>(origType)) {
      int64_t totalBits =
          vecType.getNumElements() * vecType.getElementTypeBitWidth();
      assert(totalBits % newBits == 0 &&
             "transfer size must be a multiple of the new element bit width");
      int64_t newElems = totalBits / newBits;
      return VectorType::get({newElems}, IntegerType::get(context, newBits));
    }
    return IntegerType::get(context, newBits);
  }
};

/// Converts multi-dimensional memref.reinterpret_cast on sub-byte types to a
/// linearized 1D reinterpret_cast on the converted byte-sized type. This
/// handles the cases that upstream ConvertMemRefReinterpretCast rejects
/// (rank > 1, dynamic offsets) by reading sizes and strides directly from the
/// op rather than relying on extract_strided_metadata.
struct ConvertReinterpretCastNarrowType final
    : OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType newTy =
        getTypeConverter()->convertType<MemRefType>(op.getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to convert memref type: {0}", op.getType()));
    }
    if (newTy == op.getType())
      return failure();

    MemRefType origTy = op.getType();
    int srcBits = origTy.getElementTypeBitWidth();
    int dstBits = newTy.getElementTypeBitWidth();
    if (srcBits >= dstBits || dstBits % srcBits != 0)
      return failure();

    // Require all strides to be static with innermost stride == 1.
    auto staticStrides = op.getStaticStrides();
    if (llvm::any_of(staticStrides, ShapedType::isDynamic))
      return rewriter.notifyMatchFailure(op, "dynamic strides not supported");
    if (!staticStrides.empty() && staticStrides.back() != 1)
      return rewriter.notifyMatchFailure(op, "innermost stride must be 1");

    // Require all sizes to be static.
    auto staticSizes = op.getStaticSizes();
    if (llvm::any_of(staticSizes, ShapedType::isDynamic))
      return rewriter.notifyMatchFailure(op, "dynamic sizes not supported");

    int64_t packRatio = dstBits / srcBits;
    Location loc = op.getLoc();

    int64_t totalElements = 1;
    for (int64_t s : staticSizes)
      totalElements *= s;
    int64_t packedSize = llvm::divideCeil(totalElements, packRatio);

    // Handle offset: may be static or dynamic.
    OpFoldResult newOffset;
    auto mixedOffsets = op.getMixedOffsets();
    assert(mixedOffsets.size() == 1 && "reinterpret_cast has exactly 1 offset");
    auto offsetOFR = mixedOffsets[0];

    if (auto staticAttr = dyn_cast<Attribute>(offsetOFR)) {
      int64_t staticOff = cast<IntegerAttr>(staticAttr).getInt();
      if (staticOff == ShapedType::kDynamic) {
        return rewriter.notifyMatchFailure(op, "unexpected dynamic sentinel");
      }
      newOffset = rewriter.getIndexAttr(staticOff / packRatio);
    } else {
      Value dynOff = cast<Value>(offsetOFR);
      Value ratioVal =
          arith::ConstantIndexOp::create(rewriter, loc, packRatio);
      Value divided = arith::DivUIOp::create(rewriter, loc, dynOff, ratioVal);
      newOffset = divided;
    }

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, newTy, adaptor.getSource(), newOffset,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(packedSize)},
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
    return success();
  }
};

/// Converts memref.cast on sub-byte types by forwarding to the adapted source
/// with the converted type. The type converter handles the actual type mapping.
struct ConvertMemRefCastNarrowType final
    : OpConversionPattern<memref::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to convert memref type: {0}", op.getType()));
    }
    if (newTy == op.getType())
      return failure();

    rewriter.replaceOpWithNewOp<memref::CastOp>(op, newTy,
                                                 adaptor.getSource());
    return success();
  }
};

/// Converts vector.load on multi-dimensional sub-byte memrefs by linearizing
/// indices from the type's static strides using pure arith ops. This avoids
/// creating extract_strided_metadata on sub-byte block arguments (which would
/// produce illegal ops in the partial conversion).
struct ConvertVectorLoadNarrowType final
    : OpConversionPattern<vector::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType origMemRefTy = op.getMemRefType();
    int srcBits = origMemRefTy.getElementTypeBitWidth();
    if (srcBits >= 8)
      return failure();

    auto containerElemTy =
        cast<MemRefType>(adaptor.getBase().getType()).getElementType();
    int dstBits = containerElemTy.getIntOrFloatBitWidth();
    if (dstBits % srcBits != 0)
      return rewriter.notifyMatchFailure(op, "dstBits % srcBits != 0");

    // Require all strides to be static.
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(origMemRefTy.getStridesAndOffset(strides, offset)))
      return failure();
    if (llvm::any_of(strides, ShapedType::isDynamic))
      return rewriter.notifyMatchFailure(op, "dynamic strides not supported");
    if (!strides.empty() && strides.back() != 1)
      return rewriter.notifyMatchFailure(op, "innermost stride must be 1");

    // Require rank-1 vector result.
    if (op.getVectorType().getRank() != 1)
      return rewriter.notifyMatchFailure(op, "only rank-1 vectors supported");

    int64_t origElements = op.getVectorType().getNumElements();
    int64_t packRatio = dstBits / srcBits;
    if (origElements % packRatio != 0)
      return rewriter.notifyMatchFailure(
          op, "vector length must be a multiple of pack ratio");

    Location loc = op.getLoc();

    // Linearize indices using static strides from the type (no
    // extract_strided_metadata). Indices are relative to the memref so offset
    // is not included.
    auto overflowFlags =
        arith::IntegerOverflowFlags::nsw | arith::IntegerOverflowFlags::nuw;
    Value linearIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (auto [idx, stride] : llvm::zip(op.getIndices(), strides)) {
      Value strideVal = arith::ConstantIndexOp::create(rewriter, loc, stride);
      Value product =
          arith::MulIOp::create(rewriter, loc, idx, strideVal, overflowFlags);
      linearIdx = arith::AddIOp::create(rewriter, loc, linearIdx, product,
                                        overflowFlags);
    }

    // Pack: convert from sub-byte element index to byte-element index.
    Value ratioVal = arith::ConstantIndexOp::create(rewriter, loc, packRatio);
    Value packedIdx = arith::DivUIOp::create(rewriter, loc, linearIdx,
                                             ratioVal);

    // Load from the converted 1D memref.
    int64_t numContainerElems = origElements / packRatio;
    auto loadVecTy = VectorType::get({numContainerElems}, containerElemTy);
    Value newLoad = vector::LoadOp::create(rewriter, loc, loadVecTy,
                                           adaptor.getBase(),
                                           ValueRange{packedIdx});

    // Bitcast back to the original element type so the conversion framework
    // can handle type mapping for downstream users.
    auto origVecTy = op.getVectorType();
    Value result =
        vector::BitCastOp::create(rewriter, loc, origVecTy, newLoad);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Converts func::FuncOp to legalize internal block argument types created by
/// pipelining (cf.br/cf.cond_br blocks with sub-byte memref args). The standard
/// func legality check only considers the function signature, not internal
/// blocks.
struct ConvertFuncOpNarrowTypes final
    : OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter = getTypeConverter();
    Region &body = funcOp.getBody();

    // Convert all block argument types (entry + internal blocks).
    TypeConverter::SignatureConversion sigConversion(
        funcOp.getNumArguments());
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      SmallVector<Type> convertedTypes;
      if (failed(typeConverter->convertType(funcOp.getArgumentTypes()[i],
                                            convertedTypes)))
        return failure();
      sigConversion.addInputs(i, convertedTypes);
    }
    if (failed(rewriter.convertRegionTypes(&body, *typeConverter,
                                           &sigConversion)))
      return failure();

    // Update function type.
    SmallVector<Type> newArgTypes;
    for (Type t : funcOp.getArgumentTypes()) {
      Type converted = typeConverter->convertType(t);
      if (!converted)
        return failure();
      newArgTypes.push_back(converted);
    }
    SmallVector<Type> newResultTypes;
    for (Type t : funcOp.getResultTypes()) {
      Type converted = typeConverter->convertType(t);
      if (!converted)
        return failure();
      newResultTypes.push_back(converted);
    }
    auto newFuncType =
        FunctionType::get(funcOp.getContext(), newArgTypes, newResultTypes);
    rewriter.modifyOpInPlace(funcOp,
                             [&]() { funcOp.setType(newFuncType); });
    return success();
  }
};

struct AMDGPUEmulateNarrowTypePass final
    : impl::AMDGPUEmulateNarrowTypePassBase<AMDGPUEmulateNarrowTypePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    affine::AffineDialect, IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    auto populateAMDGPUPatterns =
        [](arith::NarrowTypeEmulationConverter &typeConverter,
           RewritePatternSet &patterns, ConversionTarget &target) {
          auto opLegalCallback = [&typeConverter](Operation *op) {
            return typeConverter.isLegal(op);
          };
          target.addDynamicallyLegalDialect<amdgpu::AMDGPUDialect>(
              opLegalCallback);
          patterns.add<ConvertRawBufferCast, ConvertGatherToLDS>(
              typeConverter, patterns.getContext());
          // Higher benefit so these fire before upstream patterns that would
          // create illegal extract_strided_metadata on sub-byte block args.
          patterns.add<ConvertReinterpretCastNarrowType,
                       ConvertMemRefCastNarrowType,
                       ConvertVectorLoadNarrowType>(
              typeConverter, patterns.getContext(), PatternBenefit(2));
          // CF structural type conversions so that cf.br/cf.cond_br operands
          // carrying sub-byte memrefs are updated when block argument types
          // are converted (pipelining creates unstructured CFG).
          cf::populateCFStructuralTypeConversionsAndLegality(
              typeConverter, patterns, target);
          // Override func::FuncOp legality to also check internal block argument
          // types. Pipelining creates unstructured CFG with cf.br/cf.cond_br
          // that carry sub-byte memref block arguments. The default legality
          // only checks the function signature which is () -> () for kernels.
          target.addDynamicallyLegalOp<func::FuncOp>(
              [&typeConverter](func::FuncOp funcOp) {
                return typeConverter.isLegal(funcOp.getFunctionType()) &&
                       typeConverter.isLegal(&funcOp.getBody());
              });
          patterns.add<ConvertFuncOpNarrowTypes>(
              typeConverter, patterns.getContext(), PatternBenefit(2));
        };
    if (failed(emulateNarrowType(getOperation(), /*disableAtomic=*/true,
                                 populateAMDGPUPatterns))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
