// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "platform.h"

namespace zvec {
namespace ailego {
namespace internal {

/*! Cpu Features
 */
class CpuFeatures {
 public:
  //! 16-bit FP conversions
  static bool F16C(void);

  //! Multimedia Extensions
  static bool MMX(void);

  //! Streaming SIMD Extensions
  static bool SSE(void);

  //! Streaming SIMD Extensions 2
  static bool SSE2(void);

  //! Streaming SIMD Extensions 3
  static bool SSE3(void);

  //! Supplemental Streaming SIMD Extensions 3
  static bool SSSE3(void);

  //! Streaming SIMD Extensions 4.1
  static bool SSE4_1(void);

  //! Streaming SIMD Extensions 4.2
  static bool SSE4_2(void);

  //! Advanced Vector Extensions
  static bool AVX(void);

  //! Advanced Vector Extensions 2
  static bool AVX2(void);

  //! AVX-512 Foundation
  static bool AVX512F(void);

  //! AVX-512 DQ (Double/Quad granular) Instructions
  static bool AVX512DQ(void);

  //! AVX-512 Prefetch
  static bool AVX512PF(void);

  //! AVX-512 Exponential and Reciprocal
  static bool AVX512ER(void);

  //! AVX-512 Conflict Detection
  static bool AVX512CD(void);

  //! AVX-512 BW (Byte/Word granular) Instructions
  static bool AVX512BW(void);

  //! AVX-512 VL (128/256 Vector Length) Extensions
  static bool AVX512VL(void);

  //! AVX-512 Integer Fused Multiply-Add instructions
  static bool AVX512_IFMA(void);

  //! AVX512 Vector Bit Manipulation instructions
  static bool AVX512_VBMI(void);

  //! Additional AVX512 Vector Bit Manipulation Instructions
  static bool AVX512_VBMI2(void);

  //! Vector Neural Network Instructions
  static bool AVX512_VNNI(void);

  //! Support for VPOPCNT[B,W] and VPSHUF-BITQMB instructions
  static bool AVX512_BITALG(void);

  //! POPCNT for vectors of DW/QW
  static bool AVX512_VPOPCNTDQ(void);

  //! AVX-512 Neural Network Instructions
  static bool AVX512_4VNNIW(void);

  //! AVX-512 Multiply Accumulation Single precision
  static bool AVX512_4FMAPS(void);

  //! AVX-512 FP16 instructions
  static bool AVX512_FP16(void);

  //! CMPXCHG8 instruction
  static bool CX8(void);

  //! CMPXCHG16B instruction
  static bool CX16(void);

  //! PCLMULQDQ instruction
  static bool PCLMULQDQ(void);

  //! Carry-Less Multiplication Double Quadword
  static bool VPCLMULQDQ(void);

  //! CMOV instructions (plus FCMOVcc, FCOMI with FPU)
  static bool CMOV(void);

  //! MOVBE instruction
  static bool MOVBE(void);

  //! Enhanced REP MOVSB/STOSB instructions
  static bool ERMS(void);

  //! POPCNT instruction
  static bool POPCNT(void);

  //! XSAVE/XRSTOR/XSETBV/XGETBV instructions
  static bool XSAVE(void);

  //! Fused multiply-add
  static bool FMA(void);

  //! ADCX and ADOX instructions
  static bool ADX(void);

  //! Galois Field New Instructions
  static bool GFNI(void);

  //! AES instructions
  static bool AES(void);

  //! Vector AES
  static bool VAES(void);

  //! RDSEED instruction
  static bool RDSEED(void);

  //! RDRAND instruction
  static bool RDRAND(void);

  //! SHA1/SHA256 Instruction Extensions
  static bool SHA(void);

  //! 1st group bit manipulation extensions
  static bool BMI1(void);

  //! 2nd group bit manipulation extensions
  static bool BMI2(void);

  //! CLFLUSH instruction
  static bool CLFLUSH(void);

  //! CLFLUSHOPT instruction
  static bool CLFLUSHOPT(void);

  //! CLWB instruction
  static bool CLWB(void);

  //! RDPID instruction
  static bool RDPID(void);

  //! Onboard FPU
  static bool FPU(void);

  //! Hyper-Threading
  static bool HT(void);

  //! Hardware virtualization
  static bool VMX(void);

  // ！Running on a hypervisor
  static bool HYPERVISOR(void);

  //! Intrinsics of compiling
  static const char *Intrinsics(void);

 private:
  struct CpuFlags {
    //! Constructor
    CpuFlags(void);

    //! Members
    uint32_t L1_ECX;
    uint32_t L1_EDX;
    uint32_t L7_EBX;
    uint32_t L7_ECX;
    uint32_t L7_EDX;
  };

  //! Static Members
  static CpuFlags flags_;
};

}  // namespace internal
}  // namespace ailego
}  // namespace zvec
