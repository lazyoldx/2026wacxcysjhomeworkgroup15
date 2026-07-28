#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <bit>
#include <string>

// ===================================================================
// 跨平台指令集头文件引入与条件编译检测
// ===================================================================
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h> // AVX2 / AVX-512
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>  // ARM NEON
#endif

// 循环左移与字节序转换助手
#define ROTL32(x, n) std::rotl(static_cast<uint32_t>(x), (n))

inline uint32_t bswap32(uint32_t x) {
#if defined(_MSC_VER)
    return _byteswap_ulong(x);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(x);
#else
    return (x >> 24) | ((x >> 8) & 0x0000FF00) | ((x << 8) & 0x00FF0000) | (x << 24);
#endif
}

// SM3 算子定义
#define P0(x) ((x) ^ ROTL32((x), 9) ^ ROTL32((x), 17))
#define P1(x) ((x) ^ ROTL32((x), 15) ^ ROTL32((x), 23))

#define FF0(x, y, z) ((x) ^ (y) ^ (z))
#define FF1(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))

#define GG0(x, y, z) ((x) ^ (y) ^ (z))
#define GG1(x, y, z) (((x) & (y)) | ((~x) & (z)))

// 国家标准 GM/T 0004-2012 规定的正确初始状态 IV 常量
static const uint32_t SM3_IV[8] = {
    0x7380166F, 0x4914B2B9, 0x172442D7, 0xDA8A0600,
    0xA96F30BC, 0x163138AA, 0xE38DEE4D, 0xB0FB0E4E
};

// ===================================================================
// 核心优化 1：SIMD 消息扩展 (Message Expansion) - 架构分发
// ===================================================================

// 1.1 x86 AVX-512 极致优化 (512 位宽 ZMM 寄存器)
#if defined(__AVX512F__) && defined(__AVX512BW__)
void sm3_expand_avx512(const uint8_t block[64], uint32_t W[68], uint32_t W1[64]) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(block));
    __m512i shuf_mask = _mm512_set_epi8(
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3
    );
    v = _mm512_shuffle_epi8(v, shuf_mask);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(W), v);

    for (int j = 16; j < 68; ++j) {
        uint32_t tmp = W[j - 16] ^ W[j - 9] ^ ROTL32(W[j - 3], 15);
        W[j] = P1(tmp) ^ ROTL32(W[j - 13], 7) ^ W[j - 6];
    }

    for (int j = 0; j < 64; j += 16) {
        __m512i w_j = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(W + j));
        __m512i w_j4 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(W + j + 4));
        __m512i w1_j = _mm512_xor_si512(w_j, w_j4);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(W1 + j), w1_j);
    }
}
#endif

// 1.2 x86 AVX2 优化 (256 位宽 YMM 寄存器)
#if defined(__AVX2__) || defined(_M_X64) || defined(_M_IX86)
void sm3_expand_avx2(const uint8_t block[64], uint32_t W[68], uint32_t W1[64]) {
    const uint32_t* p = reinterpret_cast<const uint32_t*>(block);
    for (int i = 0; i < 16; ++i) {
        W[i] = bswap32(p[i]);
    }

    for (int j = 16; j < 68; ++j) {
        uint32_t tmp = W[j - 16] ^ W[j - 9] ^ ROTL32(W[j - 3], 15);
        W[j] = P1(tmp) ^ ROTL32(W[j - 13], 7) ^ W[j - 6];
    }

    for (int j = 0; j < 64; j += 8) {
        __m256i w_j = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(W + j));
        __m256i w_j4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(W + j + 4));
        __m256i w1_j = _mm256_xor_si256(w_j, w_j4);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(W1 + j), w1_j);
    }
}
#endif

// 1.3 ARM64 NEON 优化 (128 位宽)
#if defined(__aarch64__) || defined(_M_ARM64)
void sm3_expand_neon(const uint8_t block[64], uint32_t W[68], uint32_t W1[64]) {
    const uint32_t* p = reinterpret_cast<const uint32_t*>(block);
    for (int i = 0; i < 16; ++i) {
        W[i] = bswap32(p[i]);
    }

    for (int j = 16; j < 68; ++j) {
        uint32_t tmp = W[j - 16] ^ W[j - 9] ^ ROTL32(W[j - 3], 15);
        W[j] = P1(tmp) ^ ROTL32(W[j - 13], 7) ^ W[j - 6];
    }

    for (int j = 0; j < 64; j += 4) {
        uint32x4_t w_j = vld1q_u32(W + j);
        uint32x4_t w_j4 = vld1q_u32(W + j + 4);
        uint32x4_t w1_j = veorq_u32(w_j, w_j4);
        vst1q_u32(W1 + j, w1_j);
    }
}
#endif

// 标量兜底实现
void sm3_expand_scalar(const uint8_t block[64], uint32_t W[68], uint32_t W1[64]) {
    const uint32_t* p = reinterpret_cast<const uint32_t*>(block);
    for (int i = 0; i < 16; ++i) {
        W[i] = bswap32(p[i]);
    }
    for (int j = 16; j < 68; ++j) {
        uint32_t tmp = W[j - 16] ^ W[j - 9] ^ ROTL32(W[j - 3], 15);
        W[j] = P1(tmp) ^ ROTL32(W[j - 13], 7) ^ W[j - 6];
    }
    for (int j = 0; j < 64; ++j) {
        W1[j] = W[j] ^ W[j + 4];
    }
}

// ===================================================================
// 核心优化 2：通用寄存器 (GPR) 高强度压缩迭代
// ===================================================================
void sm3_compress_gpr(uint32_t state[8], const uint32_t W[68], const uint32_t W1[64]) {
    register uint32_t A = state[0], B = state[1], C = state[2], D = state[3];
    register uint32_t E = state[4], F = state[5], G = state[6], H = state[7];

    for (int j = 0; j < 16; ++j) {
        uint32_t T_j = 0x79CC4519;
        uint32_t SS1 = ROTL32(ROTL32(A, 12) + E + ROTL32(T_j, j), 7);
        uint32_t SS2 = SS1 ^ ROTL32(A, 12);
        uint32_t TT1 = FF0(A, B, C) + D + SS2 + W1[j];
        uint32_t TT2 = GG0(E, F, G) + H + SS1 + W[j];

        D = C; C = ROTL32(B, 9); B = A; A = TT1;
        H = G; G = ROTL32(F, 19); F = E; E = P0(TT2);
    }

    for (int j = 16; j < 64; ++j) {
        uint32_t T_j = 0x7A879D8A;
        uint32_t SS1 = ROTL32(ROTL32(A, 12) + E + ROTL32(T_j, j % 32), 7);
        uint32_t SS2 = SS1 ^ ROTL32(A, 12);
        uint32_t TT1 = FF1(A, B, C) + D + SS2 + W1[j];
        uint32_t TT2 = GG1(E, F, G) + H + SS1 + W[j];

        D = C; C = ROTL32(B, 9); B = A; A = TT1;
        H = G; G = ROTL32(F, 19); F = E; E = P0(TT2);
    }

    state[0] ^= A; state[1] ^= B; state[2] ^= C; state[3] ^= D;
    state[4] ^= E; state[5] ^= F; state[6] ^= G; state[7] ^= H;
}

// Block 调度分发
void sm3_process_block(uint32_t state[8], const uint8_t block[64]) {
    alignas(64) uint32_t W[68];
    alignas(64) uint32_t W1[64];

#if defined(__AVX512F__) && defined(__AVX512BW__)
    sm3_expand_avx512(block, W, W1);
#elif defined(__AVX2__) || defined(_M_X64) || defined(_M_IX86)
    sm3_expand_avx2(block, W, W1);
#elif defined(__aarch64__) || defined(_M_ARM64)
    sm3_expand_neon(block, W, W1);
#else
    sm3_expand_scalar(block, W, W1);
#endif

    sm3_compress_gpr(state, W, W1);
}

// 完整 Hash 计算封装
std::string sm3_hash_hex(const uint8_t* msg, size_t len) {
    uint32_t state[8];
    std::memcpy(state, SM3_IV, sizeof(SM3_IV));

    size_t processed = 0;
    while (processed + 64 <= len) {
        sm3_process_block(state, msg + processed);
        processed += 64;
    }

    uint8_t block[64] = { 0 };
    size_t rem = len - processed;
    std::memcpy(block, msg + processed, rem);
    block[rem] = 0x80;

    if (rem >= 56) {
        sm3_process_block(state, block);
        std::memset(block, 0, 64);
    }

    uint64_t bit_len = static_cast<uint64_t>(len) * 8;
    for (int i = 0; i < 8; ++i) {
        block[63 - i] = static_cast<uint8_t>(bit_len >> (i * 8));
    }
    sm3_process_block(state, block);

    char hex_str[65] = { 0 };
    for (int i = 0; i < 8; ++i) {
        snprintf(hex_str + i * 8, 9, "%08x", state[i]);
    }
    return std::string(hex_str);
}

// ===================================================================
// 测试用例主函数
// ===================================================================
int main() {
    const std::string input = "abc";
    const std::string expected_hash = "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0";

    std::string actual_hash = sm3_hash_hex(reinterpret_cast<const uint8_t*>(input.c_str()), input.length());

    std::cout << "Input    : " << input << std::endl;
    std::cout << "Actual   : " << actual_hash << std::endl;
    std::cout << "Expected : " << expected_hash << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    if (actual_hash == expected_hash) {
        std::cout << "TEST RESULT: [ PASS ]" << std::endl;
    }
    else {
        std::cout << "TEST RESULT: [ FAIL ]" << std::endl;
    }

    return 0;
}