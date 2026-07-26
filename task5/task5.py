import numpy as np
import tenseal as ts

def create_fhe_context():
    """初始化 CKKS 同态加密上下文并生成公私钥"""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

def im2col_2d(input_matrix, kernel_size=(3, 3)):
    """将输入图像切片展平为矩阵，便于通过向量内积实现卷积"""
    H, W = input_matrix.shape
    kh, kw = kernel_size
    out_h, out_w = H - kh + 1, W - kw + 1
    
    blocks = []
    for r in range(out_h):
        for c in range(out_w):
            patch = input_matrix[r:r+kh, c:c+kw]
            blocks.append(patch.flatten())
    return np.array(blocks)  # 形状为 (out_h * out_w, kh * kw) -> (4, 9)

def main():
    # -------------------------------------------------------------
    # 1. 准备数据
    # -------------------------------------------------------------
    # 输入矩阵 (4x4)
    x = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ])
    
    # 3x3 卷积核
    kernel = np.array([
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0]
    ])
    
    print("=== 明文数据 ===")
    print("输入矩阵 (4x4):\n", x)
    print("卷积核 (3x3):\n", kernel)
    
    # -------------------------------------------------------------
    # 2. 计算明文卷积基准结果 (Ground Truth)
    # -------------------------------------------------------------
    out_h, out_w = x.shape[0] - kernel.shape[0] + 1, x.shape[1] - kernel.shape[1] + 1
    expected_output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            expected_output[i, j] = np.sum(x[i:i+3, j:j+3] * kernel)
            
    print("\n明文标准卷积计算结果 (2x2):\n", expected_output)

    # -------------------------------------------------------------
    # 3. 同态加密运算
    # -------------------------------------------------------------
    context = create_fhe_context()
    
    # 利用 im2col 提取滑动窗口，将 4x4 输入转换为 4 个 9 维向量
    x_patches = im2col_2d(x, (3, 3)) # shape: (4, 9)
    kernel_flat = kernel.flatten().tolist() # shape: (9,)
    
    enc_outputs = []
    
    # 对每个局部窗口数据进行加密并与明文卷积核求点积
    for i, patch in enumerate(x_patches):
        # 加密输入的局部窗口向量
        enc_patch = ts.ckks_vector(context, patch.tolist())
        
        # 密文与明文卷积核计算点积 (同态乘法 + 累加)
        # result_enc 是一个包含单个卷积结果的密文对象
        enc_conv_val = enc_patch.dot(kernel_flat)
        enc_outputs.append(enc_conv_val)

    # -------------------------------------------------------------
    # 4. 解密与正确性验证
    # -------------------------------------------------------------
    decrypted_results = [enc_val.decrypt()[0] for enc_val in enc_outputs]
    fhe_output = np.array(decrypted_results).reshape(out_h, out_w)
    
    print("\n=== FHE 密文解密结果 ===")
    print(fhe_output)
    
    # 验证误差 (CKKS 为近似计算，允许存在小微误差)
    max_diff = np.max(np.abs(fhe_output - expected_output))
    print(f"\n最大绝绝对误差: {max_diff:.8f}")
    
    if np.allclose(fhe_output, expected_output, atol=1e-3):
        print("结论: 密文卷积计算成功，验证正确！")
    else:
        print("结论: 验证失败，结果超出容忍误差范围。")

if __name__ == "__main__":
    main()
