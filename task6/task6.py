import numpy as np
import tenseal as ts
from tenseal import sealapi

def main():
    # -------------------------------------------------------------
    # 1. 初始化 SEAL 底层组件与上下文
    # -------------------------------------------------------------
    parms = sealapi.EncryptionParameters(sealapi.SCHEME_TYPE.CKKS)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(sealapi.CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 60]))
    
    # 建立上下文
    context = sealapi.SEALContext(parms, True, sealapi.SEC_LEVEL_TYPE.TC128)
    
    # 初始化密钥生成器
    keygen = sealapi.KeyGenerator(context)
    secret_key = keygen.secret_key()
    
    # 实例化公钥并生成
    public_key = sealapi.PublicKey()
    keygen.create_public_key(public_key)
    
    # 实例化伽罗瓦密钥 (Galois Keys) 并生成（旋转必需）
    galois_keys = sealapi.GaloisKeys()
    keygen.create_galois_keys(galois_keys)
    
    # 初始化底层运算器、编解码器、加密器与解密器
    evaluator = sealapi.Evaluator(context)
    encoder = sealapi.CKKSEncoder(context)
    encryptor = sealapi.Encryptor(context, public_key)
    decryptor = sealapi.Decryptor(context, secret_key)
    
    scale = 2.0**40

    # -------------------------------------------------------------
    # 2. 准备数据并加密
    # -------------------------------------------------------------
    x = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ])
    
    kernel = np.array([
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0]
    ])

    print("=== 明文数据 ===")
    print("输入矩阵 (4x4):\n", x)
    print("卷积核 (3x3):\n", kernel)

    # 编码并加密 4x4 输入向量
    x_flat = x.flatten().tolist()
    plain_x = sealapi.Plaintext()
    encoder.encode(x_flat, scale, plain_x)
    
    enc_x = sealapi.Ciphertext()
    encryptor.encrypt(plain_x, enc_x)

    # -------------------------------------------------------------
    # 3. 使用原生 Evaluator + Galois Keys 自行实现卷积计算
    # -------------------------------------------------------------
    enc_acc = None
    rotate_count = 0  # 统计旋转次数

    for r in range(3):
        for c in range(3):
            w = kernel[r, c]
            
            # 【修复关键点】：跳过权重为 0 的项，防止乘以 0 导致 ciphertext is transparent 报错
            if w == 0.0:
                continue
                
            offset = r * 4 + c  # 一维相对偏移量
            
            # 1. 进行密文旋转（offset == 0 时为原密文，无需旋转）
            if offset == 0:
                enc_rot = enc_x
            else:
                enc_rot = sealapi.Ciphertext()
                evaluator.rotate_vector(enc_x, offset, galois_keys, enc_rot)
                rotate_count += 1
            
            # 2. 构造非零系数明文并进行同态乘法
            w_plain = sealapi.Plaintext()
            encoder.encode([w] * len(x_flat), scale, w_plain)
            
            enc_prod = sealapi.Ciphertext()
            evaluator.multiply_plain(enc_rot, w_plain, enc_prod)
            evaluator.rescale_to_next_inplace(enc_prod)
            
            # 3. 累加到最终结果
            if enc_acc is None:
                enc_acc = enc_prod
            else:
                evaluator.add_inplace(enc_acc, enc_prod)

    # -------------------------------------------------------------
    # 4. 解密与正确性验证
    # -------------------------------------------------------------
    plain_result = sealapi.Plaintext()
    decryptor.decrypt(enc_acc, plain_result)
    
    decrypted_vector = encoder.decode_double(plain_result)
    
    # 提取输出 2x2 结果
    fhe_result = np.array([
        [decrypted_vector[0], decrypted_vector[1]],
        [decrypted_vector[4], decrypted_vector[5]]
    ])
    
    print("\n=== 计算结果验证 ===")
    print(f"调用的底层 Evaluator 密文旋转次数: {rotate_count} 次 (理论最小值)")
    print("\n密文卷积解密结果 (2x2):\n", fhe_result)
    
    expected = np.array([
        [np.sum(x[0:3, 0:3] * kernel), np.sum(x[0:3, 1:4] * kernel)],
        [np.sum(x[1:4, 0:3] * kernel), np.sum(x[1:4, 1:4] * kernel)]
    ])
    print("\n明文标准卷积计算结果 (2x2):\n", expected)
    
    max_diff = np.max(np.abs(fhe_result - expected))
    print(f"\n最大绝对误差: {max_diff:.8f}")
    
    if np.allclose(fhe_result, expected, atol=1e-3):
        print("结论：验证成功！")

if __name__ == "__main__":
    main()
