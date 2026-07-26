import hashlib
import struct

def double_sha256(data: bytes) -> bytes:
    """计算比特币标准的 Double SHA-256 哈希值 (SHA256(SHA256(data)))"""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def parse_block_header(header_hex: str):
    """
    解析 80 字节的比特币区块头 (Block Header)
    """
    header_bytes = bytes.fromhex(header_hex)
    if len(header_bytes) != 80:
        raise ValueError("Block Header 必须正好为 80 字节")

    # 1. Version (4 bytes, Little-Endian)
    version = struct.unpack("<I", header_bytes[0:4])[0]
    
    # 2. Previous Block Hash (32 bytes, Little-Endian，转成大端显示)
    prev_block_hash = header_bytes[4:36][::-1].hex()
    
    # 3. Merkle Root (32 bytes, Little-Endian，转成大端显示)
    merkle_root = header_bytes[36:68][::-1].hex()
    
    # 4. Timestamp (4 bytes, Little-Endian)
    timestamp = struct.unpack("<I", header_bytes[68:72])[0]
    
    # 5. Bits / Difficulty Target (4 bytes)
    bits = header_bytes[72:76].hex()
    
    # 6. Nonce (4 bytes, Little-Endian)
    nonce = struct.unpack("<I", header_bytes[76:80])[0]
    
    # 计算 Block Hash (Double SHA-256 后反转字节序)
    block_hash = double_sha256(header_bytes)[::-1].hex()
    
    return {
        "Block Hash": block_hash,
        "Version": version,
        "Prev Block Hash": prev_block_hash,
        "Merkle Root": merkle_root,
        "Timestamp": timestamp,
        "Bits (Target)": bits,
        "Nonce": nonce
    }

def parse_transaction(tx_hex: str):
    """
    解析标准的 P2PKH 比特币交易字节流
    """
    tx_bytes = bytes.fromhex(tx_hex)
    offset = 0

    # 1. Version (4 bytes)
    version = struct.unpack("<I", tx_bytes[offset:offset+4])[0]
    offset += 4

    # 2. Input Count (1 byteVarInt 简化版)
    in_count = tx_bytes[offset]
    offset += 1

    inputs = []
    for i in range(in_count):
        prev_hash = tx_bytes[offset:offset+32][::-1].hex()
        offset += 32
        prev_out_idx = struct.unpack("<I", tx_bytes[offset:offset+4])[0]
        offset += 4
        script_len = tx_bytes[offset]
        offset += 1
        script_sig = tx_bytes[offset:offset+script_len].hex()
        offset += script_len
        sequence = struct.unpack("<I", tx_bytes[offset:offset+4])[0]
        offset += 4

        inputs.append({
            "input_index": i,
            "prev_tx_hash": prev_hash,
            "prev_output_index": prev_out_idx,
            "script_sig_length": script_len,
            "script_sig (Unlocking Script)": script_sig,
            "sequence": sequence
        })

    # 3. Output Count
    out_count = tx_bytes[offset]
    offset += 1

    outputs = []
    for i in range(out_count):
        value_sats = struct.unpack("<Q", tx_bytes[offset:offset+8])[0]
        offset += 8
        script_len = tx_bytes[offset]
        offset += 1
        script_pubkey = tx_bytes[offset:offset+script_len].hex()
        offset += script_len

        outputs.append({
            "output_index": i,
            "value_satoshis": value_sats,
            "value_btc": value_sats / 100000000.0,
            "script_pubkey_length": script_len,
            "script_pubkey (Locking Script)": script_pubkey
        })

    # 4. Locktime (4 bytes)
    locktime = struct.unpack("<I", tx_bytes[offset:offset+4])[0]
    offset += 4

    # 计算 Transaction ID (TXID)
    txid = double_sha256(tx_bytes[:offset])[::-1].hex()

    return {
        "TXID": txid,
        "Version": version,
        "Input Count": in_count,
        "Inputs": inputs,
        "Output Count": out_count,
        "Outputs": outputs,
        "Locktime": locktime
    }

# ==================== 测试示例 ====================
if __name__ == "__main__":
    # 示例: 创世区块 Header 80字节 Hex 数据
    sample_header_hex = (
        "01000000"
        "0000000000000000000000000000000000000000000000000000000000000000"
        "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
        "29ab5f49"
        "ffff001d"
        "1dac2b7c"
    )

    print("--- 1. 区块头解析结果 ---")
    parsed_header = parse_block_header(sample_header_hex)
    for k, v in parsed_header.items():
        print(f"{k}: {v}")

    # 示例: 标注 P2PKH 交易 Hex
    sample_tx_hex = (
        "01000000"  # Version
        "01"        # Input Count
        "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"  # Prev Tx Hash
        "00000000"  # Prev Output Index
        "19"        # Script Sig Length (25 bytes)
        "76a9141234567890abcdef1234567890abcdef1234567888ac"  # Script Sig
        "ffffffff"  # Sequence
        "01"        # Output Count
        "00e1f50500000000"  # Value (100,000,000 Sats = 1 BTC)
        "19"        # Script Pubkey Length (25 bytes)
        "76a9141234567890abcdef1234567890abcdef1234567888ac"  # Script Pubkey
        "00000000"  # Lock time
    )

    print("\n--- 2. 交易字节数据解析结果 ---")
    parsed_tx = parse_transaction(sample_tx_hex)
    for k, v in parsed_tx.items():
        print(f"{k}: {v}")
