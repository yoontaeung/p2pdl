from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import pickle

# p2pdl/utils/key_server.py

class KeyServer:
    def __init__(self):
        self.public_key_store = {}

    def register_key(self, addr, port, public_key):
        """
        Register a public key for a given node.
        """
        node_id = (addr, port)
        if node_id not in self.public_key_store:
            self.public_key_store[node_id] = public_key
            print(f"Public key registered for {addr}:{port}")
        else:
            print(f"Public key already exists for {addr}:{port}")

    def get_key(self, addr, port):
        """
        Retrieve the public key for a given node.
        """
        return self.public_key_store.get((addr, port))

    def get_all_keys(self):
        """
        Retrieve all public keys.
        """
        return self.public_key_store
    
def generate_key_pair():
    """
    Generate an RSA key pair.
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    return private_key, public_key

def sign_data(private_key, data):
    """
    Sign the data using the provided private key.
    """
    signature = private_key.sign(
        pickle.dumps(data),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return signature

def verify_signature(key_server, addr, port, serialized_data, signature):
    """
    Verify the signature of the serialized data using the sender's public key retrieved from the key server.
    """
    public_key = key_server.get_key(addr, port)
    
    if not public_key:
        print(f"Public key for {addr}:{port} not found.")
        return False

    try:
        public_key.verify(
            signature,
            serialized_data,  # Use the serialized data directly without re-serializing
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except Exception as e:
        print(f"Signature verification failed for {addr}:{port}: {e}")
        return False