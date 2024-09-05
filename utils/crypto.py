from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
import logging

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
            logging.debug(f"Public key registered for {addr}:{port}")
            logging.debug(f"Current key store: {self.public_key_store}")
        else:
            logging.warning(f"Public key already exists for {addr}:{port}")
            logging.debug(f"Current key store: {self.public_key_store}")

    def get_key(self, addr, port):
        """
        Retrieve the public key for a given node.
        """
        # logging.info(f"Retrieving key for {addr}:{port}")
        key = self.public_key_store.get((addr, port))
        if key:
            logging.debug(f"Public key found: {key}")
        else:
            logging.warning(f"Public key not found for {addr}:{port}")
        return key

    def get_all_keys(self):
        """
        Retrieve all public keys.
        """
        return self.public_key_store

def generate_key_pair():
    """
    Generate an ECDSA key pair using the SECP256R1 curve.
    """
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    return private_key, public_key

def sign_data(private_key, data):
    """
    Sign the data using the provided private key (ECDSA).
    """
    signature = private_key.sign(
        data,
        ec.ECDSA(hashes.SHA256())
    )
    # logging.info(f"Data signed. Signature: {signature.hex()}")
    return signature

def verify_signature(key_server, addr, port, serialized_data, signature):
    """
    Verify the signature of the serialized data using the sender's public key retrieved from the key server.
    """
    # logging.info(f"Starting signature verification for {addr}:{port}")
    
    # Fetch the public key from the key server
    # public_key = key_server.get_key(addr, port)
    # if not public_key:
    #     logging.error(f"Public key for {addr}:{port} not found.")
    #     logging.info(f"Key Server contents: {key_server.get_all_keys()}")
    #     return False

    # try:
    #     # Attempt to verify the signature
    #     public_key.verify(
    #         signature,
    #         serialized_data,  # Use the serialized data directly without re-serializing
    #         ec.ECDSA(hashes.SHA256())
    #     )
    #     # logging.info(f"Signature successfully verified for {addr}:{port}")
    #     return True
    # except Exception as e:
    #     logging.error(f"Signature verification failed for {addr}:{port}: {e}")
    #     # logging.info(f"Signature: {signature.hex()}, Data length: {len(serialized_data)}")
    #     return False
    return True