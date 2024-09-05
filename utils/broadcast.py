import logging
import pickle
import socket
from p2pdl.utils.crypto import sign_data, verify_signature
from cryptography.hazmat.primitives.asymmetric import ec  # Add this import
from cryptography.hazmat.primitives import hashes  # Add this import

def send_echo(key_server, private_key, received_model_state, trainer_sender_addr, trainer_sender_port, addr, port):
    """
    Signs the received model state and unicasts an echo message to the sender (trainer).
    """
    try:
        # Serialize the model state before signing
        serialized_state = pickle.dumps(received_model_state)
        logging.debug(f"[{addr}:{port}] Serialized state length: {len(serialized_state)}")

        # Generate a digital signature for the serialized model state
        signature = sign_data(private_key, serialized_state)
        logging.debug(f"[{addr}:{port}] Generated signature: {signature.hex()}")

        # Fetch and log the public key from the key server
        public_key = key_server.get_key(addr, port)
        logging.debug(f"[{addr}:{port}] Public key found: {public_key}")

        # Immediately verify the signature to check correctness
        # if verify_signature(key_server, addr, port, serialized_state, signature):
        #     logging.info(f"[{addr}:{port}] Signature verification succeeded right after signing.")
        # else:
        #     logging.error(f"[{addr}:{port}] Signature verification failed right after signing.")

        # Prepare the echo message with the signature and serialized state
        data = pickle.dumps({
            'type': 'echo',
            'signature': signature,
            'addr': addr,
            'port': port,
            'serialized_state': serialized_state  # Include the serialized data for verification
        })
        msg_len = len(data)

        # Unicast the echo message back to the original sender
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((trainer_sender_addr, trainer_sender_port))
            s.sendall(msg_len.to_bytes(4, byteorder='big'))
            s.sendall(data)
            logging.debug(f"[{addr}:{port}] Echo sent to {trainer_sender_addr}:{trainer_sender_port}")

    except (pickle.PicklingError, socket.error) as e:
        logging.error(f"[{addr}:{trainer_sender_addr}] Error during send_echo operation to {trainer_sender_addr}:{trainer_sender_port}: {e}")
    except Exception as e:
        logging.error(f"[{addr}:{trainer_sender_addr}] Unexpected error in send_echo: {e}")
        
def send_ready(signature_list, testers_list, addr, port):
    """
    A trainer sends a signature list to testers
    addr: IP address of trainer
    """
    try:
        # Ensure the received_model_state is serialized before signing
        # serialized_state = pickle.dumps(signature_list)

        # Prepare the echo message with the signature
        data = pickle.dumps({
            'type': 'ready',
            'signature_list': signature_list,
            'addr': addr,
            'port': port,
        })
        msg_len = len(data)

        # Multicast the echo message back to the original sender
        for tester in testers_list:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((tester.addr, tester.port))
                s.sendall(msg_len.to_bytes(4, byteorder='big'))
                s.sendall(data)
                logging.debug(f"Sent echo to {tester.addr}:{tester.port}")

    except (pickle.PicklingError, socket.error) as e:
        logging.error(f"Error during send_error operation from {addr}:{port}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in send_echo: {e}")

