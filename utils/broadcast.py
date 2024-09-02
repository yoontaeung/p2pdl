import logging
import pickle
import socket
from p2pdl.utils.crypto import sign_data, verify_signature

def send_echo(private_key, received_model_state, trainer_sender_addr, trainer_sender_port, addr, port):
    """
    Signs the received model state and unicasts an echo message to the sender (trainer).
    trainer_sender_addr: The IP address of the trainer. The tester sends echo message to this trainer
    addr: The IP address of tester itself.
    The message always contain the address and port of itself.
    """
    try:
        # Ensure the received_model_state is serialized before signing
        serialized_state = pickle.dumps(received_model_state)
        # Generate a digital signature for the received model state
        signature = sign_data(private_key, serialized_state)

        # Prepare the echo message with the signature
        data = pickle.dumps({
            'type': 'echo',
            'signature': signature,
            'addr': addr,
            'port': port
        })
        msg_len = len(data)

        # Unicast the echo message back to the original sender
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((trainer_sender_addr, trainer_sender_port))
            s.sendall(msg_len.to_bytes(4, byteorder='big'))
            s.sendall(data)

    except (pickle.PicklingError, socket.error) as e:
        logging.error(f"Error during send_echo operation to {trainer_sender_addr}:{trainer_sender_port}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in send_echo: {e}")

def send_ready(private_key, received_model_state, sender_addr, sender_port):
    """
    Signs the received model state and unicasts an echo message to the sender (trainer).
    """
    try:
        # Ensure the received_model_state is serialized before signing
        serialized_state = pickle.dumps(received_model_state)

        # Generate a digital signature for the received model state
        signature = sign_data(private_key, serialized_state)

        # Prepare the echo message with the signature
        data = pickle.dumps({
            'type': 'ready',
            'signature': signature,
            'addr': sender_addr,
            'port': sender_port
        })
        msg_len = len(data)

        # Unicast the echo message back to the original sender
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((sender_addr, sender_port))
            s.sendall(msg_len.to_bytes(4, byteorder='big'))
            s.sendall(data)
            # logging.info(f"Sent echo to {sender_addr}:{sender_port}")

    except (pickle.PicklingError, socket.error) as e:
        logging.error(f"Error during send_echo operation to {sender_addr}:{sender_port}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in send_echo: {e}")

