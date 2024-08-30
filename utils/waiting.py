import threading
import logging

def wait_for_models(received_models, expected_count, timeout=30):
    """
    Wait until the number of received models equals the expected count or until the timeout occurs.

    :param received_models: The list of received models.
    :param expected_count: The number of expected models.
    :param timeout: Maximum time to wait (in seconds).
    :return: True if all models were received within the timeout, False otherwise.
    """
    start_time = threading.Event().wait(0)  # Get the start time
    while len(received_models) < expected_count:
        if threading.Event().wait(1):  # Wait for 1 second increments
            break
        if threading.Event().wait(0) - start_time > timeout:  # Check if the timeout is exceeded
            logging.warning(f"Timeout exceeded while waiting for model updates. Proceeding with aggregation.")
            return False

    return True