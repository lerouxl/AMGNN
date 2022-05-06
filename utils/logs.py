import logging


def init_logger(file_name:str) -> None:
    """
    Configure the logging library.
    logger can be now crated using `log = logging.getLogger(__name__)`
    """
    logging.basicConfig(
        filename=file_name,
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
