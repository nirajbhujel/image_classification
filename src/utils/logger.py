import sys
import logging

def create_logger(log_file_path, level=logging.INFO):

    # this only works if logging is not imported before calling this function 
    # logging.basicConfig(file_name=log_file_path,
    #                     format="%(asctime)s [%(levelname)s]  %(message)s", 
    #                     level=level,
    #                     datefmt='%Y-%m-%d %H:%M:%S',)

    rootLogger = logging.getLogger('')
    rootLogger.handlers.clear()
    
    # NOTE!! Settting console handler to DEBUG causes logger.info to print twice in console -> bug: creater_logger is called twice, and handlers were added. Clear handler before adding new.
    consoleFormat = logging.Formatter("%(message)s")
    consoleHandler = logging.StreamHandler(sys.stdout)
    # consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(consoleFormat)
    rootLogger.addHandler(consoleHandler)

    fileFormat = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    fileHandler = logging.FileHandler(log_file_path)
    # fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(fileFormat)
    rootLogger.addHandler(fileHandler)

    return rootLogger


if __name__ == '__main__':
    # logger = create_logger('../log.txt')
    logger.info("Hello")