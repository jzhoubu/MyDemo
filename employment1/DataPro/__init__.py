import logging
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s%(levelname)s %(message)s',
                datefmt='%d %b %Y %H:%M:%S',
                filename='test.log',
                filemode='wb')
logger = logging.getLogger("DataProcess")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',"%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.handlers=[]
logger.addHandler(ch)
