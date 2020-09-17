import logging
# import datetime

# now = datetime.datetime.now()

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level='INFO',
    datefmt='%Y-%m-%d %H:%M:%S'#,
    # filename='results/' + now.strftime('%y%m%d-%H%M%S') + '.txt',
    # filemode='a'
)

logger = logging.getLogger('')