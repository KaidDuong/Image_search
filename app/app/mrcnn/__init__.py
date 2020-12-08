
from flask import Blueprint

blueprint = Blueprint(
    'mrcnn_blueprint',
    __name__,
    url_prefix='',
    template_folder='templates',
    static_folder='static'
)
