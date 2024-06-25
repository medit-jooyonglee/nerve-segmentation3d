import os.path
import sys

sys.path.append(os.path.dirname(os.path.dirname('__file__')))

# from
# import importlib
# importlib.import_module('name')

python -c "import sys; sys.path.append('/your/script/path'); import yourscript; yourscript.yourfunction()"