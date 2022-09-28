import os
import sys
import re
import platform

import lit.util
import lit.formats

config.name = 'Julia'
config.suffixes = ['.ll','.jl']
config.test_source_root = os.path.dirname(__file__)
config.test_format = lit.formats.ShTest(True)
config.substitutions.append(('%shlibext', '.dylib' if platform.system() == 'Darwin' else '.dll' if
    platform.system() == 'Windows' else '.so'))

system_environment = ['JULIA_LOAD_PATH', 'JULIA_DEPOT_PATH', 'JULIA_PROJECT', 'JULIA_PKG_DEVDIR']
for v in system_environment:
    value = os.environ.get(v)
    if value:
        config.environment[v] = value

julia = lit_config.params['JULIA']
config.substitutions.append(('%julia', julia))

if platform.machine() == "x86_64":
    config.available_features.add('x86_64')
