from distutils.core import setup

setup(name="ctd_toolbox",
      version="0.2",
      scripts=["dbd2asc.py"],
      description='Module for calculating CTD related stuff',
      author='Lucas Merckelbach',
      author_email='lucas.merckelbach@hzg.de',
      url='http://dockserver0.hzg.de/glider/current.php',
      packages = ["profiles"],
#     py_modules = ["dataconverters"],
      license='GPL',
      platforms='UNIX',
      )
