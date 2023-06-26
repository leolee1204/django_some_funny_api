from pathlib import Path
import os,sys
from configparser import ConfigParser

class ConfigManager:

    __instance__ = None
    __ROOT_PATH = Path(os.path.dirname(os.path.abspath(sys.modules[__name__].__file__)))\
                                        .parent.parent.parent
    __DEF_CONFIG_PATH = os.path.join(__ROOT_PATH, 'resources', 'conf')
    __EXT = 'conf'
    __DEF_NAME_WITH_EXT = '{settings}.{ext}'
        
    def __new__(cls, root_dir: str=__DEF_CONFIG_PATH, ext: str='conf') :
        if not ConfigManager.__instance__:
            ConfigManager.__instance__ = object.__new__(cls)

            ConfigManager.__DEF_CONFIG_PATH = root_dir
            ConfigManager.__EXT = ext

            checkDir = os.path.join(root_dir, '')
            if not os.path.isdir(checkDir):
                raise ValueError('Directory does not exist.(%s)' %root_dir)
            else:
                ConfigManager.__ROOT_PATH = root_dir

        return ConfigManager.__instance__

    def loadAllSettings(self, filename: str) -> dict:
        __config = ConfigParser()
        __config.optionxform = str
        path = os.path.join(self.__DEF_CONFIG_PATH, self.__DEF_NAME_WITH_EXT.format(settings=filename, ext=self.__EXT))
        try:
            num = __config.read(path,encoding='utf-8')
            if (len(num) == 0):
                raise Exception('Please check Common config.(%s)' %path)
        except:
            raise 
        rtnDict = dict(__config._sections)
        for k in rtnDict:
            rtnDict[k] = dict(rtnDict[k])
        return rtnDict

    def getDBConfig(self, settings: dict) -> dict:
        try:
            mydb_settings = {k.replace('Database_',''):settings[k] for k in settings if 'Database_' in k}
            for k in mydb_settings:
                mydb_settings[k]['mode'] = mydb_settings[k]['mode'].split(',')
            return mydb_settings
        except:
            raise