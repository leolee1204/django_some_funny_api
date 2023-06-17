from loguru import logger
import os,sys
from typing import Union

class LogManager:
    __instance__ = None
    show_console: bool = True
    root_dir: str 
    loggers: list = []
    ext: str = 'log'
    default_level: str = 'INFO'
    filename: str = '%s_{time:YYYY-MM-DD}.%s'

    def __new__(cls,root_dir: str='./logger/files',show_console: bool=True, ext: str='log', default_level='INFO'):
        # 判断該類的屬性是否為空；對第一個對像没有被創建，我们應該調用父類的方法，為第一個對象分配空間
        if not LogManager.__instance__:
            # 把類屬性中保存的對象引用返回给python的解釋器
            LogManager.__instance__ = object.__new__(cls)
            checkDir = os.path.isdir(root_dir)
            if not checkDir:
                raise ValueError("Directory does not exist.(%s)" % root_dir)
            LogManager.root_dir = root_dir
            show_console = str(show_console)
            # if show_console is not None and show_console.lower() in ['yes', 'true', 't', '1']:
            #     LogManager.show_console = True
            # else:
            #     LogManager.show_console = False
            LogManager.show_console = True if show_console else False
            LogManager.ext = ext
            LogManager.default_level = default_level
            # 刪除預設handle
            logger.remove()
        return LogManager.__instance__

    def make_filter(self, name: str):
        def filter(record):
            return record["extra"].get("name") == name

        return filter
    # extra[name] :getLogger('view') module:files line:405 level:INFO message:xxxx start
    def getLogger(self, name: str,
                        format: str='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> {extra[name]}[<light-blue>{module}</light-blue>:{line}] <lvl>{level}</lvl> {message}',
                        level: Union[str, None]=None,
                        rotation: str='00:00',
                        set_root_dir: Union[str, None]=None,
                        set_show_console: Union[bool, None]=None):

        level = level if level else self.default_level
        if name not in self.loggers:
            self.loggers.append(name)
            logger.add(
                os.path.join(
                    self.root_dir if not set_root_dir else set_root_dir,
                    self.filename % (name, self.ext),
                ),
                format=format,
                level=level,
                rotation=rotation,
                filter=self.make_filter(name),
            )
            if set_show_console is None:
                if self.show_console:
                    logger.add(
                        sys.stderr,
                        format=format,
                        level=level,
                        filter=self.make_filter(name),
                    )
            else:
                if set_show_console:
                    logger.add(
                        sys.stderr,
                        format=format,
                        level=level,
                        filter=self.make_filter(name),
                    )
        return logger.bind(name=name)
