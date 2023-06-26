from django.apps import AppConfig
from com.mts.logger import LogManager
from com.mts.config import ConfigManager


class WebsiteConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "website"

    def ready(self):
        settings = ConfigManager().loadAllSettings("settings")
        logger = LogManager(
            root_dir=settings["Logging"]["LOG_DIR"],
            show_console=settings["Logging"]["SHOW_CONSOLE"],
            default_level=settings["Logging"]["LEVEL"],
        ).getLogger("init_tasks")
        viewLogger = LogManager().getLogger("view")
