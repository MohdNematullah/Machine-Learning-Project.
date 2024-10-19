import logging
import os
from datetime import datetime

# Generate the log file name with a timestamp
Log_File = f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"

# Create the "logs" directory if it doesn't exist
Log_Path = os.path.join(os.getcwd(), "logs",Log_File)
os.makedirs(Log_Path, exist_ok=True)

# Set the full path to the log file
Log_File_Path = os.path.join(Log_Path, Log_File)

# Set up the logging configuration
logging.basicConfig(
    filename=Log_File_Path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Optional: Use a logger instance for more flexibility
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Logging has started")

