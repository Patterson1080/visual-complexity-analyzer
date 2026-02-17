import sys
import os

# Ensure src is in path if running directly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    from src.gui import main
    main()
