import sys
from anylabeling.app import main

# Принудительно открываем папку сжатых картинок, обходя баг Windows
sys.argv = ['anylabeling', 'compressed_dataset']
main()
