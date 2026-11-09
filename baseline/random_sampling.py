"""The random-query control for the same per-instance inference pipeline."""
import sys
from method.gp_acl import main
if __name__=='__main__':
    sys.argv.extend(['--variant','random'])
    main()
