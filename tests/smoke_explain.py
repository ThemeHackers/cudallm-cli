import sys
import json
from src import profile_explain


def main():
 
    r1 = profile_explain.explain_nsys("")
    if r1.get('status') != 'no_data':
        print('explain_nsys empty input failed', r1)
        sys.exit(2)

  
    r2 = profile_explain.explain_ncu('nonexistent.csv')
    if r2.get('status') != 'no_data':
        print('explain_ncu missing csv failed', r2)
        sys.exit(3)


    rc = profile_explain.explain_combined(None, None)
    if 'combined' not in rc:
        print('explain_combined structure unexpected', rc)
        sys.exit(4)

    print('SMOKE OK')


if __name__ == '__main__':
    main()
