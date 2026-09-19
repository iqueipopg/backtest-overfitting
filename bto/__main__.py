import sys

if len(sys.argv) > 1 and sys.argv[1] == "audit":
    from .audit import main

    main(sys.argv[2:])
else:
    from .experiment import main

    main()
