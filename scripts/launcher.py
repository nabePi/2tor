import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from twotor.app import main

    main()
