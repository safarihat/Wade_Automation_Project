import sys

if __name__ == "__main__":
    print("--- DEPRECATION WARNING ---", file=sys.stderr)
    print("This script (build_vector_store_standalone.py) is obsolete and should not be used.", file=sys.stderr)
    print("It has been replaced by a more robust and feature-rich Django management command.", file=sys.stderr)
    print("\nPlease use the following command instead:", file=sys.stderr)
    print("\n    python manage.py build_vector_store --rebuild\n", file=sys.stderr)
    print("To clean up the project, you can safely delete this file:", file=sys.stderr)
    print(f"\n    del {__file__}\n", file=sys.stderr)
    sys.exit(1)
