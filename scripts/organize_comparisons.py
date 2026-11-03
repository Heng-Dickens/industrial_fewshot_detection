"""Refresh method-first comparison indexes without moving original artifacts."""
from summarize_results import refresh


def main():
    refresh()
    print('Comparison indexes updated under outputs/<method>/comparisons.')


if __name__ == '__main__':
    main()
