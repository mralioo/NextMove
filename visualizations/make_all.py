"""
Run all plot scripts in one go (default paths).

    python -m visualizations.make_all
"""
import sys

from visualizations.event_effect import plot as event_plot
from visualizations.raw_flow import plot as raw_plot
from visualizations.weather_effect import plot as weather_plot


def main():
    for module in (raw_plot, weather_plot, event_plot):
        print(f"\n== {module.__name__}")
        sys.argv = [module.__name__]          # each script parses its own (default) arguments
        module.main()


if __name__ == "__main__":
    main()
