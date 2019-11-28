"""Wrapper function organizes a number of transform functions."""
import typing

from .units.unit import Unit


class ChainTransform(object):
    """Compose unit transformations into a single object."""

    def __init__(self, units: typing.Union[typing.List[Unit], Unit]):
        """
        Compose unit transformations into a single object.

        :param units: List of :class:`matchzoo.StatelessUnit`.
        """
        if isinstance(units, Unit):
            units = [units]

        self.units = units

    def __call__(self, text):
        """Perform transformations on text."""
        for unit in self.units:
            text = unit.transform(text)
        return text

    def __str__(self):
        """Convert unit names into the string."""
        unit_names = ' => '.join(unit.__class__.__name__
                                 for unit in self.units)
        return 'Chain Transform of ' + unit_names

    @property
    def __name__(self):
        """Use the unit names as the class name."""
        return self.__str__()
