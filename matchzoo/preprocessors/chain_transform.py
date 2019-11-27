"""Wrapper function organizes a number of transform functions."""
import typing
import functools

from .units.unit import Unit


def chain_transform(units: typing.List[Unit]) -> typing.Callable:
    """
    Compose unit transformations into a single function.

    :param units: List of :class:`matchzoo.StatelessUnit`.
    """

    @functools.wraps(chain_transform)
    def wrapper(arg):
        """Wrapper function of transformations composition."""
        for unit in units:
            arg = unit.transform(arg)
        return arg

    unit_names = ' => '.join(unit.__class__.__name__ for unit in units)
    wrapper.__name__ += ' of ' + unit_names
    return wrapper


class ChainTransform(object):
    """Compose unit transformations into a single object."""

    def __init__(self, units: typing.List[Unit]):
        """
        Compose unit transformations into a single object.

        :param units: List of :class:`matchzoo.StatelessUnit`.
        """
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
