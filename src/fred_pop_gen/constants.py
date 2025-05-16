from enum import Enum


class Enrollment(Enum):
    """Represents person school enrollment status."""

    PUBLIC = 0
    PRIVATE = 1
    NOT_ENROLLED = 2


class Grade(Enum):
    """Represents school grade levels."""

    PREK = 0
    K = 1
    FIRST = 2
    SECOND = 3
    THIRD = 4
    FOURTH = 5
    FIFTH = 6
    SIXTH = 7
    SEVENTH = 8
    EIGHTH = 9
    NINTH = 10
    TENTH = 11
    ELEVENTH = 12
    TWELFTH = 13

    @staticmethod
    def range(lower_bound: "Grade", upper_bound: "Grade") -> set["Grade"]:
        return {
            Grade(value) for value in range(lower_bound.value, upper_bound.value + 1)
        }
