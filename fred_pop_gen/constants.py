from enum import Enum


class Enrollment(Enum):
    """Represents person school enrollment status."""

    PUBLIC = 0
    PRIVATE = 1
    NOT_ENROLLED = 2
    NOT_SCHOOL_AGED = 3
